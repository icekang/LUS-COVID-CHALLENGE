import functools
import os
import pathlib
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Subset

from rich.console import Console
from rich.markdown import Markdown

from deepchest import preprocessing, utils

SITE_MAPPING = OrderedDict(
    [
        ("<PAD>", 0),
        ("QAID", 1),
        ("QAIG", 2),
        ("QASD", 3),
        ("QASG", 4),
        ("QLD", 5),
        ("QLG", 6),
        ("QPID", 7),
        ("QPIG", 8),
        ("QPSD", 9),
        ("QPSG", 10),
        ("QPG", 11),
    ]
)


@dataclass
class ImageInfo:
    patient_id: int
    site: str

    @staticmethod
    def from_filename(filename: str):
        match = re.match(r"^(\d+)_([A-Z]+)(_\d+)?(\..*)?\.png$", filename)
        if match is None:
            raise ValueError(f"Could not parse '{filename}' into an ImageInfo.")
        patient_id = int(match.group(1))
        site = match.group(2)
        return ImageInfo(
            patient_id,
            site,
        )


class LungUltrasoundPatientDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_directory: str,
        labels: Optional[Dict[int, int]] = None,
    ):
        if not pathlib.Path(images_directory).exists():
            raise FileNotFoundError("Images directory not found. Path given: "+images_directory)
        self.images_directory = pathlib.Path(images_directory)
        self.labels = labels

        self.images_list = os.listdir(self.images_directory)
        if labels:
            self.length = len(labels)
        else:
            self.length = len(set(ImageInfo.from_filename(f).patient_id for f in self.images_list))

    def __len__(self):
        return self.length

    def get_sites_counter(self) -> Counter:
        return Counter([ImageInfo.from_filename(f).site for f in self.images_list])

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, i):
        image_names = [f for f in self.images_list if f.startswith(f"{i}_")]
        num_patient_images = len(image_names)
        sites = [ImageInfo.from_filename(f).site for f in image_names]
        sites = torch.LongTensor([SITE_MAPPING[x] for x in sites])

        images = []
        for x in image_names:
            image = Image.open(self.images_directory / x).convert("RGB")
            images.append(image)

        mask = torch.ones(num_patient_images)

        patient_dict = {
            "images": images,
            "sites": sites,
            "mask": mask,
        }

        if self.labels is not None:
            patient_dict["label"] = self.labels[i]

        return patient_dict


def collate_fn(samples, image_transform, preprocessing_fn=None):
    samples_per_key = {key: [sample[key] for sample in samples] for key in samples[0]}
    batch = {}
    if "label" in samples_per_key:
        batch["label"] = torch.tensor(samples_per_key["label"])

    # Transform each image of site of each patient
    # and stack them together for each patient.
    samples_per_key["images"] = [
        torch.stack(
            [torch.FloatTensor(np.array(image_transform(im))) for im in patient_images], dim=0
        )
        for patient_images in samples_per_key["images"]
    ]

    # Pad with zeros and stack samples
    max_length = max(map(len, samples_per_key["mask"]))
    for k in ["images", "sites", "mask"]:
        batch[k] = torch.stack(
            [utils.pad_dim_with_zeros(t, 0, max_length) for t in samples_per_key[k]], dim=0
        )

    # Preprocessing
    if preprocessing_fn:
        batch["images"], batch["sites"], batch["mask"] = preprocessing_fn(
            batch["images"], batch["sites"], batch["mask"]
        )

    return batch


def get_data_loaders(config) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    if not pathlib.Path(config.labels_file).exists():
        with open(pathlib.Path(__file__).parent / "../dataset/README.md") as md:
            markdown = Markdown(md.read())
        Console().print(markdown)
        raise FileNotFoundError("Label file not found. Path given: "+config.labels_file)
    labels_df = pd.read_csv(config.labels_file, index_col=0)
    indices = labels_df.index
    labels = labels_df.values.flatten()
    labels_dict = dict(zip(indices, labels))

    folds = utils.split_k_folds(indices, labels, k=config.num_folds, random_state=config.random_state)
    train_folds_indices = list(range(0, config.num_folds))

    test_indices = folds[config.test_fold_index]
    train_folds_indices.remove(config.test_fold_index)

    if config.use_validation_split:
        validation_fold_index = (
            config.test_fold_index + config.delta_from_test_index_to_validation_index
        ) % config.num_folds
        validation_indices = folds[validation_fold_index]
        train_folds_indices.remove(validation_fold_index)
    else:
        validation_indices = []

    train_folds = [fold for fold_idx, fold in enumerate(folds) if fold_idx in train_folds_indices]
    train_indices = np.concatenate(train_folds)

    if config.export_folds_indices_file:
        serie_train = pd.Series(train_indices, name='train_indices')
        serie_test = pd.Series(test_indices, name='test_indices')
        serie_valid = pd.Series(validation_indices, name='validation_indices')
        df_indices = pd.concat([serie_train,serie_test,serie_valid], axis=1)
        df_indices.to_csv(config.export_folds_indices_file,index=False)

    transform_vanilla = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    transform_with_augmentation = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                224,
                scale=(0.75, 0.95),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
            ),
            transforms.ToTensor(),
        ]
    )

    dataset = LungUltrasoundPatientDataset(
        images_directory=config.images_directory,
        labels=labels_dict,
    )

    label_names = utils.get_label_names(config.labels_file)
    utils.show_splits_info(
        train_indices, test_indices, validation_indices, dataset=dataset, label_names=label_names
    )

    train_pp, eval_pp = config.preprocessing_train_eval.split(";")
    train_pp, eval_pp = map(preprocessing.make_preprocessing_fn, [train_pp, eval_pp])

    train_subset = Subset(dataset, train_indices)
    train_collate = functools.partial(
        collate_fn, image_transform=transform_with_augmentation, preprocessing_fn=train_pp
    )
    batch_sampler = BatchSampler(
        len(train_subset), batch_size=config.batch_size, num_steps=config.num_steps
    )
    train_loader = DataLoader(
        train_subset,
        # batch_size=config.batch_size,
        batch_sampler=batch_sampler,
        collate_fn=train_collate,
        num_workers=config.num_workers,
    )

    test_subset = Subset(dataset, test_indices)
    test_collate = functools.partial(
        collate_fn, image_transform=transform_vanilla, preprocessing_fn=eval_pp
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        collate_fn=test_collate,
        num_workers=config.num_workers,
    )

    validation_loader = None
    if config.use_validation_split:
        validation_subset = Subset(dataset, validation_indices)
        validation_loader = DataLoader(
            validation_subset,
            batch_size=config.batch_size,
            collate_fn=test_collate,
            num_workers=config.num_workers,
        )

    return train_loader, test_loader, validation_loader


def wrap_dataloader(dataloader, device, data_time_warning_threshold=0.5):
    data_start = time.time()
    for batch in tqdm.tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        data_time = time.time() - data_start
        compute_start = time.time()
        yield batch
        compute_time = time.time() - compute_start
        if data_time / (data_time + compute_time) > data_time_warning_threshold:
            print(
                colored("WARNING:", "red"),
                f"spending more than {int(data_time_warning_threshold * 100)}% of time fetching data (data={data_time:.4f}s, compute={compute_time:.4f}s)",
            )
        data_start = time.time()


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, batch_size, num_steps):
        self.dataset_size = dataset_size
        self.indices = np.array([], dtype=np.long)
        self.batch_size = batch_size
        self.num_steps = num_steps

    def __iter__(self):
        for _ in range(self.num_steps):
            if len(self.indices) < self.batch_size:
                new_indices = np.arange(self.dataset_size)
                np.random.shuffle(new_indices)
                self.indices = np.concatenate([self.indices, new_indices])

            sampled_indices = self.indices[: self.batch_size]
            self.indices = self.indices[self.batch_size :]
            yield sampled_indices

    def __len__(self):
        return self.num_steps
