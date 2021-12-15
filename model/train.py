import copy
import time
import matplotlib.pyplot as plt
import ml_collections
import deepchest
import os

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import get_model, get_smaller_resnet


def train_model(
    model,
    device,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    config,
    num_epochs=25,
    seed=None
):
    since = time.time()

    for epoch in range(num_epochs):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            data_loader = {
                "train": train_loader,
                "test": test_loader,
                "val": test_loader,
            }[phase]
            max_iteration = len(data_loader)
            # Iterate over data.
            batch_i = 1
            image_count = 0
            for batch in data_loader:
                images, sites, masks, labels = (
                    batch["images"],
                    batch["sites"],
                    batch["mask"],
                    batch["label"],
                )
                image_count += images.shape[0]
                images = images.to(device)
                sites = sites.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images, sites, masks)
                    outputs = torch.squeeze(outputs)
                    preds = torch.zeros_like(outputs)
                    preds[outputs > 0.5] = 1
                    preds[outputs <= 0.5] = 0
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        # print('model.fc3.weight.data', model.fc3.weight.data)

                # statistics
                label_names = deepchest.utils.get_label_names(config.labels_file)
                scores, labels = deepchest.utils.model_evaluation(model, test_loader, device)
                if phase == "train":
                    model.train()
                train_metrics = deepchest.utils.compute_metrics(labels, scores, label_names)
                running_loss = train_metrics['loss'].item()
                running_corrects = train_metrics['balanced_accuracy']
                # running_loss += loss.item() * images.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                print(
                    f"iteration: {batch_i}, running_loss: {running_loss}, correct: {running_corrects}"
                    + " " * 10,
                    end="\r",
                )
                batch_i += 1

                if image_count >= max_iteration:
                    break
                if phase == "train":
                    scheduler.step()
                
                # save best model 
                if (running_corrects > best_acc):
                    best_acc = running_corrects
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(
                        {
                            "seed": seed,
                            "batch_i": batch_i,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": running_loss,
                            "acc": running_corrects,
                        },
                        os.path.join(
                            config.save_dir,
                            f"best_model_fast_TL_epoch{epoch}_test_fold_index{config.test_fold_index}.ds2-2",
                        ),
                    )
            label_names = deepchest.utils.get_label_names(config.labels_file)
            scores, labels = deepchest.utils.model_evaluation(model, test_loader, device)
            train_metrics = deepchest.utils.compute_metrics(labels, scores, label_names)
            epoch_loss = train_metrics['loss'].item()
            epoch_acc = train_metrics['balanced_accuracy']
            # epoch_loss = running_loss / batch_i
            # epoch_acc = running_corrects / batch_i

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            print("Last prediction", preds, "\nLabels.data", labels.data)

            # # deep copy the model
            # if (phase == "val") and (epoch_acc > best_acc):
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            #     torch.save(
            #         {
            #             "epoch": epoch,
            #             "model_state_dict": model.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "loss": epoch_loss,
            #             "acc": epoch_acc,
            #         },
            #         os.path.join(
            #             config.save_dir,
            #             f"best_model_fast_TL_epoch{epoch}_test_fold_index{config.test_fold_index}.ds2-2",
            #         ),
            #     )
            #     print(
            #         "saved",
            #         f"best_model_fast_TL_epoch{epoch}_test_fold_index{config.test_fold_index}.ds2-2",
            #     )

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 16
    config.num_steps = 300

    # See preprocessing.py, if you replace with ";" no preprocessing is done
    config.preprocessing_train_eval = "independent_dropout(.2);"

    config.use_validation_split = False

    # If validation split is false, then train will have 4/5 of data and test 1/5
    # If validation split is true, then train will have 3/5 of data, test 1/5 and val 1/5
    config.num_folds = 2

    # gpu workers
    config.num_workers = 0

    # dataset
    config.images_directory = "dataset/images/"
    config.labels_file = "dataset/labels/diagnostic.csv"

    # Fold seed
    config.random_state = 0

    # Where the indices are saved
    config.save_dir = "model_saved/"
    config.export_folds_indices_file = "indices.csv"

    # Don't modify these (should not have been in the config)
    config.test_fold_index = 0
    config.delta_from_test_index_to_validation_index = 1
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def train_kfold(seed=None):
    config = get_config()

    if torch.cuda.is_available():
        print("CUDA is available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if seed is None:
        set_seed(config.random_state)
    else:
        set_seed(seed)

    metric = dict()
    for i in range(config.num_folds):
        config.test_fold_index = i
        print(f'k-fold #{config.test_fold_index} {"=" * 20}')
        (
            train_loader,
            test_loader,
            validation_loader,
        ) = deepchest.dataset.get_data_loaders(config=config)

        # model = get_model().to(device)
        model = get_smaller_resnet().to(device)
        criterion = nn.BCELoss()
        params = (
            # # for simple model
            # model.parameters()

            # for transfer learning
            list(model.fc1.parameters())
            + list(model.fc3.parameters())
            + list(model.pos_embedding.parameters())
        )
        optimizer_conv = optim.Adam(params, lr=1e-4)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)
        model = train_model(
            model,
            device,
            train_loader,
            test_loader,
            criterion,
            optimizer_conv,
            exp_lr_scheduler,
            config,
            num_epochs=2,
            seed=seed,
        )
        model.eval()
        model.to(device)

        label_names = deepchest.utils.get_label_names(config.labels_file)
        scores, labels = deepchest.utils.model_evaluation(model, test_loader, device)
        train_metrics = deepchest.utils.compute_metrics(labels, scores, label_names)

        print(scores)
        print(train_metrics)
        metric[i] = train_metrics
        torch.save(metric, f'metric-{seed}.ds2-2')

        del train_loader
        del test_loader
        del validation_loader
        torch.cuda.empty_cache()  # clear memory
    return metric

# def load_checkpoint(model, optim, path):
#     checkpoint = torch.load(path)
#     print('loaded checkpoint from ', path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optim.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == "__main__":
    variance_sum = 0.
    first_diff = None
    metrics = []
    for i in range(5):
        print(f'running {i}x2CV (seed={i})')

        metric = train_kfold(seed=i)
        metrics.append(metric)

        score_diff_1 = metric[0]['balanced_accuracy']
        score_diff_2 = metric[1]['balanced_accuracy']
        print(f'score_diff_1 {score_diff_1}\nscore_diff_2 {score_diff_2}')
        score_mean = (score_diff_1 + score_diff_2) / 2.
        score_var = ((score_diff_1 - score_mean)**2 +
                     (score_diff_2 - score_mean)**2)
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1
    torch.save(metrics, f'metrics.ds2-2')
    numerator = first_diff
    denominator = np.sqrt(1/5. * variance_sum)
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), 5)*2.
    print( 't_stat', float(t_stat), 'pvalue', float(pvalue) ) 
