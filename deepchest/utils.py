"""
Utility functions.
"""

import datetime
from collections import Counter, defaultdict
from typing import Any, Dict, Optional

import ml_collections
import numpy as np
import plotly.express as px
import torch
import wandb
from ml_collections import ConfigDict
from rich.console import Console
from rich.table import Table
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, balanced_accuracy_score, roc_curve
from termcolor import colored
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

import deepchest.calibration
import deepchest.metrics


def show_splits_info(train_indices, test_indices, valid_indices, *, dataset, label_names):
    console = Console()

    table = Table(show_header=True)
    table.add_column("split")
    table.add_column("size", justify="right")
    for label in label_names:
        table.add_column(label, justify="right")

    for split_name, indices in [
        ("train", train_indices),
        ("valid", valid_indices),
        ("test", test_indices),
    ]:
        if indices is not None and len(indices) > 0:
            total = len(indices)
            count = Counter([dataset[i]["label"] for i in indices])
            cols = [split_name, str(total)] + [
                f"{count[i]} ({count[i] / total * 100:.0f}%)" for i in range(len(label_names))
            ]
            table.add_row(*cols)

    print("Split infos:")
    console.print(table)


def format_time(elapsed: float) -> str:
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def log_metrics(title: str, metrics: dict, *, color=None) -> None:
    n = (
        metrics["true_positive"]
        + metrics["true_negative"]
        + metrics["false_positive"]
        + metrics["false_negative"]
    )
    accuracy = (metrics["true_positive"] + metrics["true_negative"]) / n
    sensitivity = metrics["true_positive"] / (metrics["true_positive"] + metrics["false_negative"])
    specificity = metrics["true_negative"] / (metrics["true_negative"] + metrics["false_positive"])

    def maybe_metric_str(metric_name):
        if metric_name in metrics:
            return f"{metric_name}: {metrics[metric_name]}"
        else:
            return ""

    print(
        colored(
            f"{title}:\t"
            f"loss: {metrics['loss']:.4f}\t"
            f"accuracy: {accuracy:.4f}\t"
            f"sensitivity {sensitivity:.4f}\t"
            f"specificity {specificity:.4f}\t"
            f"AUC {metrics['roc_auc']:.4f}\t"
            f"balanced accuracy: {metrics['balanced_accuracy']:.4f}\t" + maybe_metric_str("time"),
            color,
        )
    )


def plot_auc(fpr, tpr):
    fig = px.line(x=fpr, y=tpr, labels={"x": "False Positive Rate", "y": "True Positive Rate"})
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_calibration_curve(mean_predicted_value, fraction_of_positives):
    """Plot calibration curve. """
    fig = px.line(
        x=mean_predicted_value,
        y=fraction_of_positives,
        labels={"x": "False Positive Rate", "y": "Mean predicted value"},
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def compute_metrics(labels, logits, label_names):
    labels = labels.detach().to("cpu")
    logits = logits.detach().to("cpu")

    criterion = BCEWithLogitsLoss()

    loss = criterion(logits, labels.float())
    ece = deepchest.calibration.compute_expected_calibration_error(logits, labels.float())

    predictions = logits > 0.0

    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    true_negative = ((predictions == 0) & (labels == 0)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()

    balanced_accuracy = balanced_accuracy_score(y_true=labels, y_pred=predictions)

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=logits)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, torch.sigmoid(logits), n_bins=10
    )

    roc_auc = auc(fpr, tpr)

    return {
        "loss": loss,
        "ece": ece,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "roc_auc": roc_auc,
        "roc": plot_auc(fpr, tpr),
        "mean_predicted_value": mean_predicted_value,
        "fraction_of_positives": fraction_of_positives,
        "ece_curve": plot_calibration_curve(mean_predicted_value, fraction_of_positives),
        "labels": labels,
        "logits": logits,
    }


def get_wandb_roc_params(targets, logits, labels):
    pred = (logits > 0.0).numpy()
    pred_plot = np.zeros((len(targets), len(labels)))
    for i, v in enumerate(pred):
        pred_plot[i, v] = 1
    return wandb.plot.roc_curve(
        targets.numpy().astype("int64"), pred_plot.astype("int64"), labels=labels
    )


def prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def model_evaluation(
    model,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluates model using given data and loss criterion."""
    model.eval()

    epoch_labels = deepchest.metrics.Concatenate()
    epoch_scores = deepchest.metrics.Concatenate()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["images"].to(device)
            sites = batch["sites"].to(device)
            mask = batch["mask"].to(device)
            labels = torch.flatten(batch["label"]).to(device)

            scores = model(images, sites, mask)
            scores = torch.flatten(scores)  # if binary

            epoch_labels.append(labels)
            epoch_scores.append(scores)

    return epoch_scores.value, epoch_labels.value


def split_array_most_equaly(array, num_splits: int):
    """Split array in k arrays of similar sizes."""
    n = len(array)
    split_sizes = np.ones(num_splits, dtype=int) * (n // num_splits)
    split_sizes[: n % num_splits] += 1

    offset = 0
    splits = []
    for size in split_sizes:
        splits.append(array[offset : offset + size])
        offset += size

    return splits


def split_k_folds(indices, labels, k: int, random_state: int = 0):
    """Stratified K-fold of the indices array."""
    # split indices per label
    indices_by_label = defaultdict(lambda: [])
    for index, label in zip(indices, labels):
        indices_by_label[label].append(index)

    # shuffle each with a fixed random key
    np.random.seed(random_state)
    separate_indices = []
    for _, indices in indices_by_label.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        separate_indices.append(indices)

    # split each in k folds
    folds = [[] for _ in range(k)]
    for i, indices in enumerate(separate_indices):
        # Smallest fold first for a greedy strategy to balance the split sizes.
        folds = sorted(folds, key=lambda indices: sum(map(len, indices)))
        current_label_folds = split_array_most_equaly(indices, k)
        for j in range(k):
            folds[j].append(current_label_folds[j])

    folds = [np.concatenate(indices) for indices in folds]

    # Reshuffle
    for f in folds:
        np.random.shuffle(f)

    return folds


def override_config_dict(config: ConfigDict, overrides: Dict[str, Any]):
    for k, v in overrides.items():
        try:
            if "." in k:
                first = k.split(".")[0]
                rest = ".".join(k.split(".")[1:])
                override_config_dict(config[first], {rest: v})
            else:
                config.get_ref(k).set(v)
        except KeyError:
            raise KeyError(f"Cannot override configuration field '{k}'")


class FrozenObservedConfigDict:
    """Wraps a ConfigDict and keeps track of which field are accessed."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        accessed_config: Optional[ml_collections.ConfigDict] = None,
    ):
        # original config
        self.config = config

        self.accessed_config = accessed_config
        if accessed_config is None:
            self.accessed_config = ml_collections.ConfigDict()
        else:
            accessed_config

    def _make_subconfig(self, key):
        if key not in self.accessed_config:
            self.accessed_config[key] = ml_collections.ConfigDict()
        return FrozenObservedConfigDict(self.config[key], self.accessed_config[key])

    def __getitem__(self, key):
        if "." in key:
            first, rest = key.split(".", 1)
            return self._make_subconfig(first)[rest]
        else:
            value = self.config[key]
            if isinstance(value, ml_collections.ConfigDict):
                return self._make_subconfig(key)
            else:
                self.accessed_config[key] = value
                return value

    def __getattr__(self, key):
        return self[key]

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.config})"


def get_label_names(labels_file):
    if "diagnostic" in labels_file:
        return ["negative", "positive"]

    elif "severity" in labels_file or "prognostic" in labels_file:
        # mild = hospital,
        # severe = hospital with O2 or intubated
        return ["mild", "severe"]

    return None


def exclusive_cumsum(t, dim=-1):
    shape = list(t.shape)
    shape[dim] = 1
    zeros = torch.zeros(shape, dtype=t.dtype, device=t.device)
    return torch.cat(
        [zeros, torch.cumsum(t, dim=dim).narrow(dim=dim, start=0, length=t.shape[dim] - 1)], dim=dim
    )


def pad_dim_with_zeros(t, dim, length):
    if t.shape[dim] == length:
        return t
    t_padded_shape = list(t.shape)
    t_padded_shape[dim] = length
    t_padded = torch.zeros(t_padded_shape, device=t.device, dtype=t.dtype)
    t_padded.narrow(dim=dim, start=0, length=t.shape[dim]).copy_(t)
    return t_padded


def try_parse_exact_bool(b):
    if isinstance(b, str):
        if b.lower() == "true":
            return True
        if b.lower() == "false":
            return False
    return b
