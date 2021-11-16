import torch
from torch.nn import functional as F


def compute_expected_calibration_error(logits, labels, num_bins: int = 10):
    """Calculates the Expected Calibration Error of a model.

    The input to this metric is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015. See https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Args:
        logits : logits of a model, NOT the softmax scores.
        labels : True labels. Same dimension as logits
        num_bins: Number of bins ending with a dot (but it's not too important).

    Returns:
        float: ece metric of the model
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    new_logits = torch.stack((-logits, logits), dim=1)

    softmaxes = F.softmax(new_logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    is_correct = predictions.eq(labels)

    # Calculated |confidence - accuracy| in each bin
    lower = confidences[None, :] > bin_lowers[:, None]
    upper = confidences[None, :] <= bin_uppers[:, None]
    in_bin = lower & upper

    prop_in_bin = in_bin.float().mean(1)
    accuracy_in_bin = (is_correct[None, :] * in_bin).sum(dim=1) / (in_bin.sum(dim=1) + 1e-20)
    avg_confidence_in_bin = (confidences[None, :] * in_bin).sum(dim=1) / (in_bin.sum(dim=1) + 1e-20)

    ece = torch.sum(torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin)
    return ece
