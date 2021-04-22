import torch
from torch.nn import functional as F


def binary_clf_curve(
    preds: torch.Tensor,
    target: torch.Tensor,
    sample_weights = None,
    pos_label: int = 1.,
):
    """
    adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_ranking.py
    """
    if sample_weights is not None and not isinstance(sample_weights, torch.Tensor):
        sample_weights = torch.tensor(sample_weights, device=preds.device, dtype=torch.float)

    # remove class dimension if necessary
    if preds.ndim > target.ndim:
        preds = preds[:, 0]
    desc_score_indices = torch.argsort(preds, descending=True)

    preds = preds[desc_score_indices]
    target = target[desc_score_indices]

    if sample_weights is not None:
        weight = sample_weights[desc_score_indices]
    else:
        weight = 1.

    # pred typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
    threshold_idxs = F.pad(distinct_value_indices, (0, 1), value=target.size(0) - 1)
    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

    if sample_weights is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = torch.cumsum((1 - target) * weight, dim=0)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, preds[threshold_idxs]


def precision_recall_curve_compute(
    preds: torch.Tensor,
    target: torch.Tensor,
):

    fps, tps, thresholds = binary_clf_curve(
        preds=preds, target=target
    )
    eps = 1e-6
    precision = tps / (tps + fps + eps)
    recall = tps / (tps[-1] + eps)

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = torch.where(tps == tps[-1])[0][0]
    sl = slice(0, last_ind.item() + 1)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    precision = torch.cat([reversed(precision[sl]), torch.ones(1, dtype=precision.dtype, device=precision.device)])

    recall = torch.cat([reversed(recall[sl]), torch.zeros(1, dtype=recall.dtype, device=recall.device)])

    thresholds = reversed(thresholds[sl]).clone()

    return precision, recall, thresholds


def average_precision_compute(
    preds: torch.Tensor,
    target: torch.Tensor,
):
    precision, recall, _ = precision_recall_curve_compute(preds, target)
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
