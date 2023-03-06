"""
This section covers functionality for computing performance
for [NERP.models.Trainer][] models.
"""

import warnings
import torch

from typing import List
from sklearn.metrics import classification_report

from .utils import flatten


def compute_f1_scores(y_pred: List[List[str]],
                      y_true: List[List[str]],
                      labels: List[str]) -> list:
    """Compute F1 scores.

    Computes F1 Scores

    Args:
        y_pred (List): predicted values.
        y_true (List): observed/true values.
        labels (List): all possible tags.
        kwargs: all optional arguments for precision/recall function.

    Returns:
        list: resulting F1 scores.

    """
    # check inputs.
    assert sum([len(t) < len(p) for t, p in zip(y_true, y_pred)]
               ) == 0, "Length of predictions must not exceed length of observed values"

    # check, if some lengths of observed values exceed predicted values.
    n_exceeds = sum([len(t) > len(p) for t, p in zip(y_true, y_pred)])
    if n_exceeds > 0:
        warnings.warn(
            f'length of observed values exceeded lengths of predicted values in {n_exceeds} cases and were truncated. _Consider_ increasing max_len parameter for your model.')

    # truncate observed values dimensions to match predicted values,
    # this is needed if predictions have been truncated earlier in
    # the flow.
    y_true = [t[:len(p)] for t, p in zip(y_true, y_pred)]

    y_pred = flatten(y_pred)
    y_true = flatten(y_true)

    f1_scores = classification_report(y_true, y_pred, labels=labels, digits=4)
    return f1_scores, y_true


def compute_loss(preds, target_tags, masks, device, n_tags):

    # initialize loss function.
    lfn = torch.nn.CrossEntropyLoss()

    # Compute active loss to not compute loss of paddings
    active_loss = masks.view(-1) == 1

    active_logits = preds.view(-1, n_tags)
    active_labels = torch.where(
        active_loss,
        target_tags.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target_tags)
    )

    active_labels = torch.as_tensor(
        active_labels, device=torch.device(device), dtype=torch.long)

    # Only compute loss on actual token predictions
    loss = lfn(active_logits, active_labels)

    return loss
