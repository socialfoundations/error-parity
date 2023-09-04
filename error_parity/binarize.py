"""Module to binarize continuous-score predictions.

Based on: https://github.com/AndreFCruz/hpt/blob/main/src/hpt/binarize.py
"""
import math
import logging
from typing import Optional

import numpy as np


def compute_binary_predictions(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    threshold: Optional[float] = None,
    tpr: Optional[float] = None,
    fpr: Optional[float] = None,
    ppr: Optional[int] = None,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """Discretizes the given score predictions into binary labels.

    If necessary, will randomly untie binary predictions with equal score.

    Parameters
    ----------
    y_true : np.ndarray
        The true binary labels
    y_pred_scores : np.ndarray
        Predictions as a continuous score between 0 and 1
    threshold : Optional[float], optional
        Whether to use a specified (global) threshold, by default None
    tpr : Optional[float], optional
        Whether to target a specified TPR (true positive rate, or recall), by
        default None
    fpr : Optional[float], optional
        Whether to target a specified FPR (false positive rate), by default None
    ppr : Optional[float], optional
        Whether to target a specified PPR (positive prediction rate), by default
        None

    Returns
    -------
    np.ndarray
        The binarized predictions according to the specified target.
    """
    assert sum(1 for val in {threshold, fpr, tpr, ppr} if val is not None) == 1, (
        f"Please provide exactly one of (threshold, fpr, tpr, ppr); got "
        f"{(threshold, fpr, tpr, ppr)}."
    )

    # If threshold provided, just binarize it, no untying necessary
    if threshold:
        return (y_pred_scores >= threshold).astype(int)

    # Otherwise, we need to compute the allowed value for the numerator
    # and corresponding threshold (plus, may require random untying)
    label_pos = np.count_nonzero(y_true)
    label_neg = np.count_nonzero(1 - y_true)
    assert (total := label_pos + label_neg) == len(y_true)  # sanity check

    # Indices of predictions ordered by score, descending
    y_pred_sorted_indices = np.argsort(-y_pred_scores)

    # Labels ordered by descending prediction score
    y_true_sorted = y_true[y_pred_sorted_indices]

    # Number of positive predictions allowed according to the given metric
    # (the allowed budget for the metric's numerator)
    positive_preds_budget: int

    # Samples that count for the positive_preds_budget
    # (LPs for TPR, LNs for FPR, and all samples for PPR)
    # (related to the metric's denominator)
    target_samples_mask: np.ndarray
    if tpr:
        # TPs budget to ensure >= the target TPR
        positive_preds_budget = math.ceil(tpr * label_pos)
        target_samples_mask = y_true_sorted == 1  # label positive samples
        non_target_samples_mask = y_true_sorted == 0  # label negative samples

    elif fpr:
        # FPs budget to ensure <= the target FPR
        positive_preds_budget = math.floor(fpr * label_neg)
        target_samples_mask = y_true_sorted == 0  # label negative samples
        non_target_samples_mask = y_true_sorted == 1  # label positive samples

    elif ppr:
        # PPs budget to ensure <= the target PPR
        positive_preds_budget = math.floor(ppr * total)
        target_samples_mask = np.ones_like(y_true_sorted).astype(bool)  # all samples

    # Indices of target samples (relevant for the target metric), ordered by descending score
    target_samples_indices = y_pred_sorted_indices[target_samples_mask]

    # Find the threshold at which the specified numerator_budget is met
    threshold_idx = target_samples_indices[(positive_preds_budget - 1)]
    threshold = y_pred_scores[threshold_idx]

    ####################################
    # Code for random untying follows: #
    ####################################
    y_pred_binary = (y_pred_scores >= threshold).astype(int)

    # 1. compute actual number of positive predictions (on relevant target samples)
    actual_pos_preds = np.sum(y_pred_binary[target_samples_indices])

    # 2. check if this number corresponds to the target
    if actual_pos_preds != positive_preds_budget:
        logging.warning(
            "Target metric for thresholding could not be met, will randomly "
            "untie samples with the same predicted score to fulfill target."
        )

        assert actual_pos_preds > positive_preds_budget, (
            "Sanity check: actual number of positive predictions should always "
            "be higher or equal to the target number when following this "
            f"algorithm; got actual={actual_pos_preds}, target={positive_preds_budget};"
        )

        # 2.1. if target was not met, compute number of extra predicted positives
        extra_pos_preds = actual_pos_preds - positive_preds_budget

        # 2.2. randomly select extra_pos_preds among the relevant
        # samples (either TPs or FPs or PPs) with the same score
        rng = np.random.RandomState(random_seed)

        samples_at_target_threshold_mask = (
            y_pred_scores[y_pred_sorted_indices] == threshold
        )

        target_samples_at_target_threshold_indices = y_pred_sorted_indices[
            samples_at_target_threshold_mask
            & target_samples_mask  # Filter for samples at target threshold  # Filter for relevant (target) samples
        ]

        # # The extra number of positive predictions must be fully explained by this score tie
        # import ipdb; ipdb.set_trace()   # TODO: figure out why this assertion fails
        # assert extra_pos_preds < len(target_samples_at_target_threshold_indices)

        extra_pos_preds_indices = rng.choice(
            target_samples_at_target_threshold_indices,
            size=extra_pos_preds,
            replace=False,
        )

        # 2.3. give extra_pos_preds_indices a negative prediction
        y_pred_binary[extra_pos_preds_indices] = 0

        # 2.4. Randomly sample the non-target labels at same rate
        if tpr or fpr:
            sampled_fraction = 1 - (positive_preds_budget / actual_pos_preds)

            non_target_samples_at_target_threshold_indices = y_pred_sorted_indices[
                samples_at_target_threshold_mask
                & non_target_samples_mask  # Filter for samples at target threshold  # Filter for positive samples
            ]
            num_samples = (
                non_target_samples_at_target_threshold_indices.shape[0]
                * sampled_fraction
            )

            num_samples = int(round(num_samples, 0))

            if num_samples:
                extra_neg_preds_indices = rng.choice(
                    non_target_samples_at_target_threshold_indices,
                    size=num_samples,
                    replace=False,
                )
                y_pred_binary[extra_neg_preds_indices] = 0

    # Sanity check: the number of positive_preds_budget should now be exactly fulfilled
    assert np.sum(y_pred_binary[target_samples_indices]) == positive_preds_budget

    return y_pred_binary
