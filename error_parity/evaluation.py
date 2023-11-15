"""A set of functions to evaluate predictions on common performance
and fairness metrics, possibly at a specified FPR or FNR target.

Based on: https://github.com/AndreFCruz/hpt/blob/main/src/hpt/evaluation.py
"""
from __future__ import annotations

import logging
import statistics
from typing import Optional
from itertools import product

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, mean_squared_error, accuracy_score

from .roc_utils import compute_roc_point_from_predictions
from .binarize import compute_binary_predictions
from ._commons import join_dictionaries


def _is_valid_number(num) -> bool:
    return isinstance(num, (float, int, np.number)) and not np.isnan(num)


def _safe_division(a: float, b: float, *, worst_result: float):
    """Tries to divide the given arguments and returns `worst_result` if unsuccessful."""
    if b == 0 or not _is_valid_number(a) or not _is_valid_number(b):
        logging.debug(f"Error in the following division: {a} / {b}")
        return worst_result

    else:
        return a / b


def eval_accuracy_and_equalized_odds(
        y_true: np.ndarray,
        y_pred_binary: np.ndarray,
        sensitive_attr: np.ndarray,
        display: bool = False,
    ) -> tuple[float, float]:
    """Evaluate accuracy and equalized odds of the given predictions.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred_binary : np.ndarray
        The predicted class labels.
    sensitive_attr : np.ndarray
        The sensitive attribute data.
    display : bool, optional
        Whether to print results or not, by default False.

    Returns
    -------
    tuple[float, float]
        A tuple of (fairness, equalized odds violation).
    """
    n_groups = len(np.unique(sensitive_attr))

    roc_points = [
        compute_roc_point_from_predictions(
            y_true[sensitive_attr == i],
            y_pred_binary[sensitive_attr == i])
        for i in range(n_groups)
    ]

    roc_points = np.vstack(roc_points)

    linf_constraint_violation = [
        np.linalg.norm(roc_points[i] - roc_points[j], ord=np.inf)
        for i, j in product(range(n_groups), range(n_groups))
        if i < j
    ]

    acc_val = accuracy_score(y_true, y_pred_binary)
    eq_odds_violation = max(linf_constraint_violation)

    if display:
        print(f"\tAccuracy:   {acc_val:.2%}")
        print(f"\tUnfairness: {eq_odds_violation:.2%}")

    return (acc_val, eq_odds_violation)


def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluates the provided predictions on common performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The discretized predictions.

    Returns
    -------
    dict
        A dictionary with key-value pairs of (metric name, metric value).
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()

    total = tn + fp + fn + tp
    pred_pos = tp + fp
    pred_neg = tn + fn
    assert pred_pos + pred_neg == total

    label_pos = tp + fn
    label_neg = tn + fp
    assert label_pos + label_neg == total

    results = {}

    # Accuracy
    results["accuracy"] = (tp + tn) / total

    # True Positive Rate (Recall)
    results["tpr"] = _safe_division(tp, label_pos, worst_result=0)

    # False Negative Rate (1 - TPR)
    results["fnr"] = _safe_division(fn, label_pos, worst_result=1)
    assert results["tpr"] + results["fnr"] == 1

    # False Positive Rate
    results["fpr"] = _safe_division(fp, label_neg, worst_result=1)

    # True Negative Rate
    results["tnr"] = _safe_division(tn, label_neg, worst_result=0)
    assert results["tnr"] + results["fpr"] == 1

    # Precision
    results["precision"] = _safe_division(tp, pred_pos, worst_result=0)

    # Positive Prediction Rate
    results["ppr"] = _safe_division(pred_pos, total, worst_result=0)

    return results


def evaluate_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
    return_groupwise_metrics: Optional[bool] = False,
) -> dict:
    """Evaluates fairness as the ratios between group-wise performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The discretized predictions.
    sensitive_attribute : np.ndarray
        The sensitive attribute (protected group membership).
    return_groupwise_metrics : Optional[bool], optional
        Whether to return group-wise performance metrics (bool: True) or only
        the ratios between these metrics (bool: False), by default False.

    Returns
    -------
    dict
        A dictionary with key-value pairs of (metric name, metric value).
    """
    # All unique values for the sensitive attribute
    unique_groups = np.unique(sensitive_attribute)

    results = {}
    groupwise_metrics = {}
    unique_metrics = set()

    # Helper to compute key/name of a group-wise metric
    def group_metric_name(metric_name, group_name):
        return f"{metric_name}_group={group_name}"

    assert (
        len(unique_groups) > 1
    ), f"Found a single unique sensitive attribute: {unique_groups}"

    for s_value in unique_groups:
        # Indices of samples that belong to the current group
        group_indices = np.argwhere(sensitive_attribute == s_value).flatten()

        # Filter labels and predictions for samples of the current group
        group_labels = y_true[group_indices]
        group_preds = y_pred[group_indices]

        # Evaluate group-wise performance
        curr_group_metrics = evaluate_performance(group_labels, group_preds)

        # Add group-wise metrics to the dictionary
        groupwise_metrics.update(
            {
                group_metric_name(metric_name, s_value): metric_value
                for metric_name, metric_value in curr_group_metrics.items()
            }
        )

        unique_metrics = unique_metrics.union(curr_group_metrics.keys())

    # Compute ratios and absolute diffs
    for metric_name in unique_metrics:
        curr_metric_results = [
            groupwise_metrics[group_metric_name(metric_name, group_name)]
            for group_name in unique_groups
        ]

        # Metrics' ratio
        ratio_name = f"{metric_name}_ratio"

        # NOTE: should this ratio be computed w.r.t. global performance?
        # - i.e., min(curr_metric_results) / global_curr_metric_result;
        # - same question for the absolute diff calculations;
        results[ratio_name] = _safe_division(
            min(curr_metric_results), max(curr_metric_results),
            worst_result=0,
        )

        # Metrics' absolute difference
        diff_name = f"{metric_name}_diff"
        results[diff_name] = max(curr_metric_results) - min(curr_metric_results)

    # Equal odds: maximum constraint violation for TPR and FPR equality
    # i.e., the smallest ratio
    results["equalized_odds_ratio"] = min(
        results["fnr_ratio"],
        results["fpr_ratio"],
    )

    # or the largest absolute difference
    results["equalized_odds_diff"] = max(
        results["tpr_diff"],  # same as FNR diff
        results["fpr_diff"],  # same as TNR diff
    )

    # Optionally, return group-wise metrics as well
    if return_groupwise_metrics:
        results.update(groupwise_metrics)

    return results


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    sensitive_attribute: Optional[np.ndarray] = None,
    return_groupwise_metrics: bool = False,
    **threshold_target,
) -> dict:
    """Evaluates the given predictions on both performance and fairness metrics.

    Will only evaluate fairness if `sensitive_attribute` is provided.

    Note
    ----
    The value of `log_loss` may be inaccurate when using `scikit-learn<1.2`.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred_scores : np.ndarray
        The predicted scores.
    sensitive_attribute : np.ndarray, optional
        The sensitive attribute - which protected group each sample belongs to.
        If not provided, will not compute fairness metrics.
    return_groupwise_metrics : bool
        Whether to return groupwise performance metrics (requires providing
        `sensitive_attribute`).

    Returns
    -------
    dict
        A dictionary of (key, value) -> (metric_name, metric_value).
    """
    # Binarize predictions according to the given threshold target
    y_pred_binary = compute_binary_predictions(
        y_true,
        y_pred_scores,
        **threshold_target,
    )

    # Compute global performance metrics
    results = evaluate_performance(y_true, y_pred_binary)

    # Compute loss metrics
    results.update(
        {
            "squared_loss": mean_squared_error(y_true, y_pred_scores),
            "log_loss": log_loss(
                y_true, y_pred_scores,
                # eps=np.finfo(y_pred_scores.dtype).eps,    # NOTE: for sklearn<1.2

                # NOTE: this parameterization of `eps` is no longer useful as
                # per sklearn 1.2, and will be removed in sklearn 1.5;
            ),
        }
    )

    # (Optionally) Compute fairness metrics
    if sensitive_attribute is not None:
        results.update(
            evaluate_fairness(
                y_true,
                y_pred_binary,
                sensitive_attribute,
                return_groupwise_metrics=return_groupwise_metrics,
            )
        )

    return results


def evaluate_predictions_bootstrap(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    sensitive_attribute: np.ndarray,
    k: int = 200,
    confidence_pct: float = 95,
    seed: int = 42,
    **threshold_target,
) -> dict:
    """Computes bootstrap estimates of several metrics for the given predictions.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred_scores : np.ndarray
        The score predictions.
    sensitive_attribute : np.ndarray
        The sensitive attribute data.
    k : int, optional
        How many bootstrap samples to draw, by default 200.
    confidence_pct : float, optional
        How large of a confidence interval to use when reporting lower and upper
        bounds, by default 95 (i.e., 2.5 to 97.5 percentile of results).
    seed : int, optional
        The random seed, by default 42.

    Returns
    -------
    dict
        A dictionary of results
    """
    assert len(y_true) == len(y_pred_scores)
    rng = np.random.default_rng(seed=seed)

    # Set threshold target if unset
    threshold_target.setdefault("threshold", 0.50)

    # Draw k bootstrap samples with replacement
    results = []
    for _ in range(k):
        # Indices of current bootstrap sample
        indices = rng.choice(len(y_true), replace=True, size=len(y_true))

        # Evaluate predictions on this bootstrap sample
        results.append(
            evaluate_predictions(
                y_true=y_true[indices],
                y_pred_scores=y_pred_scores[indices],
                sensitive_attribute=sensitive_attribute[indices],
                **threshold_target,
            )
        )

    # Compute statistics from bootstrapped results
    all_metrics = set(results[0].keys())

    bt_mean = {}
    bt_stdev = {}
    bt_percentiles = {}

    low_percentile = (100 - confidence_pct) / 2
    confidence_percentiles = [low_percentile, 100 - low_percentile]

    for m in all_metrics:
        metric_values = [r[m] for r in results]

        bt_mean[m] = statistics.mean(metric_values)
        bt_stdev[m] = statistics.stdev(metric_values)
        bt_percentiles[m] = tuple(np.percentile(metric_values, confidence_percentiles))

    # Construct DF with results

    return join_dictionaries(
        *(
            {
                f"{metric}_mean": bt_mean[metric],
                f"{metric}_stdev": bt_stdev[metric],
                f"{metric}_low-percentile": bt_percentiles[metric][0],
                f"{metric}_high-percentile": bt_percentiles[metric][1],
            }
            for metric in sorted(bt_mean.keys())
        )
    )
