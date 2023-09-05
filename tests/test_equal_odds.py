"""Test the relaxed equal odds constraint fulfillment.
"""

import logging
from itertools import product

import pytest
import numpy as np
from sklearn.metrics import roc_auc_score

from error_parity import RelaxedThresholdOptimizer
from error_parity.cvxpy_utils import SOLUTION_TOLERANCE, calc_cost_of_point
from error_parity.roc_utils import compute_roc_point_from_predictions


def test_synthetic_data_generation(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    sensitive_attribute: np.ndarray,
):
    # Check that group-wise ROC AUC makes sense
    unique_groups = np.unique(sensitive_attribute)
    for g in unique_groups:
        group_filter = sensitive_attribute == g

        group_auc = roc_auc_score(
            y_true=y_true[group_filter],
            y_score=y_pred_scores[group_filter],
        )
        # print(f"Group {g} AUC: {group_auc:.3}")

        assert 0.52 < group_auc < 0.98


@pytest.fixture(params=[0.01, 0.02, 0.05, 0.1, 0.2, 1.0])
def constraint_slack(request) -> float:
    """aka. fairness tolerance"""
    return request.param


def get_metric_abs_tolerance(group_size: int) -> float:
    """Reasonable value for metric fulfillment given the inherent randomization
    of predictions and the size of the group over which the metric is computed.
    """
    return (0.1 * group_size) ** (-1 / 1.5)
    # return group_size ** (-1/2)


def check_metric_tolerance(
    theory_val: float, empirical_val, group_size: int, metric_name: str = ""
) -> bool:
    assert np.isclose(
        theory_val,
        empirical_val,
        atol=get_metric_abs_tolerance(group_size),
        rtol=0.01,
    ), f"> '{metric_name}' mismatch; expected {theory_val:.3}; got {empirical_val:.3};"


def test_invalid_constraint_name():
    with pytest.raises(ValueError) as excinfo:
        clf = RelaxedThresholdOptimizer(
            predictor=print,
            constraint="random constraint name",
        )


def test_equalized_odds_constraint_relaxation(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    sensitive_attribute: np.ndarray,
    constraint_slack: float,
    random_seed: int,
):
    num_samples = len(y_true)
    unique_groups = np.unique(
        sensitive_attribute
    )  # return is sorted in ascending order
    label_prevalence = np.mean(y_true)

    # Predictor function
    # # > predicts the generated scores from the sample indices
    predictor = lambda idx: y_pred_scores[idx]

    # Hence, for this example, the features are the sample indices
    X_features = np.arange(num_samples)

    clf = RelaxedThresholdOptimizer(
        predictor=predictor,
        constraint="equalized_odds",
        tolerance=constraint_slack,
        false_pos_cost=1,
        false_neg_cost=1,
        seed=random_seed,
    )

    # Fit postprocessing to data
    clf.fit(X=X_features, y=y_true, group=sensitive_attribute)

    # Check that theoretical solution fulfills relaxed constraint
    assert clf.equalized_odds_violation() <= constraint_slack + SOLUTION_TOLERANCE, (
        f"Solution fails to hit the target EO constraint; "
        f"got: {clf.equalized_odds_violation()}; "
        f"expected less than {constraint_slack};"
    )

    # Optimal binarized predictions
    y_pred_binary = clf(X_features, group=sensitive_attribute)

    # Check realized group-specific ROC points
    actual_group_roc_points = np.vstack(
        [
            compute_roc_point_from_predictions(
                y_true=y_true[sensitive_attribute == g],
                y_pred_binary=y_pred_binary[sensitive_attribute == g],
            )
            for g in unique_groups
        ]
    )

    for g in unique_groups:
        g_filter = sensitive_attribute == g
        g_size = np.sum(g_filter)
        g_label_prevalence = np.mean(y_true[g_filter])

        actual_fpr, actual_tpr = actual_group_roc_points[g]
        target_fpr, target_tpr = clf.groupwise_roc_points[g]

        # Check group FPR
        check_metric_tolerance(
            target_fpr,
            actual_fpr,
            g_size * (1 - g_label_prevalence),
            metric_name=f"group {g} FPR",
        )

        # Check group TPR
        check_metric_tolerance(
            target_tpr,
            actual_tpr,
            g_size * g_label_prevalence,
            metric_name=f"group {g} TPR",
        )

    # Check realized constraint violation
    groupwise_differences = [
        np.linalg.norm(
            actual_group_roc_points[i] - actual_group_roc_points[j],
            ord=np.inf,
        )
        for i, j in product(unique_groups, unique_groups)
        if i < j
    ]

    # > empirical tolerance for constraint violation depends on the smallest group size
    smallest_denominator = min(
        np.sum(labels[sensitive_attribute == g])
        for g in unique_groups
        for labels in (y_true, 1 - y_true)
    )
    actual_equalized_odds_violation = np.max(groupwise_differences)
    check_metric_tolerance(
        empirical_val=max(actual_equalized_odds_violation - constraint_slack, 0),
        theory_val=0.0,
        group_size=smallest_denominator,
        metric_name="EO violation above slack",
    )

    # Check realized global ROC point
    target_fpr, target_tpr = clf.global_roc_point
    actual_fpr, actual_tpr = compute_roc_point_from_predictions(
        y_true=y_true,
        y_pred_binary=y_pred_binary,
    )
    check_metric_tolerance(
        target_fpr,
        actual_fpr,
        group_size=np.sum(1 - y_true),
        metric_name="global FPR",
    )
    check_metric_tolerance(
        target_tpr,
        actual_tpr,
        group_size=np.sum(y_true),
        metric_name="global TPR",
    )

    # Check realized classification loss
    theoretical_cost = clf.cost()
    actual_cost = calc_cost_of_point(
        fpr=actual_fpr,
        fnr=1 - actual_tpr,
        prevalence=label_prevalence,
        false_pos_cost=clf.false_pos_cost,
        false_neg_cost=clf.false_neg_cost,
    )

    check_metric_tolerance(
        theoretical_cost,
        actual_cost,
        group_size=num_samples,
        metric_name="classification loss",
    )


def test_tpr_parity_constraint():
    # TODO: test relaxation of TPR parity constraint
    pass


def test_fpr_parity_constraint():
    # TODO: test relaxation of FPR parity constraint
    pass


def test_unprocessing():
    # TODO: test that unconstrained postprocessing strictly increases accuracy, whichever the input
    pass
