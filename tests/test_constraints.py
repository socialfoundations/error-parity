"""Test the relaxed equal odds constraint fulfillment.
"""

import logging

import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from error_parity import RelaxedThresholdOptimizer
from error_parity.cvxpy_utils import SOLUTION_TOLERANCE, calc_cost_of_point
from error_parity.roc_utils import compute_roc_point_from_predictions
from error_parity.evaluation import evaluate_fairness


def test_synthetic_data_generation(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    sensitive_attribute: np.ndarray,
):
    """Tests the synthetic data generated for the constraints' tests."""
    # Check that group-wise ROC AUC makes sense
    unique_groups = np.unique(sensitive_attribute)
    for g in unique_groups:
        group_filter = sensitive_attribute == g

        group_auc = roc_auc_score(
            y_true=y_true[group_filter],
            y_score=y_pred_scores[group_filter],
        )
        logging.info(f"Group {g} AUC: {group_auc:.3}")

        assert 0.52 < group_auc < 0.98, \
            f"Synthetic data generated has group-{g} AUC of {group_auc}"


def get_metric_abs_tolerance(group_size: int) -> float:
    """Reasonable value for metric fulfillment given the inherent randomization
    of predictions and the size of the group over which the metric is computed.
    """
    return (0.5 * group_size) ** (-1 / 2)
    # return group_size ** (-1/2)


def check_metric_tolerance(
    theory_val: float, empirical_val, group_size: int, metric_name: str = ""
) -> bool:
    """Checks that the empirical value is within a reasonable tolerance of the expected theoretical value."""
    assert np.isclose(
        theory_val,
        empirical_val,
        atol=get_metric_abs_tolerance(group_size),
        rtol=0.01,
    ), f"> '{metric_name}' mismatch; expected {theory_val:.3}; got {empirical_val:.3};"


def test_invalid_constraint_name():
    with pytest.raises(ValueError):
        _ = RelaxedThresholdOptimizer(
            predictor=print,
            constraint="random constraint name",
        )


def test_equalized_odds_lp_relaxation(
    X_features: np.ndarray,
    y_true: np.ndarray,
    sensitive_attribute: np.ndarray,
    predictor: callable,
    constraint_slack: float,
    l_p_norm: int,
    random_seed: int,
):
    """Tests different l-p relaxations for the "equalized-odds" constraint."""
    clf = RelaxedThresholdOptimizer(
        predictor=predictor,
        constraint="equalized_odds",
        tolerance=constraint_slack,
        false_pos_cost=1,
        false_neg_cost=1,
        seed=random_seed,
        l_p_norm=l_p_norm,
    )

    # Fit postprocessing to data
    clf.fit(X=X_features, y=y_true, group=sensitive_attribute)

    check_constraint_fulfillment(
        X_features=X_features,
        y_true=y_true,
        sensitive_attribute=sensitive_attribute,
        postprocessed_clf=clf,
    )


def test_constraint_fulfillment(
    X_features: np.ndarray,
    y_true: np.ndarray,
    sensitive_attribute: np.ndarray,
    predictor: callable,
    fairness_constraint: str,
    constraint_slack: float,
    random_seed: int,
):
    """Tests fairness constraint fulfillment on the given synthetic data."""
    clf = RelaxedThresholdOptimizer(
        predictor=predictor,
        constraint=fairness_constraint,
        tolerance=constraint_slack,
        false_pos_cost=1,
        false_neg_cost=1,
        seed=random_seed,
    )

    # Fit postprocessing to data
    clf.fit(X=X_features, y=y_true, group=sensitive_attribute)

    check_constraint_fulfillment(
        X_features=X_features,
        y_true=y_true,
        sensitive_attribute=sensitive_attribute,
        postprocessed_clf=clf,
    )


def check_constraint_fulfillment(
    X_features: np.ndarray,
    y_true: np.ndarray,
    sensitive_attribute: np.ndarray,
    postprocessed_clf: RelaxedThresholdOptimizer,
):
    """Checks that the postprocessed classifier fulfills its target constraint."""

    # Dataset metadata
    unique_groups = np.unique(
        sensitive_attribute
    )  # return is sorted in ascending order
    label_prevalence = np.mean(y_true)

    fairness_constraint = postprocessed_clf.constraint

    # Check that theoretical solution fulfills constraint tolerance
    assert postprocessed_clf.constraint_violation() <= postprocessed_clf.tolerance + SOLUTION_TOLERANCE, (
        f"Solution fails to fulfill the '{fairness_constraint}' inequality; "
        f"got: {postprocessed_clf.constraint_violation()}; "
        f"expected less than {postprocessed_clf.tolerance};"
    )

    # Optimal binarized predictions
    y_pred_binary = postprocessed_clf(X_features, group=sensitive_attribute)

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
        target_fpr, target_tpr = postprocessed_clf.groupwise_roc_points[g]

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

    # > empirical tolerance for constraint violation depends on the smallest group size
    smallest_denominator = min(
        np.sum(labels[sensitive_attribute == g])
        for g in unique_groups
        for labels in (y_true, 1 - y_true)
    )

    # Compute realized constraint violation
    empirical_fairness_results = evaluate_fairness(
        y_true=y_true,
        y_pred=y_pred_binary,
        sensitive_attribute=sensitive_attribute,
    )

    empirical_constraint_violation: float
    if fairness_constraint == "equalized_odds":
        l_p_norm = postprocessed_clf.l_p_norm
        if l_p_norm == 1:
            empirical_constraint_violation = empirical_fairness_results["equalized_odds_diff_l1"]
        elif l_p_norm == 2:
            empirical_constraint_violation = empirical_fairness_results["equalized_odds_diff_l2"]
        elif l_p_norm == np.inf:
            empirical_constraint_violation = empirical_fairness_results["equalized_odds_diff"]
        else:
            raise NotImplementedError(f"Tests not implemented for eq. odds with l-p norm {l_p_norm}")

    elif fairness_constraint in {"true_positive_rate_parity", "false_negative_rate_parity"}:
        empirical_constraint_violation = empirical_fairness_results["tpr_diff"]

    elif fairness_constraint in {"false_positive_rate_parity", "true_negative_rate_parity"}:
        empirical_constraint_violation = empirical_fairness_results["fpr_diff"]

    elif fairness_constraint == "demographic_parity":
        empirical_constraint_violation = empirical_fairness_results["ppr_diff"]

    else:
        raise NotImplementedError(f"Tests not implemented for constraint {fairness_constraint}")

    # Assert realized constraint violation is close to theoretical solution found
    check_metric_tolerance(
        # NOTE: it's fine if actual violation is below slack (and not fine if above)
        empirical_val=max(empirical_constraint_violation - postprocessed_clf.tolerance, 0),
        theory_val=0.0,
        group_size=smallest_denominator,
        metric_name=f"{fairness_constraint} violation above slack",
    )

    # Check realized global ROC point
    target_fpr, target_tpr = postprocessed_clf.global_roc_point
    actual_fpr, actual_tpr = compute_roc_point_from_predictions(
        y_true=y_true,
        y_pred_binary=y_pred_binary,
    )

    # Check realized global FPR
    check_metric_tolerance(
        target_fpr,
        actual_fpr,
        group_size=np.sum(1 - y_true),
        metric_name="global FPR",
    )

    # Check realized global TPR
    check_metric_tolerance(
        target_tpr,
        actual_tpr,
        group_size=np.sum(y_true),
        metric_name="global TPR",
    )

    # Check realized classification loss
    theoretical_cost = postprocessed_clf.cost()
    actual_cost = calc_cost_of_point(
        fpr=actual_fpr,
        fnr=1 - actual_tpr,
        prevalence=label_prevalence,
        false_pos_cost=postprocessed_clf.false_pos_cost,
        false_neg_cost=postprocessed_clf.false_neg_cost,
    )

    check_metric_tolerance(
        theoretical_cost,
        actual_cost,
        group_size=len(y_true),
        metric_name="classification loss",
    )


def test_unprocessing(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    sensitive_attribute: np.ndarray,
    random_seed: int,
):
    """Tests that unprocessing strictly increases accuracy.
    """
    # Predictor function
    # # > predicts the generated scores from the sample indices
    def predictor(idx):
        return y_pred_scores[idx]

    # Hence, for this example, the features are the sample indices
    num_samples = len(y_true)
    X_features = np.arange(num_samples)

    clf = RelaxedThresholdOptimizer(
        predictor=predictor,
        tolerance=1,
        false_pos_cost=1,
        false_neg_cost=1,
        seed=random_seed,
    )

    # Fit postprocessing to data
    clf.fit(X=X_features, y=y_true, group=sensitive_attribute)

    # Optimal binarized predictions
    y_pred_binary = clf(X_features, group=sensitive_attribute)

    # Original accuracy (using group-blind thresholds)
    original_acc = accuracy_score(y_true, (y_pred_scores >= 0.5).astype(int))

    # Unprocessed accuracy (using group-dependent thresholds)
    unprocessed_acc = accuracy_score(y_true, y_pred_binary)

    # Assert that unprocessing always improves (or maintains) accuracy
    assert unprocessed_acc >= original_acc
