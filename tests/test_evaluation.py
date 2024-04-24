"""Test functions in the evaluation.py module."""

import logging
import pytest
import numpy as np

from error_parity import RelaxedThresholdOptimizer
from error_parity.cvxpy_utils import SOLUTION_TOLERANCE
from error_parity.evaluation import _safe_division


@pytest.mark.parametrize("a,b,ret", [(0, 0, -1), (float("nan"), 1, -1), (1, float("nan"), -1)])
def test_invalid_safe_division(a, b, ret, caplog):
    """Test invalid division operation."""
    with caplog.at_level(logging.DEBUG):

        div_result = _safe_division(a, b, worst_result=ret)
        assert "error" in caplog.text.lower()
        assert div_result == ret


def test_valid_safe_division(caplog, rng):
    """Test invalid division operation."""
    with caplog.at_level(logging.DEBUG):
        a = (0.5 - rng.random()) * rng.integers(0, 1e6)
        b = (0.5 - rng.random()) * rng.integers(0, 1e6)

        div_result = _safe_division(a, b, worst_result=float("nan"))
        assert div_result == a / b
        assert "error" not in caplog.text.lower()


def test_equalized_odds_relaxation_costs(
    X_features: np.ndarray,
    y_true: np.ndarray,
    sensitive_attribute: np.ndarray,
    predictor: callable,
    constraint_slack: float,
    random_seed: int,
):
    """Tests whether l-p norms follow standard orders (lower p -> higher norm)."""

    results = {}
    sorted_p_norms = (1, 2, 3, 10, np.inf)
    for norm in sorted_p_norms:
        # Fit postprocessing to data
        clf = RelaxedThresholdOptimizer(
            predictor=predictor,
            constraint="equalized_odds",
            tolerance=constraint_slack,
            false_pos_cost=1,
            false_neg_cost=1,
            seed=random_seed,
            l_p_norm=norm,
        )
        clf.fit(X=X_features, y=y_true, group=sensitive_attribute)

        # Store results
        results[norm] = clf.cost()

    # Check that l-p norms with lower p achieve lower costs and higher unfairness
    for idx in range(1, len(sorted_p_norms)):

        lower_p_norm = sorted_p_norms[idx - 1]
        higher_p_norm = sorted_p_norms[idx]

        lower_p_cost = results[lower_p_norm]
        higher_p_cost = results[higher_p_norm]

        # Assert lower-p costs are higher (accuracy is lower)
        assert lower_p_cost > higher_p_cost - SOLUTION_TOLERANCE, \
            f"l-{lower_p_norm} cost: {lower_p_cost} < l-{higher_p_norm} cost: {higher_p_cost}"
