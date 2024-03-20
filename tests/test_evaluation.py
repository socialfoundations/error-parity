"""Test functions in the evaluation.py module."""

import logging
import pytest
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


def test_equalized_odds_measure():
    pass
    # # Check realized constraint violation
    # groupwise_differences = [
    #     np.linalg.norm(
    #         actual_group_roc_points[i] - actual_group_roc_points[j],
    #         ord=np.inf,
    #     )
    #     for i, j in product(unique_groups, unique_groups)
    #     if i < j
    # ]
