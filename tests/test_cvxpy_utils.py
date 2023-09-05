"""Tests the cvxpy utils.
"""

import scipy
import numpy as np

import pytest

from error_parity import cvxpy_utils


@pytest.mark.parametrize("slope", [0, 2, -3])
@pytest.mark.parametrize("intercept", [0, -1, 1])
def test_compute_line(slope: float, intercept: float, rng: np.random.Generator):
    # Generate two points on this line
    X = rng.uniform(low=-1e10, high=1e10, size=2)

    # y = mx + b
    Y = slope * X + intercept
    p1, p2 = map(np.array, zip(X, Y))

    # Recover slope and intercept
    slope_check, intercept_check = cvxpy_utils.compute_line(p1, p2)

    # Check that returned values match original values
    assert slope == slope_check
    assert intercept == intercept_check


def test_compute_line_inf_slope(rng: np.random.Generator):
    # Generate a random x value
    (x,) = rng.uniform(low=-1e10, high=1e10, size=1)

    # Generate two random y values
    y1, y2 = rng.uniform(low=-1e10, high=1e10, size=2)

    # Recover slope and intercept
    p1, p2 = map(np.array, zip([x, x], [y1, y2]))
    slope_check, _intercept_check = cvxpy_utils.compute_line(p1, p2)

    assert np.isinf(slope_check)


def test_compute_line_invalid_args(rng: np.random.Generator):
    with pytest.raises(ValueError) as excinfo:
        random_point = rng.random(size=2)
        cvxpy_utils.compute_line(random_point, random_point)
