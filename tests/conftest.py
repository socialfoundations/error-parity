"""Common pytest fixtures to be used in different tests.

NOTE: file name must be "conftest.py" to enable sharing fixtures across multiple
files.

Pytest reference:
https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from __future__ import annotations
from typing import Iterable

import pytest
import numpy as np

from error_parity.cvxpy_utils import ALL_CONSTRAINTS


@pytest.fixture(params=[42])
def random_seed(request) -> int:
    return request.param


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    return np.random.default_rng(random_seed)


@pytest.fixture(params=[0.01, 0.02, 0.05, 0.1, 0.2, 1.0])
def constraint_slack(request) -> float:
    """Fixture for constraint slack/violation (fairness tolerance)."""
    return request.param


@pytest.fixture(params=list(ALL_CONSTRAINTS))
def fairness_constraint(request) -> float:
    """Fixture for the fairness constraint to test."""
    return request.param


@pytest.fixture(params=[1_000, 10_000, 100_000])
def num_samples(request) -> int:
    return request.param


@pytest.fixture
def y_pred_scores(num_samples: int, rng) -> np.ndarray:
    return rng.random(size=num_samples)


@pytest.fixture(
    params=[
        0.2,
        0.5,
        (0.3, 0.3, 0.4),
        (0.1, 0.2, 0.7),
        (0.1, 0.1, 0.2, 0.6),
        (0.1, 0.1, 0.1, 0.2, 0.5),
    ]
)
def group_relative_size(request) -> tuple[float, ...]:
    """The relative size of each group in the population."""
    prev = request.param
    if isinstance(prev, float):
        return (prev, 1 - prev)
    else:
        assert isinstance(prev, Iterable)
        assert sum(prev) == 1  # sanity check
        return prev


@pytest.fixture
def sensitive_attribute(
    group_relative_size: tuple[float, ...],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly generates sensitive attribute following a provided distribution.

    Parameters
    ----------
    group_relative_size : tuple[float]
        A tuple containing the probabilities associated with each sensitive
        attribute.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        A 1-D array whose entries are the group (or sensitive attribute value)
        each sample belongs to.
    """
    num_groups = len(group_relative_size)

    return rng.choice(
        np.arange(num_groups),
        size=num_samples,
        p=group_relative_size,
    )


@pytest.fixture
def y_true(
    y_pred_scores: np.ndarray, sensitive_attribute: np.ndarray, rng
) -> np.ndarray:
    """Randomly generate labels and predictions with different group-wise
    classification performance.
    """
    n_groups = len(np.unique(sensitive_attribute))
    n_samples = len(sensitive_attribute)

    # Different levels of gaussian noise per group
    group_noise = [0.2 + rng.random() / 2 for _ in range(n_groups)]

    # Generate predictions
    label_prevalence = 0.2 + (rng.random() * 0.6)  # in [0.2, 0.8]

    # Generate labels with different noise levels for each group
    y_true = np.zeros(n_samples)

    for i in range(n_groups):
        group_filter = sensitive_attribute == i
        y_true_groupwise = (
            (
                y_pred_scores[group_filter]
                + rng.normal(size=np.sum(group_filter), scale=group_noise[i])
            )
            > (1 - label_prevalence)
        ).astype(int)

        y_true[group_filter] = y_true_groupwise

    return y_true
