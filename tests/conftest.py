"""Common pytest fixtures to be used in different tests.

NOTE: file name must be "conftest.py" to enable sharing fixtures across multiple
files.

Pytest reference:
https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import pytest
import numpy as np


# Number of samples used for testing
NUM_SAMPLES = 10_000


@pytest.fixture(params=[23, 42])
def random_seed(request) -> int:
    return request.param


@pytest.fixture
def rng(random_seed) -> np.random.Generator:
    return np.random.default_rng(random_seed)


@pytest.fixture
def y_true(prevalence: float, rng: np.random.Generator) -> np.ndarray:
    assert 0 <= prevalence <= 1
    # Generate labels
    return (rng.random(NUM_SAMPLES) <= prevalence).astype(int)


@pytest.fixture
def y_pred_scores(rng: np.random.Generator) -> np.ndarray:
    return rng.random(NUM_SAMPLES)


@pytest.fixture(params=[2, 5, 10, 100, 1000, 10000])
def num_score_bins(request) -> int:
    return request.param


@pytest.fixture
def y_pred_scores_with_ties(num_score_bins: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly generates score predictions with ties.

    NOTE
    - I know there's a bit of confusion with the num_score_bins because to
    get scores evenly spaced by 0.1 we need 11 buckets not 10;
    - This works just fine as each bin gets all scores within range 
    (bin_score - 1/num_score_bins/2, bin_score + 1/num_score_bins/2);
    - For bins 0.0 and 1.0 this range is halved as there are no negative scores
    or scores above 1.0;
    - All in all, it works just fine...


    Parameters
    ----------
    num_score_bins : int
        Number of bins used to discretize scores.

    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Returns score predictions with ties.
    """
    # Discretized scores distributed uniformly at random in range [0, n_buckets]
    scores = ((rng.random(NUM_SAMPLES) + 0.05) * num_score_bins).astype(int)
    return (scores / num_score_bins).clip(0.0, 1.0)
