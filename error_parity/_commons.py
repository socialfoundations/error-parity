from __future__ import annotations

import logging
import operator
from functools import reduce

import numpy as np
from scipy.spatial import qhull, ConvexHull


def arrays_are_equal(*arrays: list[np.ndarray]) -> bool:
    """Compares two or more arrays and returns whether they are equal."""
    assert len(arrays) >= 2, \
        f"At least two arguments must be provided, got {len(arrays)}."

    # Reference array
    ref_array = arrays[0]
    ref_array_np = np.array(ref_array)

    for curr_arr in arrays[1:]:
        curr_arr_np = np.array(curr_arr)

        # Check shape and contents
        if (ref_array_np.shape != curr_arr_np.shape
            or not np.allclose(ref_array_np, curr_arr_np)
            ):
            return False    # arrays are not equal

    # All checks passed, return True (arrays are equal)
    return True


def join_dictionaries(*dicts) -> dict:
    """Joins a sequence of dictionaries into a single dictionary."""
    return reduce(operator.or_, dicts)


def is_part_of_convexhull(costs: np.ndarray) -> np.ndarray:
    """Returns a boolean array indicating whether each point is in the convex hull."""
    try:
        hull = ConvexHull(costs)
        hull_indices = set(hull.vertices)
        return np.array([idx in hull_indices for idx in range(len(costs))], dtype=np.bool_)

    except qhull.QhullError as err:
        logging.error(f"Failed to compute ConvexHull with error: {err}")
        return np.ones(len(costs), dtype=np.bool_)


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Returns a boolean array indicating whether each point is Pareto efficient.

    NOTE: assume values indicate costs (smaller is better)!

    Parameters
    ----------
    costs : np.ndarray
        An (n_points, n_costs) array.
    Returns
    -------
    np.ndarray
        A (n_points,) boolean array corresponding to which input points are Pareto optimal.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self

    return is_efficient


def get_cost_envelope(costs: np.ndarray) -> np.ndarray:
    """Finds the Pareto efficient points that are part of the convex hull."""
    return costs[
        is_part_of_convexhull(costs)
        & is_pareto_efficient(costs)
    ]
