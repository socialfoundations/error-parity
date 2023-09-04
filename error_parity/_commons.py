import logging
import operator
from functools import reduce

import numpy as np
from scipy.spatial import qhull, ConvexHull


def join_dictionaries(*dicts) -> dict:
    """Joins a sequence of dictionaries into a single dictionary."""
    return reduce(operator.or_, dicts)


def get_convexhull_indices(
        adjusted_results_df,
        perf_metric: str = "accuracy_fit",
        disp_metric: str = "equalized_odds_diff_fit",
    ) -> np.array:
    """Get indices of points in the convex hull."""
    costs = np.stack(
        (
            1 - adjusted_results_df[f"{perf_metric}"],
            adjusted_results_df[f"{disp_metric}"],
        ),
        axis=1)

    try:
        hull = ConvexHull(costs)
        return hull.vertices

    except qhull.QhullError as err:
        logging.error(f"Failed to compute ConvexHull with error: {err}")
        return np.arange(len(adjusted_results_df))      # Return all points as in the convex hull
