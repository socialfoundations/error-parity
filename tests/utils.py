import numpy as np


def get_metric_abs_tolerance(group_size: int) -> float:
    """Reasonable value for metric fulfillment given the inherent randomization
    of predictions and the size of the group over which the metric is computed.
    """
    return (0.1 * group_size) ** (-1 / 1.7)     # tighter for larger groups, less tight for smaller groups
    # return group_size ** (-1/2)


def check_metric_tolerance(
    theory_val: float,
    empirical_val: float,
    group_size: int,
    metric_name: str = "",
    less_or_equal: bool = False,
) -> bool:
    """Checks that empirical value approximately matches theoretical value.

    Parameters
    ----------
    theory_val : float
        The theoretical value to fulfill for the metrics.
    empirical_val : float
        The actual realized value for the metric.
    group_size : int
        The smallest group size over which the metric is evaluated.
    metric_name : str, optional
        The metric's name, by default "". This is used for debugging purposes.
    less_or_equal : bool, optional
        Whether a lower empirical value compared to theory is fine, by default
        False.
    """
    if less_or_equal and empirical_val <= theory_val:
        return True

    assert np.isclose(
        theory_val,
        empirical_val,
        atol=get_metric_abs_tolerance(group_size),
        rtol=0.01,
    ), f"> '{metric_name}' mismatch; expected {theory_val:.3}; got {empirical_val:.3};"
