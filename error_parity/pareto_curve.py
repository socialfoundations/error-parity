"""Utils for computing the fairness-accuracy Pareto frontier of a classifier.

TODO:
- review this whole file and make a minimal example for the README;
- plus make a notebook example (repurposed from EDS);

Based on: https://github.com/socialfoundations/error-parity/blob/supp-materials/scripts/utils/postprocessing.py
"""

from __future__ import annotations

import os
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .threshold_optimizer import RelaxedThresholdOptimizer
from .evaluation import evaluate_predictions, evaluate_predictions_bootstrap
from ._commons import join_dictionaries, get_cost_envelope


def fit_and_evaluate_postprocessing(
        predictor: callable,
        tolerance: float,
        fit_data: tuple,
        eval_data: tuple or dict[tuple],
        fairness_constraint: str = "equalized_odds",
        false_pos_cost: float = 1.,
        false_neg_cost: float = 1.,
        max_roc_ticks: int = 200,
        seed: int = 42,
        y_fit_pred_scores: np.ndarray = None,    # pre-computed predictions on the fit data
        bootstrap: bool = True,
        bootstrap_kwargs: dict = None,
    ) -> dict[str, dict]:
    """Fit and evaluate a postprocessing intervention on the given predictor.

    Parameters
    ----------
    predictor : callable
        The callable predictor to fit postprocessing on.
    tolerance : float
        The tolerance (or slack) for fairness constraint fulfillment.
    fit_data : tuple
        The data used to fit postprocessing.
    eval_data : tuple or dict[tuple]
        The data or sequence of data to evaluate postprocessing on.
        If a tuple is provided, will call it "eval" data in the returned results
        dictionary; if a dict is provided, will assume {<key_1>: <data_1>, ...}.
    fairness_constraint : str, optional
        The name of the fairness constraint to use, by default "equalized_odds".
    false_pos_cost : float, optional
        The cost of a false positive error, by default 1.
    false_neg_cost : float, optional
        The cost of a false negative error, by default 1.
    max_roc_ticks : int, optional
        The maximum number of ticks (precision) to use when computing
        group-specific ROC curves, by default 200.
    seed : int, optional
        The random seed, by default 42
    y_fit_pred_scores : np.ndarray, optional
        The pre-computed predicted scores for the `fit_data`; if provided, will
        avoid re-computing these predictions for each function call.
    bootstrap : bool, optional
        Whether to use bootstrapping when computing metric results for
        postprocessing, by default True.
    bootstrap_kwargs : dict, optional
        Any extra arguments to pass on to the bootstrapping function, by default
        None.

    Returns
    -------
    results : dict[str, dict]
        A dictionary of results, whose keys are the data type, and values the
        metric values obtained by postprocessing on that data type.

        For example:
        >>> {
        >>>     "validation": {"accuracy": 0.7, ...},
        >>>     "test": {"accuracy": 0.65, ...},
        >>> }
    """
    clf = RelaxedThresholdOptimizer(
        predictor=predictor,
        constraint=fairness_constraint,
        tolerance=tolerance,
        false_pos_cost=false_pos_cost,
        false_neg_cost=false_neg_cost,
        max_roc_ticks=max_roc_ticks,
        seed=seed,
    )

    # Unpack data
    X_fit, y_fit, s_fit = fit_data

    logging.basicConfig(level=logging.WARNING, force=True)
    clf.fit(X=X_fit, y=y_fit, group=s_fit, y_scores=y_fit_pred_scores)

    results = {}
    # (Theoretical) fit results
    results["fit-theoretical"] = {
        "accuracy": 1 - clf.cost(1.0, 1.0),
        fairness_constraint: clf.constraint_violation(),
    }

    ALLOWED_ABS_ERROR = 1e-5
    assert clf.constraint_violation() <= tolerance + ALLOWED_ABS_ERROR, \
        f"Got {clf.constraint_violation()} > {tolerance}"

    # Map of data_type->data_tuple to evaluate postprocessing on
    data_to_eval = (
        {"fit": fit_data}
        | (eval_data if isinstance(eval_data, dict) else {"test": eval_data})
    )

    def _evaluate_on_data(data: tuple):
        """Helper function to evaluate on the given data tuple."""
        X, Y, S = data

        if bootstrap:
            kwargs = bootstrap_kwargs or dict(
                confidence_pct=95,
                seed=seed,
                threshold=0.50,
            )

            eval_func = partial(
                evaluate_predictions_bootstrap,
                **kwargs,
            )

        else:
            eval_func = partial(
                evaluate_predictions,
                threshold=0.50,
            )

        return eval_func(
            y_true=Y,
            y_pred_scores=clf.predict(X, group=S),
            sensitive_attribute=S,
        )

    # Empirical results
    for data_type, data_tuple in data_to_eval.items():
        results[data_type] = _evaluate_on_data(data_tuple)

    return results


def compute_postprocessing_curve(
        model: object,
        fit_data: tuple,
        eval_data: tuple or dict[tuple],
        fairness_constraint: str = "equalized_odds",
        bootstrap: bool = True,
        tolerance_tick_step: float = 1e-2,
        tolerance_ticks: list = None,
        predict_method: str = "predict_proba",
        n_jobs: int = None,
        **kwargs) -> pd.DataFrame:
    """Computes the fairness and performance of the given classifier after
    adjusting (postprocessing) for varying levels of fairness tolerance.

    Parameters
    ----------
    model : object
        The model to use.
    fit_data : tuple
        Data triplet to use to fit postprocessing intervention, (X, Y, S),
        respectively containing the features, labels, and sensitive attribute.
    eval_data : tuple or dict[tuple]
        Data triplet to use to evaluate postprocessing intervention on (same
        format as `fit_data`), or a dictionary of <data_name>-><data_triplet>
        containing multiple datasets to evaluate on.
    fairness_constraint : str, optional
        _description_, by default "equalized_odds"
    bootstrap : bool, optional
        Whether to compute uncertainty estimates via bootstrapping, by default
        False.
    tolerance_tick_step : float, optional
        Distance between constraint tolerances in the adjustment curve, by
        default 1e-2.
    tolerance_ticks : list, optional
        List of constraint tolerances to use when computing adjustment curve.
        If not provided, will use `tolerance_tick_step` to construct evenly-
        -spaced ticks.
    predict_method : str, optional
        Which method to call to obtain predictions out of the given model.
    n_jobs : int, optional
        Number of parallel jobs to use, if omitted will use `os.cpu_count()-1`.

    Returns
    -------
    postproc_results_df : pd.DataFrame
        A DataFrame containing the results, one row per tolerance tick.
    """
    def _func_call(tol: float):
        try:
            return fit_and_evaluate_postprocessing(
                predictor=lambda X: getattr(model, predict_method)(X)[:, -1],
                tolerance=tol,
                fit_data=fit_data,
                eval_data=eval_data,
                fairness_constraint=fairness_constraint,
                bootstrap=bootstrap,
                **kwargs)

        except Exception as exc:
            logging.error(f"FAILED fit_relaxed_postprocessing with `tolerance={tol}`: {exc}")

        return {}   # return empty dictionary

    # If n_jobs not provided: use number of CPU cores - 1
    if n_jobs is None:
        n_jobs = max(os.cpu_count() - 1, 1)
    logging.info(f"Using `n_jobs={n_jobs}` to compute adjustment curve.")

    from tqdm.auto import tqdm
    tolerances = tolerance_ticks if tolerance_ticks is not None else np.arange(0.0, 1.0, tolerance_tick_step)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        func_call_results = list(
            tqdm(
                executor.map(_func_call, tolerances),
                total=len(tolerances),
            )
        )

    results = dict(zip(tolerances, func_call_results))
    return _parse_postprocessing_curve(results)


def _parse_postprocessing_curve(postproc_curve_dict: dict) -> pd.DataFrame:
    """Parses the postprocessing curve dictionary results into a pd.DataFrame.

    Parameters
    ----------
    postproc_curve_dict : dict
        The result of computing the postprocessing adjustment curve on a model.

    Returns
    -------
    postproc_results_df : pd.DataFrame
        A DataFrame containing the results for each tolerance value.
    """
    return pd.DataFrame([
        join_dictionaries(
            {
                "tolerance": float(tol),
            },

            *[{
                f"{metric_name}_{data_type}": metric_value
                for data_type, results in results_at_tol.items()
                for metric_name, metric_value in results.items()
            }]
        )
        for tol, results_at_tol in postproc_curve_dict.items()
    ])


def get_envelope_of_postprocessing_frontier(
        postproc_results_df: pd.DataFrame,
        perf_col: str = "accuracy_mean_test",
        disp_col: str = "equalized_odds_diff_mean_test",
        constant_clf_perf: float = 0.5,
        constant_clf_disp: float = 0.0,
    ) -> np.ndarray:
    """Computes points in envelope of the given postprocessing frontier results.

    Parameters
    ----------
    postproc_results_df : pd.DataFrame
        The postprocessing frontier results DF.
    perf_col : str, optional
        Name of the column containing performance results, by default "accuracy_mean_test"
    disp_col : str, optional
        Name of column containing disparity results, by default "equalized_odds_diff_mean_test"
    constant_clf_perf : float, optional
        The performance of a dummy constant classifier (in the same metric as
        `perf_col`), by default 0.5.
    constant_clf_disp : float, optional
        The disparity of a dummy constant classifier (in the same metric as
        `disp_col`), by default 0.0; assumes a constant classifier fulfills
        fairness!

    Returns
    -------
    np.ndarray
        A 2-D array containing the points in the convex hull of the Pareto curve.
    """
    # Add bottom left point (postprocessing to constant classifier is always trivial)
    postproc_results_df = pd.concat(
        objs=(
            pd.DataFrame(
                {
                    perf_col: [constant_clf_perf],
                    disp_col: [constant_clf_disp],
                },
            ),
            postproc_results_df,
        ),
        ignore_index=True,
    )

    # Make costs array
    costs = np.stack(
        (
            1 - postproc_results_df[perf_col],
            postproc_results_df[disp_col],
        ),
        axis=1,
    )

    # Get points in the envelope of the Pareto frontier
    costs_envelope = get_cost_envelope(costs)

    # Get original metric values back
    adjustment_frontier = np.stack(
        (
            1 - costs_envelope[:, 0],     # flip perf values back to original metric
            costs_envelope[:, 1],         # keep disparity values (were already costs)
        ),
        axis=1,
    )

    # Sort by x-axis to plot properly (should already be sorted but, just making sure...)
    adjustment_frontier = adjustment_frontier[np.argsort(adjustment_frontier[:, 0])]
    return adjustment_frontier


def compute_inner_and_outer_adjustment_ci(
        postproc_results_df,
        perf_metric: str,
        disp_metric: str,
        data_type: str = "test",    # by default, fetch results on test data
        constant_clf_perf: float = None,
    ) -> tuple:
    """Computes the interior/inner and exterior/outer adjustment curves,
    corresponding to the confidence intervals (by default 95% c.i.).

    Returns
    -------
    postproc_results_df : tuple[np.array, np.array, np.array]
        A tuple containing (xticks, inner_yticks, outer_yticks).
    """
    # Make INTERIOR/UPPER envelope of the adjustment frontier
    # (i.e., the WORST points, with lower performance and higher disparity)
    interior_adjusted_df = postproc_results_df.copy()
    interior_adjusted_df[perf_metric] = \
        interior_adjusted_df[f"{perf_metric}_low-percentile_{data_type}"]

    interior_adjusted_df[disp_metric] = \
        interior_adjusted_df[f"{disp_metric}_high-percentile_{data_type}"]

    # Make OUTER/BOTTOM envelope of the adjustment frontier
    # (i.e., the BEST points, with higher performance and lower disparity)
    outer_adjusted_df = postproc_results_df.copy()
    outer_adjusted_df[perf_metric] = \
        outer_adjusted_df[f"{perf_metric}_high-percentile_{data_type}"]

    outer_adjusted_df[disp_metric] = \
        outer_adjusted_df[f"{disp_metric}_low-percentile_{data_type}"]

    # Process each frontier
    interior_adj_frontier = get_envelope_of_postprocessing_frontier(
        interior_adjusted_df,
        perf_col=perf_metric,
        disp_col=disp_metric,
        constant_clf_perf=constant_clf_perf,
    )
    outer_adj_frontier = get_envelope_of_postprocessing_frontier(
        outer_adjusted_df,
        perf_col=perf_metric,
        disp_col=disp_metric,
        constant_clf_perf=constant_clf_perf,
    )

    # Create functions that interpolate points within each frontier (interior or outer)
    # Because ax.fill_between requires both lines to have the same xticks
    from scipy.interpolate import interp1d
    interior_adj_func = interp1d(
        x=interior_adj_frontier[:, 0], y=interior_adj_frontier[:, 1],
        bounds_error=False,
        fill_value=(
            np.min(interior_adj_frontier[:, 1]),
            np.max(interior_adj_frontier[:, 1]),
        ),
    )
    outer_adj_func = interp1d(
        x=outer_adj_frontier[:, 0], y=outer_adj_frontier[:, 1],
        bounds_error=False,
        fill_value=(
            np.min(outer_adj_frontier[:, 1]),
            np.max(outer_adj_frontier[:, 1]),
        ),
    )

    # Get common xticks (union)
    adjustment_frontier_xticks = np.sort(np.unique(np.hstack(
        (interior_adj_frontier[:, 0], outer_adj_frontier[:, 0])
    )))

    interior_frontier_yticks = np.array([interior_adj_func(x) for x in adjustment_frontier_xticks])
    outer_frontier_yticks = np.array([outer_adj_func(x) for x in adjustment_frontier_xticks])

    return adjustment_frontier_xticks, interior_frontier_yticks, outer_frontier_yticks
