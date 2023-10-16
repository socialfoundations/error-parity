import logging
from pathlib import Path

import pandas as pd
from error_parity.pareto_curve import compute_postprocessing_curve


def load_or_compute_adjustment_curve(
        model: object,
        exp_dir: str | Path,
        fit_data: tuple,
        eval_data: tuple or dict[str, tuple],
        bootstrap: bool = True,
        load_if_available: bool = True,
        **kwargs) -> pd.DataFrame:
    """Loads (or computes if necessary) the post-processing adjustment curve of
    the given model.

    Parameters
    ----------
    model : object
        A trained ML model.
    exp_dir : str | Path
        Path to the experiment directory where the model details are, and the 
        results will be saved to.
    bootstrap : bool, optional
        Whether to compute bootstrapped results (will return mean, stdev, and
        low-high percentiles for each metric).
    load_if_available : bool, optional
        Whether to load the pre-computed adjustment curve from disk, if one is
        available; True by default.
    kwargs : dict
        Remaining key-word arguments that will be used to compute the adjustment
        curve.

    Returns
    -------
    adj_curve_df : pd.DataFrame
        A pd.DataFrame detailing the adjustment curve for the given model.
    """

    # Path for pre-computed adjustment results (load if they exist)
    adj_curve_path = (
        Path(exp_dir) / (
            f"model-adjustment-curve"
            f"{'.bootstrap' if bootstrap else ''}"
            f".csv"
        ))

    # Check if results were already computed
    adj_curve_df = None
    if adj_curve_path.exists() and load_if_available:
        print(f"Loading pre-computed adjustment curve from '{adj_curve_path}'")
        try:
            adj_curve_df = pd.read_csv(adj_curve_path, index_col=0)
        except IOError as err:
            logging.error(f"Error when trying to read pre-saved adjustment curve: {err}")
            adj_curve_df = None

    # If not, compute and save
    if adj_curve_df is None:
        print(f"Computing adjustment curve and saving to disk at '{adj_curve_path}'")

        adj_curve_df = compute_postprocessing_curve(
            model=model,
            fit_data=fit_data,
            eval_data=eval_data,
            bootstrap=bootstrap,
            **kwargs,
        )

        # Save DF
        adj_curve_df.to_csv(adj_curve_path, index=True)

    return adj_curve_df
