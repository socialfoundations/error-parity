"""Utils for jupyter notebooks

"""
import os
import logging
from pathlib import Path
from functools import partial

import numpy as np
from scipy.spatial import qhull, ConvexHull

from error_parity import RelaxedThresholdOptimizer

from .datasets import load_data, get_default_sensitive_col
from .files import load_json, save_json, load_pickle, save_pickle
from .hyperparameters import instantiate_given_model_config
from .temporal import TimeIt
from .models import (
    fit_model,
    evaluate_model,
    fit_transform_preprocessor,
    compute_model_predictions,
    get_equalized_odds_clf_metrics,
)
from .constants import (
    ARGS_JSON_FILE_NAME,
    MODEL_KWARGS_JSON_NAME,
    RESULTS_JSON_FILE_NAME,
    RESULTS_MODEL_PKL_NAME,
    MODEL_PREDICTIONS_VALIDATION,
    MODEL_PREDICTIONS_TEST,
)


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Finds the pareto-efficient points.
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


def get_convexhull_indices(
        adjustment_results_df,
        perf_metric: str = "accuracy_fit",
        disp_metric: str = "equalized_odds_diff_fit",
    ) -> np.array:
    """Get indices of points in the convex hull."""
    costs = np.stack(
        (
            1 - adjustment_results_df[f"{perf_metric}"],
            adjustment_results_df[f"{disp_metric}"],
        ),
        axis=1)

    try:
        hull = ConvexHull(costs)
        return hull.vertices

    except qhull.QhullError as err:
        logging.error(f"Failed to compute ConvexHull with error: {err}")
        return np.arange(len(adjustment_results_df))      # Return all points as in the convex hull


def load_experiment_args(exp_dir: Path) -> dict:
    """Loads experiment cmd-line args.
    """
    return load_json(exp_dir / ARGS_JSON_FILE_NAME)


def load_experiment_data(exp_dir: str | Path) -> dict:
    """Loads the train/test/validation data used for the experiment
    at the given directory.
    """
    # Get experiment's details (e.g., task, and one-hot encoding)
    exp_args = load_experiment_args(Path(exp_dir))

    # Hardcoded data path when using /fast filesystem
    home_data_dir = Path(exp_args["data_dir"])

    # Gather experiment args
    dataset = exp_args["dataset"] if "dataset" in exp_args else exp_args["acs_task"]
    one_hot = exp_args.get("one_hot", False)
    preprocessor_yaml = exp_args.get("preprocessor_yaml", None)
    sensitive_col = exp_args.get("sens_attr") or get_default_sensitive_col(dataset)
    logging.info(f"Using column '{sensitive_col}' for sensitive attribute.")

    # Load appropriate dataset
    all_data = load_data(
        dir_path=home_data_dir,
        dataset_name=dataset,
        sensitive_col=sensitive_col,
        one_hot=one_hot,
    )

    # Apply pre-processing if needed
    if preprocessor_yaml:

        # If data is one-hot encoded, sensitive attributes will be as well
        numeric_sensitive_cols = [
            col for col in all_data["train"][0].columns
            if col.startswith(sensitive_col) and col != sensitive_col
        ]

        # (Optional) preprocess data
        logging.info(
            f"Using preprocessor '{Path(preprocessor_yaml).stem}' on "
            f"input data.")

        all_data = fit_transform_preprocessor(
            preprocessor_yaml=preprocessor_yaml,
            datasets=all_data,
            sensitive_cols=numeric_sensitive_cols if one_hot else sensitive_col,
            seed=exp_args["seed"],
            results_dir=Path(exp_dir),
        )

    return all_data


def load_or_refit_experiment_model(
        exp_dir: str | Path,
        all_data: dict = None,
        skip_predictions: bool = False,
        bootstrap: bool = True,
        force_refit: bool = False) -> object:
    """This function will try to load the model object corresponding to the 
    given experiment (at the given `exp_dir` directory). If no such model 
    exists, will re-fit the model on the train data.

    It will:
    1. check if the model pickle is saved in disk;
     1.1. load pickle (if it exists);
    2. else:
     2.1. load model hyperparameters/kwargs;
     2.2. re-fit the model on the train dataset;
     2.3. save pickle to disk;
    3. finally, will check if model metrics match saved metrics;

    Parameters
    ----------
    exp_dir : str | Path
        Path to the experiment directory where the model details are, and the 
        model pickle will be saved to.
    all_data : dict, optional
        A pre-loaded dataset to re-fit the model (if needed), by default None.
    skip_predictions : bool, optional
        If True, will skip predictions checks. If False (default), will compute
        model predictions, check if results match saved results, and update
        results file in disk.
    bootstrap : bool, optional
        Whether to compute results with bootstrapping.

    Returns
    -------
    object
        The loaded model.
    """

    exp_dir = Path(exp_dir)
    assert exp_dir.exists()

    exp_args = load_experiment_args(Path(exp_dir))

    # Load appropriate dataset if none was provided
    if all_data is None:
        all_data = load_experiment_data(exp_dir)

    # Load model pickle...
    model_pkl_path = exp_dir / RESULTS_MODEL_PKL_NAME
    # model_pkl_path_in_work_fs = Path(model_pkl_path.replace("/fast/", "/work/"))    # TODO
    # TODO: look for the model pickle in /work/acruz as well, and save to work if save_to_work_fs=True
    model_pkl_path_work_fs = Path("/work/acruz") / exp_dir.parent.name / exp_dir.name / RESULTS_MODEL_PKL_NAME

    if not model_pkl_path.exists():
        model_pkl_path = model_pkl_path_work_fs

    # 1. Check if model pickle exists
    model = None
    if model_pkl_path.exists():
        print(f"Found existing model pickle at '{model_pkl_path}'")
        try:
            model = load_pickle(model_pkl_path)
        except Exception as err:
            logging.error(
                f"Error when loading model pickle at '{model_pkl_path}': \n"
                f"{err};")

    # 2. If not, and if there are no saved predictions, re-fit model
    val_preds_path = exp_dir / MODEL_PREDICTIONS_VALIDATION
    test_preds_path = exp_dir / MODEL_PREDICTIONS_TEST

    # Refit model IF:
    if force_refit or (
            model is None and (
            not val_preds_path.exists()
            or not test_preds_path.exists()
        )):

        # 2.1. Load model hyperparameters and instantiate object
        print("Couldn't find model pickle, refitting model...")
        model_hyperparams = load_json(exp_dir / MODEL_KWARGS_JSON_NAME)

        # NOTE: I'm not sure if changing the `n_jobs` changes the predictions...
        # model_hyperparams["n_jobs"] = -1

        model = instantiate_given_model_config(**model_hyperparams)

        # 2.2. Re-fit model to training data
        X_train, y_train, s_train = all_data["train"]

        with TimeIt(exp_dir / "time-to-fit-original-model.refit.txt", name="refit_model"):
            fit_metadata = fit_model(model, X_train, y_train, s_train)
        print("Model re-fit metadata:\n", fit_metadata)

        print(f"Saving model pickle to disk at '{model_pkl_path}'")
        save_pickle(obj=model, path=model_pkl_path)

        # Delete previously saved predictions if they exist
        val_preds_path.unlink(missing_ok=True)
        test_preds_path.unlink(missing_ok=True)

    # 3. (Optional) Check that evaluation matches saved results
    if skip_predictions:    
        print("Skipping model prediction checks.")
        return model

    print("Checking if model predictions correspond to saved results...")
    # > evaluate current model object on test and validation
    eval_val = evaluate_model(
        model, all_data["validation"],
        predictions_save_path=val_preds_path,
        bootstrap=bootstrap)
    eval_test = evaluate_model(
        model, all_data["test"],
        predictions_save_path=test_preds_path,
        bootstrap=bootstrap)

    # Check predictions for unadjusted model as well
    unadjusted_model = RelaxedThresholdOptimizer(
        predictor=partial(compute_model_predictions, model),
        tolerance=1.0,  # max. tolerance (ignore fairness constraint)
        seed=exp_args["seed"])
    X_val, Y_val, S_val = all_data["validation"]
    unadjusted_model.fit(X_val, Y_val, group=S_val)

    # Evaluate unadjusted classifier on validation and test data
    unadjusted_eval_val_solution = get_equalized_odds_clf_metrics(unadjusted_model)
    unadjusted_eval_val = evaluate_model(unadjusted_model, all_data["validation"])
    unadjusted_eval_test = evaluate_model(unadjusted_model, all_data["test"])

    # > load saved results
    saved_results = load_json(exp_dir / RESULTS_JSON_FILE_NAME)

    # > check that all metrics match --- FOR ORIGINAL MODEL!
    metric_mismatch_count = 0
    for metric in (eval_val.keys() & eval_test.keys()):
        for curr_eval_typ_, curr_eval_values in zip(["validation", "test"], [eval_val, eval_test]):

            metric_col_name = f"original_{curr_eval_typ_}_{metric}"

            if (metric_col_name in saved_results and
                not np.isclose(
                    saved_results[metric_col_name],
                    curr_eval_values[metric])):

                logging.error(
                    f"Metric mismatch for OG {metric} ({curr_eval_typ_}); "
                    f"expected {saved_results[metric_col_name]:.4f}, "
                    f"got {curr_eval_values[metric]:.4f}.")
                metric_mismatch_count += 1

            # Regardless, update the previous results with the new results
            saved_results[metric_col_name] = curr_eval_values[metric]

    # > check that metrics match --- FOR UNADJUSTED MODEL!
    # TODO: adapt code to newest dict keys "fit-solution" and "fit"
    for curr_eval_typ_, curr_eval_values in zip(
            ["validation-solution", "validation", "test"],
            [unadjusted_eval_val_solution, unadjusted_eval_val, unadjusted_eval_test]
        ):

        for metric in curr_eval_values.keys():
            # Metric name in the saved_results
            metric_col_name = f"unadjusted_{curr_eval_typ_}_{metric}"

            if (metric_col_name in saved_results and
                not np.isclose(
                    saved_results[metric_col_name],
                    curr_eval_values[metric])):

                logging.error(
                    f"Metric mismatch for UNADJUSTED {metric} ({curr_eval_typ_}); "
                    f"expected {saved_results[metric_col_name]:.4f}, "
                    f"got {curr_eval_values[metric]:.4f}.")
                metric_mismatch_count += 1

            # Regardless, update the previous results with the new results
            saved_results[metric_col_name] = curr_eval_values[metric]

    # Log number of mismatched metrics (if any)
    if metric_mismatch_count > 0:
        print(f"Metric mismatch count: {metric_mismatch_count};")

        # Add the "deprecated" suffix to the previous results file
        os.rename(
            src=str(exp_dir / RESULTS_JSON_FILE_NAME),
            dst=str((exp_dir / RESULTS_JSON_FILE_NAME).with_suffix(".deprecated.json")),
        )

    else:
        print("All prediction metrics match saved results!")

    # Save updated results dictionary to disk (overwrite)
    save_json(
        obj=saved_results,
        path=exp_dir / RESULTS_JSON_FILE_NAME,
        overwrite=True,
    )

    return model
