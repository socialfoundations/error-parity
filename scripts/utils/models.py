"""Set of useful functions to form a common API from all the unconstrained
and constrained ML methods.
"""
import time
import logging
from pathlib import Path
from inspect import signature

import numpy as np
from error_parity import RelaxedThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient
from error_parity.evaluation import evaluate_predictions, evaluate_predictions_bootstrap

from utils.temporal import TimeIt
from utils.hyperparameters import instantiate_random_model_config, instantiate_given_model_config
from utils import PREPROCESSOR_KWARGS_JSON_NAME
from utils.files import load_json


def fit_model(model, X_train, y_train, s_train=None, verbose=True) -> dict:
    sig = signature(model.fit)
    fit_metadata = {
        "process-time": time.process_time(),
        "wall-time": time.time(),
    }

    # compatibility with fairgbm
    if 'constraint_group' in sig.parameters:
        logging.info(
            f"Using `constraint_group=s_train` as a positional argument in "
            f"{type(model)}.fit(.)")
        model.fit(X_train, y_train, constraint_group=s_train)

    # compatibility with fairlearn
    elif 'sensitive_features' in sig.parameters:
        logging.info(
            f"Using `sensitive_features=s_train` as a positional argument in "
            f"{type(model)}.fit(.)")
        model.fit(X_train, y_train, sensitive_features=s_train)

    # TODO: add adhoc compatibility with other fairness-aware libraries here
    # else, train without sensitive attribute data
    else:
        if s_train is not None and verbose:
            # TODO: check if this warning is visible when fitting unconstrained models, it should be!

            logging.warning(
                f"Can't figure out how to use sensitive_attribute data for "
                f"training with object of type '{type(model)}'."
            )

        model.fit(X_train, y_train)

    # Save time to fit
    fit_metadata["process-time"] = time.process_time() - fit_metadata["process-time"]
    fit_metadata["wall-time"] = time.time() - fit_metadata["wall-time"]

    logging.info(
        f"Took {fit_metadata['wall-time']:.1f}s to train object of type {type(model)} on "
        f"data of shape {X_train.shape}.")

    logging.info(f"Trained model: {model}.")

    # return model
    return fit_metadata


def fit_transform_preprocessor(
        preprocessor_yaml: Path,
        datasets: dict,
        sensitive_cols: str | int | list[str | int],
        seed: int,
        results_dir: Path,
    ) -> dict:
    """Instantiated a preprocessor model according to the provided YAML config,
    and returns the pre-processed input data.

    Parameters
    ----------
    preprocessor_yaml : Path
        Path to a YAML file that details the configurations for the preprocessor
        model.
    datasets : dict
        A dictionary containing the input data (with possible keys 'train', 
        'validation', and 'test').
    sensitive_cols : str | int | list[str | int]
        The name (or index) of the column(s) containing the sensitive attribute(s).
    seed : int
        A random seed. Mandatory in order to break symmetry between experiments.
    results_dir : Path
        Path to the results directory.

    Returns
    -------
    dict
        A dict following the same structure as the provided `datasets` arg,
        but with the pre-processed datasets.
    """

    # Check if preprocessor kwargs were already sampled
    preprocessor = None
    preprocessor_kwargs_path = results_dir / PREPROCESSOR_KWARGS_JSON_NAME
    if preprocessor_kwargs_path.exists():
        try:
            preproc_hyperparams = load_json(preprocessor_kwargs_path)
            preprocessor = instantiate_given_model_config(**preproc_hyperparams)
        except Exception as err:
            logging.error(f"Error instantiating preprocessor from kwargs at {preprocessor_kwargs_path} :: {err}")

    # If not, randomly sample config and instantiate preprocessor model
    if preprocessor is None:
        preprocessor = instantiate_random_model_config(
            hyperparameter_space_path=preprocessor_yaml,
            seed=seed,
            sensitive_feature_ids=sensitive_cols if isinstance(sensitive_cols, list) else [sensitive_cols],
            save_file_path=preprocessor_kwargs_path,
        )

    # Fit to training data
    X_train, y_train, *_s_train = datasets["train"]

    with TimeIt(results_dir / "time-to-fit-preprocessor.txt", name="fit_preprocessor"):
        preprocessor.fit(X_train, y_train)

    # Transform/preprocess all given data
    return {
        key: (preprocessor.transform(X), *others) for key, (X, *others) in datasets.items()
    }


def compute_model_predictions(
        model,
        X_eval,
        s_eval=None,
        predictions_save_path: Path = None,
        **kwargs):

    # Check if predictions were already computed and saved to disk
    if predictions_save_path and predictions_save_path.exists():
        return np.load(predictions_save_path)

    # If the model is callable, use this to compute predictions
    if isinstance(model, RelaxedThresholdOptimizer):
        y_pred = model.predict(X_eval, group=s_eval, **kwargs)

    elif isinstance(model, ExponentiatedGradient):
        y_pred = model._pmf_predict(X_eval, **kwargs)

    elif hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_eval, **kwargs)

    # TODO: add adhoc compatibility with other libraries here
    else:
        y_pred = model.predict(X_eval, **kwargs)

    # Keep only predictions for positive class
    if len(y_pred.shape) > 1:
        y_pred = y_pred[:, -1]

    # If `predictions_save_path` was provided, save to disk
    if predictions_save_path:
        np.save(predictions_save_path, y_pred)

    return y_pred


def get_equalized_odds_clf_metrics(clf: RelaxedThresholdOptimizer) -> dict:
    return {
        "accuracy": 1 - clf.cost(1.0, 1.0),
        "equalized_odds_diff": clf.equalized_odds_violation(),
        "fairness_constraint_violation": clf.constraint_violation(),
    }


def evaluate_model(
        model: object,
        eval_data: tuple,
        predictions_save_path: Path = None,
        bootstrap: bool = True,
        bootstrap_kwargs: dict = {},
    ) -> dict:

    X_eval, y_eval, s_eval = eval_data
    y_pred_scores = compute_model_predictions(
        model, X_eval=X_eval, s_eval=s_eval,
        predictions_save_path=predictions_save_path,
    )

    # Evaluate predictions
    results = evaluate_predictions(
        y_true=y_eval,
        y_pred_scores=y_pred_scores,
        sensitive_attribute=s_eval,
        return_groupwise_metrics=False,
        threshold=0.5,      # NOTE: threshold should be changed when l(0,1) != l(1,0)
    )

    # (Optional) Compute bootstrap results
    if bootstrap:
        bootstrap_results = evaluate_predictions_bootstrap(
            y_true=y_eval,
            y_pred_scores=y_pred_scores,
            sensitive_attribute=s_eval,
            threshold=0.5,
            **bootstrap_kwargs,
        )

        results.update(bootstrap_results)

    return results
