#!/usr/bin/env python3
"""This file will run the main T5.6 experiment with the specified parameters.

> Main experiment for the "Unprocessing Seven Years of Algorithmic Fairness"
paper;
> It will: load data, sample hyperparams, train base, unprocess+train, evaluate,
save results;
> Each call will run for a single model; this script is meant to be called
multiple (100s of) times as separate cluster jobs.
"""
import sys
import logging
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

from error_parity import RelaxedThresholdOptimizer

from utils import (
    RESULTS_JSON_FILE_NAME,
    ARGS_JSON_FILE_NAME,
    MODEL_KWARGS_JSON_NAME,
    RESULTS_MODEL_PKL_NAME,
    MODEL_PREDICTIONS_TRAIN,
    MODEL_PREDICTIONS_TEST,
    MODEL_PREDICTIONS_VALIDATION,
)
from utils.models import (
    fit_model,
    evaluate_model,
    compute_model_predictions,
    get_equalized_odds_clf_metrics,
    fit_transform_preprocessor,
)
from utils.temporal import TimeIt
from utils.files import save_json, save_pickle
from utils.hyperparameters import instantiate_random_model_config


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(description="Run single T5.6 paper experiment on a folktables dataset")

    # List of command-line arguments, with type and helper string
    cli_args = [
        ("--dataset",           str, "[string] Name of the dataset (e.g., the name of the folktables task, or 'MEPS')"),
        ("--base-model-yaml",   str, "[string] Path of YAML file with kwargs config for the standard / base model"),
        ("--meta-model-yaml",   str, "[string] Path of YAML file with kwargs config for the meta / ensemble model", False),
        ("--preprocessor-yaml", str, "[string] Path of YAML file with kwargs config for the pre-processor model", False),
        ("--seed",              int, "[int] Random seed to use for the experiment; this will also be used as the experiment ID when saving results!"),
        ("--data-dir",          str, "[string] Path of ACS data directory"),
        ("--save-dir",          str, "[string] Path of directory where to save results from these experiments"),
        ("--sens-attr",         str, "[string] Column with sensitive attribute", False),
    ]

    for arg in cli_args:
        parser.add_argument(
            arg[0],
            type=arg[1],
            help=arg[2],
            required=(arg[3] if len(arg) > 3 else True),  # NOTE: required by default
        )

    # Add special boolean arguments
    parser.add_argument("--one-hot", action="store_true", help="Whether to use one-hot-encoded data")
    parser.add_argument("--verbose", action="store_true", help="Whether to output verbose logs")

    return parser


def run_experiment(
        model: callable,
        results_dir: Path,
        train_data: tuple,
        test_data: tuple,
        validation_data: tuple = None,
        unadjust_on_validation: bool = True,
        save_pkl_to_work_fs: bool = True,
        seed: int = 42,
    ) -> dict:
    """Runs one experiment trial: trains, evaluates, and saves the results.

    Specifically, this script will:
    1. Fit the given model object on the given training data;
    2. Evaluate on both validation and test data;
    3. Unadjust the model (on train or validation data) to maximize accuracy;
    4. Evaluate unadjusted version on validation and test data;
    5. Save all results to disk, including:
        - pickle of trained model and unadjusted model;
        - validation and test results;
        - process and wall-clock times for all fit operations.

    Parameters
    ----------
    model : callable
        An ML model object, ready to be fitted on a given dataset.
    results_dir : Path
        The path to a directory where the experiment's results will be saved to,
        including model pickles.
    train_data : tuple
        A tuple (X, Y, S) of training data containing features, labels, and
        group membership, respectively.
    test_data : tuple
        A tuple (X, Y, S) of test data.
    validation_data : tuple, optional
        A tuple (X, Y, S) of validation data.
    unadjust_on_validation : bool, optional
        Whether to unadjust the trained model on validation data, by default
        False (i.e., will unadjust on training data by default).
    save_pkl_to_work_fs : bool, optional
        Whether to save the model pickle to the /work file-system instead of the
        given results_dir path.
    seed : int, optional
        A random seed for reproducibility.
        It's a required argument as this seed breaks the symmetry between
        different experiments whose remaining arguments are the same.

    Returns
    -------
    results : dict
        The performance and fairness results of the experiment (onm validation
        and test data), as well as total time to fit the model.
    """
    if unadjust_on_validation:
        assert validation_data is not None

    # Unpack X, Y, S data
    X_train, y_train, s_train = train_data

    # Fit model on train data
    with TimeIt(results_dir / "time-to-fit-original-model.txt", name="fit_model"):
        fit_metadata = fit_model(model, X_train, y_train, s_train)

    # Save pickle of trained model to /work instead of /fast
    model_pkl_save_path = results_dir / RESULTS_MODEL_PKL_NAME
    if save_pkl_to_work_fs:
        # NOTE: this is a bit too hard-coded, but works for now...
        model_pkl_save_path = Path("/work/acruz") / Path(*model_pkl_save_path.parts[-3:])
        model_pkl_save_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(obj=model, path=model_pkl_save_path)

    # Evaluate model on train/test/val
    eval_train = evaluate_model(
        model, train_data,
        predictions_save_path=results_dir / MODEL_PREDICTIONS_TRAIN,
        bootstrap=True,
    )
    eval_test = evaluate_model(
        model, test_data,
        predictions_save_path=results_dir / MODEL_PREDICTIONS_TEST,
        bootstrap=True,
    )
    eval_val = evaluate_model(
        model, validation_data,
        predictions_save_path=results_dir / MODEL_PREDICTIONS_VALIDATION,
        bootstrap=True,
    ) if validation_data is not None else None

    # Unadjust model on validation data (post-process for max. accuracy)
    unadjusted_model = RelaxedThresholdOptimizer(
        predictor=partial(compute_model_predictions, model),
        tolerance=1.0,  # max. tolerance (ignore fairness constraint)
        seed=seed,
    )

    with TimeIt(results_dir / "time-to-fit-postprocessor.txt", name="fit_adjustment"):
        X_for_unproc, y_for_unproc, s_for_unproc = validation_data if unadjust_on_validation else train_data
        unadjusted_model.fit(X=X_for_unproc, y=y_for_unproc, group=s_for_unproc)

    # Evaluate unadjusted classifier on train/test/val
    unadjusted_eval_solution = get_equalized_odds_clf_metrics(unadjusted_model)
    unadjusted_eval_train = evaluate_model(unadjusted_model, train_data)
    unadjusted_eval_test = evaluate_model(unadjusted_model, test_data)
    unadjusted_eval_val = evaluate_model(
        unadjusted_model,
        validation_data,
    ) if validation_data is not None else None

    # Construct results dictionary
    def process_results(results: dict, model_name: str, data_type: str):
        assert data_type in ("train", "test", "validation", "solution")
        return {
            f"{model_name}_{data_type}_{key}": val for key, val in results.items()
        }

    all_results = (
        # Training times
        fit_metadata

        # Original model on TRAIN and TEST
        | process_results(eval_train, "original", "train")
        | process_results(eval_test, "original", "test")

        # Unadjusted model on TRAIN and TEST, and the theoretical solution
        | process_results(unadjusted_eval_solution, "unadjusted", "solution")
        | process_results(unadjusted_eval_train, "unadjusted", "train")
        | process_results(unadjusted_eval_test, "unadjusted", "test")
    )

    if validation_data:
        all_results |= (
            # Results for OG model on validation data
            process_results(eval_val, "original", "validation")

            # Unadjusted model on VALIDATION (LP solution and empirical results)
            | process_results(unadjusted_eval_val, "unadjusted", "validation")
        )

    ### Save all results to disk (evaluation results and model pickles)
    # Ensure folder exists
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=False, exist_ok=True)

    # Save json of results dict
    save_json(all_results, path=results_dir / RESULTS_JSON_FILE_NAME)

    # Log main results
    msg = (
        f"\n** Test results **\n"
        f"Model accuracy:       {eval_test['accuracy']:.1%};\n"
        f"Model eq. odds diff:  {eval_test['equalized_odds_diff']:.1%};\n"
        f"Unadj. accuracy:      {unadjusted_eval_test['accuracy']:.1%};\n"
        f"Unadj. eq. odds diff: {unadjusted_eval_test['equalized_odds_diff']:.1%};\n\n"
    )
    logging.info(msg)
    logging.info(f"Saved experiment (exp-id: {seed}) results to '{str(results_dir)}'")

    return all_results


if __name__ == '__main__':
    """Prepare and launch a single experiment trial.

    This script serves as a helper so that some external script can launch
    multiple cluster jobs by submitting this script multiple times with
    different random seeds.
    """

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()
    print("Received the following cmd-line args:", args, end="\n\n")
    print(f"Current python executable: '{sys.executable}'\n")

    # Set logging level to INFO if verbose
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    # Check if args.save_dir exists
    save_dir = Path(args.save_dir).resolve()
    assert save_dir.exists() and save_dir.is_dir(), (
        f"The directory --save-dir='{str(save_dir)}' does not exist."
    )

    # Create a sub-folder to store results from *this* experiment
    from utils.hashing import get_unique_experiment_name
    exp_name = get_unique_experiment_name(**vars(args))
    save_dir = save_dir / exp_name
    save_dir.mkdir(parents=False, exist_ok=True)
    logging.info(f"Saving `exp-id={args.seed}` to `save-dir={str(save_dir)}`")

    # Save experiment's cmd-line args to save-dir
    save_json(vars(args), path=save_dir / ARGS_JSON_FILE_NAME)

    ###
    # Load data
    ###
    from utils.datasets import get_default_sensitive_col, load_data
    sensitive_col = args.sens_attr or get_default_sensitive_col(args.dataset)
    logging.info(f"Loading data from '{args.data_dir}'.")

    all_data = load_data(
        dir_path=args.data_dir,
        dataset_name=args.dataset,
        sensitive_col=sensitive_col,
        one_hot=args.one_hot,
    )

    # If data is one-hot encoded, sensitive attributes will be as well
    numeric_sensitive_cols = [
        col for col in all_data["train"][0].columns
        if col.startswith(sensitive_col) and col != sensitive_col
    ]
    if len(numeric_sensitive_cols) > 0 and args.one_hot:
        logging.info(
            f"Using the following numeric sensitive columns for model fitting: "
            f"{numeric_sensitive_cols}")

    # (Optional) preprocess data
    if args.preprocessor_yaml:
        logging.info(
            f"Using preprocessor '{Path(args.preprocessor_yaml).stem}' on "
            f"input data.")

        all_data = fit_transform_preprocessor(
            preprocessor_yaml=args.preprocessor_yaml,
            datasets=all_data,
            sensitive_cols=(
                numeric_sensitive_cols
                if (args.one_hot and len(numeric_sensitive_cols) > 0)
                else sensitive_col
            ),
            seed=args.seed,
            results_dir=save_dir,
        )

    # Unpack data
    train_data = all_data["train"]
    test_data = all_data["test"]
    validation_data = all_data.get("validation", None)

    # Generate a random hyperparameter config from the given YAML file(s)
    model_obj = instantiate_random_model_config(
        hyperparameter_space_path=args.base_model_yaml,
        meta_hyperparameter_space_path=args.meta_model_yaml,
        seed=args.seed,
        save_file_path=save_dir / MODEL_KWARGS_JSON_NAME,
    )

    # Run experiment (train, evaluate, save)
    run_experiment(
        model=model_obj,
        results_dir=save_dir,
        train_data=train_data,
        test_data=test_data,
        validation_data=validation_data,
        unadjust_on_validation=True,
        seed=args.seed,
    )

    print(
        "\n\n"
        "********************************\n"
        "Finished experiment successfully\n"
        "********************************\n"
    )
