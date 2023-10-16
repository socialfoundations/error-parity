"""This script will refit the model at the given experiment folder.
"""
import os
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from utils.models import compute_model_predictions
from utils.notebook import load_experiment_data, load_or_refit_experiment_model
from utils.postprocessing import load_or_compute_adjustment_curve
from utils.constants import MODEL_PREDICTIONS_VALIDATION, MODEL_PREDICTIONS_TRAIN


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(
        description="Re-fit the model at the given experiment (and save pickle).")

    # Input dir argument
    parser.add_argument(
        "-i", "--experiment-dir",
        type=str,
        help="[string] The path to the experiment directory.",
        required=True,
    )

    # Optional args
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="How many separate threads to use to run this experiment; will default to `os.cpu_count()`.",
        required=False,
        default=os.cpu_count(),
    )

    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Will force refitting the trained model instead of loading .pkl from disk.",
    )

    parser.add_argument(
        "--skip-adjustment",
        action="store_true",
        help="Will skip computing the postprocessing adjustment curve.",
    )

    parser.add_argument(
        "--check-predictions",
        action="store_true",
        help="Will check whether model predictions and saved metrics match.",
    )

    return parser


if __name__ == "__main__":

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()
    print("Received the following cmd-line args:", args, end="\n\n")

    exp_dir = Path(args.experiment_dir)

    # 1. Load experiment data
    print("Loading data...")
    all_data = load_experiment_data(exp_dir)

    # 2. Load (or refit) model object (re-fit may be necessary)
    print("Refitting model...")
    model = load_or_refit_experiment_model(
        exp_dir,
        all_data=all_data,
        bootstrap=True,
        force_refit=args.force_refit,
        skip_predictions=(not args.check_predictions),
    )

    # 3. (Optionally) Compute adjustment curve
    if not args.skip_adjustment:
        print("Computing adjustment curve...")

        # Check whether postprocessing adjustment will be fit on validation or train
        if "validation" in all_data:
            fit_data = all_data["validation"]
            predictions_save_path = exp_dir / MODEL_PREDICTIONS_VALIDATION
        else:
            fit_data = all_data["train"]
            predictions_save_path = exp_dir / MODEL_PREDICTIONS_TRAIN

        # Pre-compute score predictions on fit data
        X_fit, y_fit, s_fit = fit_data
        y_scores_fit = compute_model_predictions(
            model, X_eval=X_fit, s_eval=s_fit,
            predictions_save_path=predictions_save_path,
        )

        # Compute postprocessing adjustment (or load from disk if it already exists)
        adjustment_results_df = load_or_compute_adjustment_curve(
            model=model,
            exp_dir=exp_dir,
            fit_data=fit_data,
            y_fit_pred_scores=y_scores_fit,
            eval_data={
                "validation": all_data.get("validation"),
                "test": all_data["test"],
            },
            n_jobs=args.n_jobs,
            fairness_constraint="equalized_odds",
            tolerance_ticks=np.hstack((
                np.arange(0.0, 0.5, 1e-2),
                np.arange(0.5, 1.0, 1e-1),
            )),
            bootstrap=True,
            load_if_available=(not args.force_refit),
        )

    print(f"Finished experiment re-run successfully for args {args}.")
