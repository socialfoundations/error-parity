#!/usr/bin/env python3
"""Compute postprocessing adjustment curve on the best models.
"""
import re
import sys
from pathlib import Path

import htcondor
import classad
import pandas as pd


JOB_BID = 25
JOB_CPUS = 16
JOB_MEMORY_GB = 64

# Number of trials used in the experiment
# > will search for a file detailing exactly which {n_trials} trials to use
N_TRIALS = 50


def launch_refit_model_job(exp_dir: Path):
    """Launches the cluster jobs to refit the model at the given experiment dir.
    """

    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(exp_dir / "postprocessing_$(Cluster).$(Process)")

    # Construct job description
    job_description = htcondor.Submit({
        "executable": f"{sys.executable}",  # correct env for the python executable
        "arguments": (
            "refit_experiment_model.py "
            f"--experiment-dir {str(exp_dir)} "
            f"--n-jobs {JOB_CPUS}"
        ),
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{JOB_CPUS}",
        "request_memory": f"{JOB_MEMORY_GB}GB",
        "request_disk": "2GB",
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "andre.cruz@tuebingen.mpg.de",
        "notification": "error",

        # Concurrency limits:
        "concurrency_limits": "user.t56_postprocessing:500",    # max 20 jobs in parallel

        "+MaxRunningPrice": 100,
        "+RunningPriceExceededAction": classad.quote("restart"),
    })

    # Submit `n_trials` jobs to the scheduler
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description)

    print(f"Launched process with cluster-ID={submit_result.cluster()}\n")


def _select_best_model(results_df: pd.DataFrame, select_on_data_type: str = "validation") -> str:
    """Returns the row of the best model according to some sorting criteria."""

    # Option used in the paper:
    HIGHER_IS_BETTER = True
    SELECTION_METRIC = "accuracy"
    MODEL_TYPE = "unadjusted"

    # # Another option:
    # HIGHER_IS_BETTER = False
    # SELECTION_METRIC = "squared_loss"
    # MODEL_TYPE = "original"

    return results_df.sort_values(
            f"{MODEL_TYPE}_{select_on_data_type}_{SELECTION_METRIC}",
            ascending=(not HIGHER_IS_BETTER),
    ).iloc[0]


if __name__ == "__main__":

    # Parent directory for all aggregated results
    parent_results_dir = Path("../results").resolve()

    # Sub-folder containing relevant results
    results_subfolders = [
        "MEPS_T5.6-results_train_23380_test_15675_val_10020_3-groups_2023-09-08",
        "ACS_T5.6-results_train=0.6_test=0.2_validation=0.2_max-groups=4_2023-09-06",
        "ACS_T5.6-results_train=0.6_test=0.2_validation=0.2_max-groups=2_2023-09-06",
    ]

    # Launch postprocessing adjustment for each unique dataset on each listed folder
    for curr_folder in results_subfolders:

        # Check for files matching the following pattern
        regex_str = r"^(?P<dataset>\w+)[.](?P<n_trials>\d+)-trials[.]csv$"
        regex_obj = re.compile(regex_str)

        for file in (parent_results_dir / curr_folder).iterdir():
            match = regex_obj.match(str(file.name))

            if match and int(match.group("n_trials")) == N_TRIALS:
                print(f"Launching postprocessing adjustment for dataset '{match.group('dataset')}'.")

                # Compute postprocessing curve for best model (including unprocessed)
                dataset_results_df = pd.read_csv(str(file), index_col=0)
                best_model_row = _select_best_model(dataset_results_df, select_on_data_type="validation")

                exp_dir = Path(best_model_row["results_dir_path"])
                launch_refit_model_job(exp_dir)

                # Plus, postprocessing for the best *unconstrained* model
                best_unconstrained_row = _select_best_model(
                    dataset_results_df[dataset_results_df["intervention"] == "None"],
                    select_on_data_type="validation")

                exp_dir_unconstrained = Path(best_unconstrained_row["results_dir_path"])
                if exp_dir_unconstrained.name != exp_dir.name:
                    launch_refit_model_job(exp_dir_unconstrained)
