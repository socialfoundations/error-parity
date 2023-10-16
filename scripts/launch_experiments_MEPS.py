#!/usr/bin/env python3
"""
Python script to launch condor jobs for all T5.6 ACS experiments.
"""
from pathlib import Path

from utils.experiments import ExperimentConfigs, launch_experiments_jobs

N_TRIALS = 50
VERBOSE = True

######################
# Useful directories #
######################
ROOT_DIR = Path("/fast/acruz")

# MEPS data directory
DATA_DIR = ROOT_DIR / "data" / "MEPS" / "train_23380_test_15675_val_10020_3-groups"

# Directory for YAML config files
CONFIGS_DIR = Path("../hyperparameter_spaces").resolve() / "MEPS"

# Directory to save results in (make sure it exists)
RESULTS_DIR = ROOT_DIR / f"MEPS_T5.6-results_{DATA_DIR.name}_2023-09-08"
RESULTS_DIR.mkdir(exist_ok=True, parents=False)


if __name__ == '__main__':

    # Generate all experiments to run
    all_experiments = ExperimentConfigs.get_all_experiments(n_trials=N_TRIALS)

    # Common configs among all experiments
    common_exp_configs = dict(
        dataset     = "MEPS",
        data_dir    = DATA_DIR,
        results_dir = RESULTS_DIR,
        configs_dir = CONFIGS_DIR,
    )

    # Log all experiments that we want to run
    num_experiments = len(all_experiments)
    print(f"\nLaunching the following MEPS experiments (n={num_experiments})")
    for i, exp_obj in enumerate(all_experiments):
        print(f"{i}. Launching {exp_obj.n_trials} trial(s) for the experiment '{exp_obj}'")
        launch_experiments_jobs(
            exp_obj=exp_obj,
            **common_exp_configs,
        )
