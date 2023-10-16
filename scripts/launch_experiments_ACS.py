#!/usr/bin/env python3
"""
Python script to launch condor jobs for all T5.6 ACS experiments.
"""
from pathlib import Path

from utils.experiments import ExperimentConfigs, launch_experiments_jobs

N_TRIALS = 50
VERBOSE = True

ACS_TASKS = (
    "ACSIncome",
    "ACSEmployment",
    "ACSMobility",
    "ACSTravelTime",
    "ACSPublicCoverage",
)

######################
# Useful directories #
######################
ROOT_DIR = Path("/fast/acruz")

# ACS data directory (contains datasets for all ACS tasks)
DATA_DIR = ROOT_DIR / "data/folktables/train=0.6_test=0.2_validation=0.2_max-groups=4"
# DATA_DIR = ROOT_DIR / "data/folktables/train=0.6_test=0.2_validation=0.2_max-groups=2"

# Directory to save results in (make sure it exists)
RESULTS_DIR = ROOT_DIR / f"ACS_T5.6-results_{DATA_DIR.name}_2023-09-06"
RESULTS_DIR.mkdir(exist_ok=True, parents=False)

# Directory for YAML config files
CONFIGS_DIR = Path("../hyperparameter_spaces").resolve() / "ACS"


if __name__ == '__main__':

    # Generate all experiments to run
    all_experiments = ExperimentConfigs.get_all_experiments(n_trials=N_TRIALS)

    # Common configs among all experiments
    common_exp_configs = dict(
        data_dir    = DATA_DIR,
        results_dir = RESULTS_DIR,
        configs_dir = CONFIGS_DIR,
    )

    # Log all experiments that we want to run
    num_experiments = len(all_experiments)
    print(
        f"\nLaunching the following experiments (n={num_experiments}) per "
        f"dataset (n={len(ACS_TASKS)}):\n")
    for i, exp in enumerate(all_experiments):
        print(f"{i}. {exp};")

    # For each ACS task
    for acs_task in ACS_TASKS:
        print(f"\n\nTASK: {acs_task}")

        for i, exp_obj in enumerate(all_experiments):
            print(f"{i}. Launching {exp_obj.n_trials} trial(s) for the experiment '{exp_obj}'")
            launch_experiments_jobs(
                dataset=acs_task,
                exp_obj=exp_obj,
                **common_exp_configs,
            )
