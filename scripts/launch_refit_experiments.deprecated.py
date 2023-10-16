#!/usr/bin/env python3
"""
Launch condor jobs to re-run missed parts of previous T5.6 experiments.
"""
import re
import sys
from pathlib import Path

import htcondor
import classad


JOB_BID = 25
JOB_CPUS = 10
# JOB_BID = 30
# JOB_CPUS = 50

USE_BOOTSTRAPPING = True
COMPUTE_ADJUSTMENT = True
USE_FAST_FS = True
FORCE_REFIT = True              # NOTE! danger!
# FORCE_REFIT = False


def launch_refit_model_job(exp_dir: Path):
    """Launches the cluster jobs to refit the model at the given experiment dir.
    """

    # Some heuristics to approximate memory needs
    job_memory_gb = 16
    if "random-forest" in exp_dir.name:
        job_memory_gb *= 2
    if "aif360-LFR" in exp_dir.name:
        job_memory_gb *= 2
    if "EG" in exp_dir.name or "GS" in exp_dir.name:
        job_memory_gb *= 8

    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(exp_dir / f"model-refit_$(Cluster).$(Process)")

    # Construct job description
    job_description = htcondor.Submit({
        "executable": f"{sys.executable}",  # correct env for the python executable
        "arguments": (
            "refit_experiment_model.py "
            f"--experiment-dir {str(exp_dir)} "
            f"{'--use-bootstrapping' if USE_BOOTSTRAPPING else ''} "
            f"{'--compute-adjustment' if COMPUTE_ADJUSTMENT else ''} "
            f"{'--use-fast-fs' if USE_FAST_FS else ''} "
            f"{'--force-refit' if FORCE_REFIT else ''} "
        ),
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{JOB_CPUS}",
        # "request_gpus": f"{JOB_GPUS}",
        "request_memory": f"{job_memory_gb}GB",
        "request_disk": "2GB",
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "andre.cruz@tuebingen.mpg.de",
        "notification": "error",

        # Concurrency limits:
        "concurrency_limits": "user.model_refit:40",     # 250 jobs in parallel

        "+MaxRunningPrice": 100,
        "+RunningPriceExceededAction": classad.quote("restart"),
    })

    # Submit `n_trials` jobs to the scheduler
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description)

    print(f"Launched process with cluster-ID={submit_result.cluster()}\n")


if __name__ == "__main__":
    from utils.files import load_json
    from utils.constants import RESULTS_MODEL_PKL_NAME

    N_TRIALS = 50

    # Results dir
    home_results_dir = Path("../results").resolve()
    home_results_dir /= "T5.6-results_train=0.6_test=0.2_validation=0.2_max-groups=4_2023-05"

    # Matches with all experiments that we're gonna use (for each task)
    regex_str = r"^experiments-to-use[.](?P<acs_task>\w+)[.](?P<n_trials>\d+)-trials[.]json$"

    # Matches with all experiments that we're gonna use with bootstrapping results (a subset)
    # regex_str = r"^experiments-to-use[.](?P<acs_task>\w+)[.](?P<n_trials>\d+)-trials[.]WITH-BOOTSTRAP[.]json$"
    regex = re.compile(regex_str)

    for file in home_results_dir.iterdir():

        match = regex.match(str(file.name))
        if match and int(match.group("n_trials")) == N_TRIALS:

            if match and match.group("acs_task") != "ACSIncome": continue
            # if match and match.group("acs_task") != "ACSPublicCoverage": continue

            experiments_to_use = load_json(file)
            for exp_dir in experiments_to_use:

                if not (
                    "lfr" in str(exp_dir).lower()
                    # and "lightgbm" in str(exp_dir).lower()
                ):
                    continue

                # Launch refit job
                launch_refit_model_job(Path(exp_dir))
