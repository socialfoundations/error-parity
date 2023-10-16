"""General constants and helper classes to run the main experiments on htcondor.
"""
import sys
import random
import dataclasses
from pathlib import Path
from copy import deepcopy

import htcondor
import classad

# Number of experiments to run per algorithm, per dataset
DEFAULT_N_TRIALS = 1

# Cluster settings
DEFAULT_JOB_BID = 25        # htcondor bid (min. is 15 apparently...)
DEFAULT_JOB_CPUS = 1         # number of CPUs per experiment (per cluster job)
DEFAULT_JOB_MEMORY_GB = 16        # GBs of memory


@dataclasses.dataclass
class ExperimentConfigs:
    base_model_yaml: str
    meta_model_yaml: str = None
    preprocessor_yaml: str = None
    one_hot: bool = False

    n_trials: int = DEFAULT_N_TRIALS
    job_cpus: int = DEFAULT_JOB_CPUS
    job_gpus: int = 0
    job_memory_gb: int = DEFAULT_JOB_MEMORY_GB
    job_bid: int = DEFAULT_JOB_BID

    @staticmethod
    def get_all_experiments(**configs):
        experiments = deepcopy(
            _unconstrained_experiments
            + _inprocessing_experiments
            + _preprocessing_experiments
        )

        return [
            dataclasses.replace(exp, **configs)
            for exp in experiments
        ]


####################################################
#  START of: details on which experiments to run.  #
####################################################

# (1) Standard (unconstrained) models
_unconstrained_experiments = [
    ExperimentConfigs(base_model_yaml="lightgbm.yaml"),
    ExperimentConfigs(base_model_yaml="random-forest.yaml"),
    ExperimentConfigs(base_model_yaml="logistic-regression.yaml", one_hot=True),
    ExperimentConfigs(base_model_yaml="skorch-NN.yaml", one_hot=True),
]

# (2) In-processing fairness methods
_inprocessing_meta_models_yaml_files = [
    "fairlearn-EG_equal-odds-constraint.yaml",
    "fairlearn-GS_equal-odds-constraint.yaml",
]

_inprocessing_experiments = [
    dataclasses.replace(
        exp,
        meta_model_yaml=meta_file,
        job_memory_gb=exp.job_memory_gb * 2,
    )
    for exp in _unconstrained_experiments
    for meta_file in _inprocessing_meta_models_yaml_files
    if not ("skorch" in exp.base_model_yaml and "fairlearn-GS" in meta_file)
    # NOTE: skorch is incompatible with fairlearn-GS for now...
]

# (2.1) Add non-meta-model in-processing methods
_inprocessing_experiments.extend([
    ExperimentConfigs(base_model_yaml="fairgbm_equal-odds-constraint.yaml")
])

# (3) Preprocessors (for each unconstrained model)
_preprocessor_yaml_files = [
    "fairlearn-correlation-remover.yaml",
    "aif360-LFR.yaml",
]

_preprocessing_experiments = [
    dataclasses.replace(
        exp,
        preprocessor_yaml=preproc_file,
        one_hot=True,
        job_memory_gb=exp.job_memory_gb * 2,
    )
    for exp in _unconstrained_experiments
    for preproc_file in _preprocessor_yaml_files
]

##################################################
#  END of: details on which experiments to run.  #
##################################################


def launch_experiments_jobs(
        dataset: str,
        exp_obj: ExperimentConfigs,
        data_dir: Path,
        results_dir: Path,
        configs_dir: Path,
        verbose: bool = True,
    ):
    """Launches the cluster jobs to execute all `n_trials` of a given experiment.

    Parameters
    ----------
    dataset : str
        The name of the dataset to use (e.g., ACS task or MEPS).
    exp_obj : ExperimentConfigs
        The detailed configs to run an experiment.
    data_dir : Path
        Path to data directory.
    results_dir : Path
        Path to results directory.
    configs_dir : Path
        Path to configurations directory (contains model YAML files).
    verbose : bool, optional
        Whether to output verbose logs, by default True.
    """

    # meta model cmd-line arg (if one is provided)
    meta_model_arg_line = ""
    if exp_obj.meta_model_yaml:
        meta_model_arg_line = f"--meta-model-yaml {str(configs_dir / exp_obj.meta_model_yaml)} "

    # preprocessor model cmd-line arg (if one is provided)
    preprocessor_arg_line = ""
    if exp_obj.preprocessor_yaml:
        preprocessor_arg_line = f"--preprocessor-yaml {str(configs_dir / exp_obj.preprocessor_yaml)} "

    # Get a unique name for this experiment/job
    from utils.hashing import get_unique_experiment_name
    unique_exp_name = get_unique_experiment_name(
        dataset=dataset,
        base_model_yaml=exp_obj.base_model_yaml,
        meta_model_yaml=exp_obj.meta_model_yaml,
        preprocessor_yaml=exp_obj.preprocessor_yaml,
        one_hot=exp_obj.one_hot,
    )

    # Name/prefix for cluster logs related to this job
    cluster_logs_save_dir = results_dir / "cluster-logs"
    cluster_logs_save_dir.mkdir(exist_ok=True)
    cluster_job_log_name = str(
        cluster_logs_save_dir / f"{unique_exp_name}_$(Cluster).$(Process)"
    )

    # Construct job description
    job_description = htcondor.Submit({
        "executable": f"{sys.executable}",  # correct env for the python executable
        # "arguments": "foo.py",    # NOTE: used for testing
        "arguments": (
            "run_main_experiment.py "
            f"--dataset {dataset} "
            f"--base-model-yaml {str(configs_dir / exp_obj.base_model_yaml)} "
            f"{meta_model_arg_line} "
            f"{preprocessor_arg_line} "
            f"--seed $(job_seed) "
            f"--data-dir {str(data_dir)} "
            f"--save-dir {str(results_dir)} "
            f"{'--verbose' if verbose else ''} "
            f"{'--one-hot' if exp_obj.one_hot else ''} "
        ),
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{exp_obj.job_cpus}",
        "request_gpus": f"{exp_obj.job_gpus}",
        "request_memory": f"{exp_obj.job_memory_gb}GB",
        "request_disk": "2GB",
        "jobprio": f"{exp_obj.job_bid - 1000}",
        "notify_user": "andre.cruz@tuebingen.mpg.de",
        "notification": "error",
        "job_seed_macro": f"$(Process) + {random.randrange(int(1e9))}",      # add random salt to all job seeds
        "job_seed": "$INT(job_seed_macro)",

        # Concurrency limits:
        # > each job uses this amount of resources out of a pool of 10k
        "concurrency_limits": f"user.t56_experiment_{dataset}:200",     # 200 jobs in parallel

        "+MaxRunningPrice": 100,
        "+RunningPriceExceededAction": classad.quote("restart"),
    })

    # Submit `n_trials` jobs to the scheduler
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description, count=exp_obj.n_trials)

    if verbose:
        print(
            f"Launched {submit_result.num_procs()} processes with "
            f"cluster-ID={submit_result.cluster()}\n")
