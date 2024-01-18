# Supplementary Materials for "Unprocessing Seven Years of Algorithmic Fairness"
> Published as a conference [paper at ICLR 2024](https://arxiv.org/abs/2306.07261).

**Table of contents:**

- Notebook(s) used to generate the paper's plots:
  - [`notebooks/MAIN-results-analysis.supp-materials.ipynb`](notebooks/MAIN-results-analysis.supp-materials.ipynb);
  - Other plotting notebooks available under the same folder [`notebooks`](notebooks/);
- Notebooks used to download and parse datasets used in the paper:
  - ACS / folktables datasets: [`notebooks/data.folktables-datasets-preprocessing-1hot.ipynb`](notebooks/data.folktables-datasets-preprocessing-1hot.ipynb);
  - MEPS dataset: [`notebooks/data.MEPS-datasets-preprocessing-1hot.ipynb`](notebooks/data.MEPS-datasets-preprocessing-1hot.ipynb);
- Hyperparameter space explored for each algorithm:
  - See folder for the respective dataset under [`hyperparameter_spaces`](/hyperparameter_spaces/);
- Re-running all paper experiments:
  - Each individual experiment (training and evaluating a single model) can be run independently using the [`scripts/run_main_experiment.py`](scripts/run_main_experiment.py) script;
  - All experiments can be launched using the [`scripts/launch_experiments_ACS.py`](scripts/launch_experiments_ACS.py) script for all ACS datasets, or using [`scripts/launch_experiments_MEPS.py`](scripts/launch_experiments_MEPS.py) for the MEPS dataset (_note_: experiments were ran using an `htcondor` cluster setup);
- All results gathered from the paper's experiments can be found under folder [`results`](results);
- Python packages required to re-run experiments or notebooks: see varied requirements files under folder [requirements](/requirements/);

**Note:** scripts and notebooks only tested on a Linux environment.
