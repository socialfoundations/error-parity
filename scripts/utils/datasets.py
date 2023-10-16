"""Helper functions to load and pre-process ACS task data.

See the `data.folktables-datasets-preprocessing.ipynb` notebook for how to
pre-process ACS task data and construct csv files that are compatible with this
module.

"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import folktables


# Set of categorical column names for ACS data
ACS_CATEGORICAL_COLS = {
    'COW',  # class of worker
    'MAR',  # marital status
    'OCCP', # occupation code
    'POBP', # place of birth code
    'RELP', # relationship status
    'SEX',
    'RAC1P', # race code
    'DIS',  # disability
    'ESP',  # employment status of parents
    'CIT',  # citizenship status
    'MIG',  # mobility status
    'MIL',  # military service
    'ANC',  # ancestry
    'NATIVITY',
    'DEAR',
    'DEYE',
    'DREM',
    'ESR',
    'ST',
    'FER',
    'GCL',
    'JWTR',
#     'PUMA',
#     'POWPUMA',
}

# Name of all possible ACS tasks
ALL_ACS_TASKS = {
    'ACSIncome',
    'ACSPublicCoverage',
    'ACSMobility',
    'ACSEmployment',
    'ACSTravelTime',
}

# Set of categorical column names for MEPS data
MEPS_CATEGORICAL_COLS = {
    'REGION','SEX','MARRY',
    'FTSTU','ACTDTY','HONRDC',
    'RTHLTH','MNHLTH','HIBPDX',
    'CHDDX','ANGIDX','MIDX',
    'OHRTDX','STRKDX','EMPHDX',
    'CHBRON','CHOLDX','CANCERDX',
    'DIABDX','JTPAIN','ARTHDX',
    'ARTHTYPE','ASTHDX','ADHDADDX',
    'PREGNT','WLKLIM','ACTLIM',
    'SOCLIM','COGLIM','DFHEAR42',
    'DFSEE42','ADSMOK42','PHQ242',
    'EMPST','POVCAT','INSCOV',
}


def split_X_Y_S(data, label_col: str, sensitive_col: str, ignore_cols=None, unawareness=False) -> tuple:
    ignore_cols = ignore_cols or []
    ignore_cols.append(label_col)
    if unawareness:
        ignore_cols.append(sensitive_col)
    
    feature_cols = [c for c in data.columns if c not in ignore_cols]
    
    return (
        data[feature_cols],                           # X
        data[label_col].to_numpy().astype(int),       # Y
        data[sensitive_col].to_numpy().astype(int),   # S
    )


def get_acs_task_obj(task_name: str) -> object:
    """Returns the object that contains information for the given ACS task.
    """
    return getattr(folktables, task_name)


def get_default_sensitive_col(dataset: str) -> str:
    """Gets the default sensitive column name for the given dataset.
    """
    if dataset.lower() == "meps":
        return "RACE"
    else:
        return get_acs_task_obj(dataset).group


def get_default_label_col(dataset: str) -> str:
    """Gets the default target/label column name for the given dataset.
    """
    if dataset.lower() == "meps":
        return "UTILIZATION"
    else:
        return get_acs_task_obj(dataset).target
    

def get_default_categorical_columns(dataset: str) -> set[str]:
    """Gets the name of the categorical columns of the given dataset.

    Note that these correspond to none one-hot encoded data, as with one-hot
    encoding using dtype="category" or dtype=int is all the same.
    """
    if dataset.lower() == "meps":
        return MEPS_CATEGORICAL_COLS
    else:
        return ACS_CATEGORICAL_COLS


def load_data(
        dir_path: str,
        dataset_name: str,
        sensitive_col: str = None,
        one_hot: bool = False,
    ) -> dict[str, pd.DataFrame]:
    """Loads the given data from pre-generated datasets.

    Currently compatible with:
    - ACS datasets (all tasks);
    - MEPS dataset;

    Parameters
    ----------
    dir_path : str
        Path to directory that contains pre-processed datasets.
    dataset_name : str
        Name of the dataset to load (e.g., the name of the ACS task, or MEPS).
    sensitive_col : str, optional
        Sensitive column name. If not provided (None), will assume the default
        sensitive column for the respective dataset (recommended).
    one_hot : bool, optional
        Whether to (try to) load one-hot encoded data or regular label encoded
        data, by default False (will not load one-hot data).

    Returns
    -------
    dict[str, tuple]
        A dict with triplets composed of (features, label, sensitive_attribute).
        The dict keys will be the data type, and values will be the triplets.
        e.g., ```{"train": (X, Y, S)}```
    """
    # Log which sensitive attribute column we're using
    sensitive_col = sensitive_col or get_default_sensitive_col(dataset_name)
    logging.info(
        f"Loading {dataset_name} data using the following sensitive attribute "
        f"column: '{sensitive_col}'.")

    # Load train, test, and validation data
    data = dict()
    for data_type in ['train', 'test', 'validation']:

        # Construct file path
        path = Path(dir_path) / (
            f"{dataset_name}.{data_type}"
            f"{'.1-hot' if one_hot else ''}"
            f".csv"
        )

        if not path.exists():
            logging.warning(
                f"Couldn't find {data_type} data for '{dataset_name}' "
                f"with one-hot={one_hot}. Path {str(path)} doesn't exist."
            )
            continue

        # Read data from disk
        df = pd.read_csv(path, index_col=0)

        # Set categorical columns
        cat_cols = get_default_categorical_columns(dataset_name) & set(df.columns)
        df = df.astype({col: "category" for col in cat_cols})

        data[data_type] = split_X_Y_S(
            df,
            label_col=get_default_label_col(dataset_name),
            sensitive_col=sensitive_col,
        )

    return data
