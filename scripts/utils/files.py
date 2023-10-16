import json
import pickle
import logging
from pathlib import Path
from typing import Union
from collections.abc import Collection

import cloudpickle

import numpy as np
import pandas as pd


def load_pickle(path: str | Path) -> object:
    with open(path, "rb") as f_in:
        return pickle.load(f_in)


def load_json(path: str | Path) -> object:
    with open(path, "r") as f_in:
        return json.load(f_in)


def save_pickle(obj: object, path: str | Path, overwrite: bool = True) -> bool:
    """Saves the given object as a pickle with the given file path.

    Parameters
    ----------
    obj : object
        The object to pickle
    path : str or Path
        The file path to save the pickle with.

    Returns
    -------
    success : bool
        Whether pickling was successful.
    """
    logging.info(f"Saving pickle file to '{str(path)}'")
    try:
        with open(path, "wb" if overwrite else "xb") as f_out:
            cloudpickle.dump(obj, f_out)
            return True

    except Exception as e:
        logging.error(f"Pickling failed with exception '{e}'")
        return False


def save_json(obj: Collection, path: str | Path, overwrite: bool = True):
    logging.info(f"Saving JSON file to '{str(path)}'")
    with open(path, "w" if overwrite else "x") as f_out:
        json.dump(obj, f_out, indent=4, sort_keys=True)


def save_object_information(obj: object, path: str | Path, **encoding_kwargs):
    """Encodes and saves object information to the given file path."""

    # 1. encode
    encoded_obj = encode_object_information(obj, **encoding_kwargs)

    # 2. save to disk
    save_json(encoded_obj, path=path)


def encode_object_information(obj: object, max_depth: int = 5) -> dict | str:
    """Encodes all kwargs (or hyperparameters) of the given object.

    The goal is to save all information required to re-train a model, but 
    without saving the trained model weights (which take up the most space).

    Parameters
    ----------
    obj : object
        Some python object.
    max_depth : int, optional
        The maximum recursion depth, by default 2.

    Returns
    -------
    dict | str
        A dictionary that recursively details all kwargs needed to instantiate
        a clean copy of this object.
    """

    # base case
    if max_depth <= 0 or not hasattr(obj, "__dict__"):
        return str(obj)

    # 2nd base case: don't save DFs or arrays
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        return str(type(obj))

    # recursive case
    else:
        return {
            key: encode_object_information(value, max_depth=max_depth-1)
            for key, value in vars(obj).items()
        }
