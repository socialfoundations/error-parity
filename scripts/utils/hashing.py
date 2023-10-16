import json
import hashlib
import argparse
from pathlib import Path
from copy import deepcopy

from .temporal import get_current_timestamp


def hash_dict(d: dict, length: int = 8) -> str:
    d_enc = json.dumps(d, sort_keys=True).encode()
    return hashlib.shake_256(d_enc).hexdigest(length // 2)


def hash_exp_args(args: dict | argparse.Namespace) -> str:
    """Generate a hash that uniquely identifies the experiment's arguments.

    The hash should not depend on the `seed` or `save_dir`.

    Parameters
    ----------
    d : dict | argparse.Namespace
        The experiment's arguments (in dictionary form).

    Returns
    -------
    hash : str
        The experiment's hash.
    """
    args_for_hash = deepcopy(args)
    if isinstance(args, argparse.Namespace):
        args_for_hash = vars(args_for_hash)

    # Arguments unused for hash uniqueness
    args_for_hash.pop('seed', None)
    args_for_hash.pop('save_dir', None)

    for key in args_for_hash:
        if isinstance(args_for_hash[key], (Path, str)):
            p = Path(args_for_hash[key])
            if p.exists() and p.is_file() and p.suffix.lower() == 'yaml':
                args_for_hash[key] = p.resolve(strict=True)

    return hash_dict(args_for_hash)


# TODO: receive a dataclass here instead of each element separately?
def get_unique_experiment_name(
        dataset: str,
        base_model_yaml: str | Path,
        meta_model_yaml: str | Path = None,
        preprocessor_yaml: str | Path = None,
        one_hot: bool = False,
        seed: int = None,
        **_other_args,      # ignored
    ) -> str:

    # Generate a unique hash from the experiment's cmd arguments
    # NOTE: similar experiments with only different seeds will have the same hash
    args_hash = hash_exp_args({
        "acs_task": dataset,
        "base_model_yaml": base_model_yaml,
        "meta_model_yaml": meta_model_yaml,
        "preprocessor_yaml": preprocessor_yaml,
        "one_hot": one_hot,
    })
    current_timestamp = get_current_timestamp()

    return (
        f"{dataset}_"
        f"base={Path(base_model_yaml).stem}_"
        f"meta={Path(meta_model_yaml).stem if meta_model_yaml else 'None'}_"
        f"preprocessor={Path(preprocessor_yaml).stem if preprocessor_yaml else 'None'}_"
        f"one-hot={one_hot}_"
        f"seed={seed or 'unset'}_"
        f"hash={args_hash}_"
        f"{current_timestamp}"
    )
