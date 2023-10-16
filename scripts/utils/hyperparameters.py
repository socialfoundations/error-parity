from pathlib import Path
from typing import Union

from hpt import suggest_random_hyperparams
from hpt.utils import load_hyperparameter_space, import_object

from .wrappers.fairlearn import FairlearnClassifier
from .files import save_json


def instantiate_given_model_config(
        base_kwargs: dict = None,
        meta_kwargs: dict = None,
        **kwargs,
    ) -> object:

    model_kwargs = base_kwargs or kwargs
    model_obj = instantiate_model(**model_kwargs)

    if meta_kwargs is not None:
        return instantiate_meta_model(
            base_model=model_obj,
            meta_kwargs=meta_kwargs,
        )

    else:
        return model_obj


def instantiate_random_model_config(
        hyperparameter_space_path: str | Path,
        seed: int,
        meta_hyperparameter_space_path: str | Path = None,
        save_file_path: str | Path = None,
        **other_kwargs,
    ) -> object:
    """Randomly samples a set of hyperparameters from the provided YAML spaces,
    and instantiates the corresponding model.

    It is compatible with constrained models that require instantiating a
    base unconstrained model; i.e., first it instantiates the 
    "standard/unconstrained/base" model, then it instantiates the 
    "super/constrained/meta" model using the first model.

    Parameters
    ----------
    hyperparameter_space_path : str | Path
        The path to a YAML file that details the hyperparameter space to sample 
        from. This is sufficient for standard unconstrained models (e.g., GBM) 
        or constrained models that don't require base models to be constructed 
        (e.g., FairGBM).
    seed : int
        This seed deterministically determines the sampled hyperparameter 
        values. That is, to generate different models you must provide different
        random seeds.
    meta_hyperparameter_space_path : str | Path, optional
        The path to a 2nd hyperparameter space that details the ensemble/meta
        parameters, by default None.
        That is, to construct some ensemble/constrained/meta models, it's
        required to instantiate the meta model and a prototype base model; in 
        these cases, the `hyperparameter_space_path` details the base model 
        parameters, and the `meta_hyperparameter_space_path` details the meta 
        model parameters.
    save_file_path : str | Path, optional
        If provided, will save the sampled parameters using this file path.
    other_kwargs : dict
        These arguments will be passed to the base model constructor as is,
        without random sampling.

    Returns
    -------
    object
        An instantiated ML model, ready to fit.
    """

    # Randomly sample kwargs from the (base) hyperparameter space
    kwargs = suggest_random_hyperparams(
        hyperparameter_space=_load_hyperparameter_space_with_cache(hyperparameter_space_path),
        seed=seed,
    )

    # Add the other (non-sampled) kwargs to the base model's constructor
    kwargs.update(other_kwargs)

    # Instantiate the corresponding ML model
    model = instantiate_model(**kwargs)

    # If a meta hyperparameter space is provided
    if meta_hyperparameter_space_path is not None:

        # Randomly sample kwargs from the meta hyperparameter space
        meta_kwargs = suggest_random_hyperparams(meta_hyperparameter_space_path, seed=seed)

        # (Optional) Save sampled kwargs to disk
        if save_file_path:
            save_json(
                obj={
                    "base_kwargs": kwargs,
                    "meta_kwargs": meta_kwargs,
                },
                path=save_file_path,
            )

        # Construct the meta model using the base model
        return instantiate_meta_model(
            base_model=model,
            meta_kwargs=meta_kwargs,
        )

    # Else, return the constructed model
    else:
        # (Optional) Save sampled kwargs to disk
        if save_file_path:
            save_json(
                obj=kwargs,
                path=save_file_path,
            )

        return model


def instantiate_meta_model(
        base_model: object,
        meta_kwargs: dict,
    ) -> object:
    """Instantiates a meta/super/ensemble model whose constructor requires a
    prototype for another model to serve as the base learner.

    Parameters
    ----------
    base_model : object
        An ML model to serve as the base learner (e.g., for an ensemble).
    meta_kwargs : dict
        A dictionary detailing the key-word arguments of the meta model.

    Returns
    -------
    meta_model : object
        The instantiated meta model.

    Raises
    ------
    NotImplementedError
        Thrown when this function is not (yet) compatible with the provided
        meta model classpath (`meta_kwargs['classpath']`).
    """

    # Unpack classpaths
    meta_classpath = meta_kwargs.pop("classpath")

    # Compatibility with fairlearn EG and GridSearch
    if "fairlearn" in meta_classpath.lower():
        meta_model = FairlearnClassifier(
            estimator=base_model,
            **meta_kwargs,
        )

    # NOTE: add compatibility with other meta classifiers here

    else:
        raise NotImplementedError(
            f"Not yet compatible with meta models of type '{meta_classpath}'")

    return meta_model


def instantiate_model(classpath: str, **hyperparams) -> object:
    constructor = import_object(classpath)
    assert callable(constructor), f"Invalid constructor '{classpath}'"

    return constructor(**hyperparams)


def _load_hyperparameter_space_with_cache(
        path_to_yaml: Union[str, Path],
        _cache: dict = {},       # cache to avoid re-loading the same file over and over
    ) -> dict:
    """Private function to load a given hyperparameter space from a yaml file.

    Parameters
    ----------
    path_to_yaml : str | Path
        The path to a YAML file detailing the hyperparameter space.
    _cache : dict, optional
        Cache shared between all function calls; you should NOT use this kwarg!

    Returns
    -------
    dict
        The loaded hyperparameter space (if loading was successful).
    """
    if not isinstance(path_to_yaml, (str, Path)):
        raise ValueError(
            f"Invalid argument type; got type(path_to_yaml)=={type(path_to_yaml)}")

    # Expand path to normalize relative paths
    path_to_yaml = Path(path_to_yaml).resolve()
    assert path_to_yaml.exists() and path_to_yaml.is_file()

    # Check if already in cache
    path_to_yaml = str(path_to_yaml)
    if path_to_yaml in _cache:
        return _cache[path_to_yaml]

    # If not, load it and store it in the cache
    _cache[path_to_yaml] = load_hyperparameter_space(path_to_yaml)
    return _cache[path_to_yaml]
