import inspect
from typing import Union
from collections import OrderedDict

import numpy as np
import pandas as pd
from fairlearn.reductions import Moment, ExponentiatedGradient

from hpt.utils import import_object


def parse_kwargs(kwargs: dict) -> dict:
    MODEL_ARGS_KEY = 'model__'
    CONSTRAINT_ARGS_KEY = 'constraint__'

    # Parse key-word arguments
    model_kwargs = {
        k[len(MODEL_ARGS_KEY):]: v for k, v in kwargs.items()
        if k.startswith(MODEL_ARGS_KEY)
    }

    # -> for the constraint
    constraint_kwargs = {
        k[len(CONSTRAINT_ARGS_KEY):]: v for k, v in kwargs.items()
        if k.startswith(CONSTRAINT_ARGS_KEY)
    }

    # -> finally, everything left is a kwarg to the meta_estimator
    kwargs = {
        k: v for k, v in kwargs.items() if not any([
            k.startswith(MODEL_ARGS_KEY),
            k.startswith(CONSTRAINT_ARGS_KEY),
        ])
    }

    # Build results dict
    results = OrderedDict()
    results['model_kwargs'] = model_kwargs
    results['constraint_kwargs'] = constraint_kwargs
    results['kwargs'] = kwargs
    return results


class FairlearnClassifier:
    """Module for wrapping a classifier under restrictions from the
    fairlearn package.
    """

    def __init__(
            self,
            fairlearn_reduction: Union[str, callable],
            estimator: Union[str, callable, object],
            constraint: Union[str, Moment],
            random_state: int = 42,
            protected_column: str = None,
            unawareness: bool = False,
            **kwargs,
        ):
        # If these were passed as classpath strings, load the correct class for each
        if isinstance(fairlearn_reduction, str):
            fairlearn_reduction = import_object(fairlearn_reduction)
        if isinstance(estimator, str):
            estimator = import_object(estimator)
        if isinstance(constraint, str):
            constraint = import_object(constraint)

        self.protected_column = protected_column
        self.random_state = random_state
        self.unawareness = unawareness

        # Parse key-word arguments
        self.model_kwargs, self.constraint_kwargs, self.kwargs = parse_kwargs(kwargs).values()

        # Build the base estimator
        # > if it was NOT already given as an object, create it
        if inspect.isclass(estimator):
            self.base_estimator = estimator(random_state=self.random_state, **self.model_kwargs)

        # > otherwise, use the provided object
        else:
            self.base_estimator = estimator

        self.constraint = constraint(**self.constraint_kwargs)

        self.fairlearn_reduction = fairlearn_reduction(
            estimator=self.base_estimator,
            constraints=self.constraint,
            **self.kwargs,
        )

        if isinstance(self.fairlearn_reduction, ExponentiatedGradient):
            self.predict_proba_method = lambda clf: clf._pmf_predict
        else:
            self.predict_proba_method = lambda clf: clf.predict_proba

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive_features: pd.Series, **kwargs):
        return self.fairlearn_reduction.fit(X, y, sensitive_features=sensitive_features, **kwargs)

    def predict(self, X: pd.DataFrame):
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]
        return self.fairlearn_reduction.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if self.unawareness and self.protected_column in X.columns:
            X = X[set(X.columns) - {self.protected_column}]

        return self.predict_proba_method(self.fairlearn_reduction)(X)
