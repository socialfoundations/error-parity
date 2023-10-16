"""A wrapper helper class for AIF360 inprocessing algorithms.
"""
import logging

import pandas as pd
import numpy as np

from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

from .datasets import convert_df_to_aif360_compatible
from ...temporal import get_current_timestamp


class AdversarialDebiasingWrapper(AdversarialDebiasing):
    """A sklearn-compatible wrapper for aif360 AdversarialDebiasing.

    NOTE
    This was made redundant by the official sklearn-compatible class:
    `aif360.sklearn.inprocessing.AdversarialDebiasing`.
    """

    LABEL_COL_NAME = "label_"

    def __init__(
            self,
            protected_attribute_names: list[str] = [],
            **kwargs,
        ):

        # Names of columns containing protected attributes
        self._protected_attribute_names = protected_attribute_names

        # Ensure eager execution is disabled (incompatible with aif360)
        tf.disable_eager_execution()
        tf.reset_default_graph()

        # Dictionary of default kwargs
        default_kwargs = dict(
            scope_name=(
                f"AdversarialDebiasing_"
                f"debias={kwargs.get('debias', True)}_"
                f"{get_current_timestamp()}"
            ),
            sess=tf.Session(),
            debias=True,
            privileged_groups=[{col: 0.} for col in protected_attribute_names],
            unprivileged_groups=[{col: 1.} for col in protected_attribute_names],
        )

        super().__init__(**(default_kwargs | kwargs))
        # super().__init__(unprivileged_groups, privileged_groups, scope_name, sess, seed, adversary_loss_weight, num_epochs, batch_size, classifier_num_hidden_units, debias)

    def fit(self, X, y, sensitive=None):
        return super().fit(
            convert_df_to_aif360_compatible(
                X, y, sensitive=sensitive,
                protected_attribute_names=self._protected_attribute_names,
                label_col_name=self.LABEL_COL_NAME,
            )
        )

    def predict(self, X, sensitive=None) -> np.ndarray:
        return super().predict(
            convert_df_to_aif360_compatible(
                X, sensitive=sensitive,
                protected_attribute_names=self._protected_attribute_names,
                label_col_name=self.LABEL_COL_NAME,
            )
        ).labels.ravel()

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        return self.predict(*args, **kwargs)
