"""A wrapper for using AIF360 preprocessing algorithms.
"""

import logging

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from hpt.utils import import_object


class AIF360SklearnPreprocessing(BaseEstimator, ClassifierMixin, TransformerMixin):
    """An sklearn-api wrapper for AIF360 sklearn-api preprocessing objs.

    i.e., can wrap algorithms under aif360.sklearn.preprocessing.*
    """

    def __init__(
            self,
            preprocessing_class: str | type,
            sensitive_feature_ids: str | list[str],
            privileged_group: int = None,
            **kwargs,
        ):
        """Construct an AIF360 preprocessor object.

        This class serves as a wrapper for AIF360 preprocessing algorithms.

        Parameters
        ----------
        preprocessing_class : str | callable
            The class or classpath (i.e., type(class_)) for the underlying 
            aif360 preprocessing algorithm.
        sensitive_feature_ids : str | list[str]
            The name of the sensitive attribute column (or a list of columns).
        privileged_group : int, optional
            The sensitive attribute value for the privileged group, by default 
            None.
        """

        self.preprocessing_class = preprocessing_class
        self.sensitive_feature_ids = sensitive_feature_ids  # Using fairlearn naming convention for this kwarg
        assert "prot_attr" not in kwargs        # Provide this info in `sensitive_feature_ids` kwarg

        self.privileged_group = privileged_group
        self.kwargs = kwargs

        # Check if constructor/class is already loaded or in classpath format
        if isinstance(preprocessing_class, str):
            self.preprocessing_class = import_object(preprocessing_class)
        assert callable(self.preprocessing_class)

        # Construct preprocessing object
        self.preprocessing_obj = self.preprocessing_class(
            prot_attr=self.sensitive_feature_ids,
            **kwargs,
        )

    @staticmethod
    def data_to_aif360_preprocessor_compatible(
            data: pd.DataFrame,
            prot_attr_cols: str | list[str],
        ) -> pd.DataFrame:
        """Adapt input data as required by AIF360 preprocessing classes.

        1. sensitive attribute data should be on the index of `X`;
        2. all data should be floating point;

        Parameters
        ----------
        data : pd.DataFrame
            Input data to be made AIF360-compatible.
        prot_attr_cols : str | list[str]
            The name of the column (or list of column names) that contain 
            protected attributes.

        Returns
        -------
        pd.DataFrame
            The adapted data.
        """
        df = data.astype(float).set_index(prot_attr_cols)

        # No other column's name should match the beginning of the protected attribute columns
        # (i.e., ensure no other columns are suspiciously named...)
        suspicious_cols = [
            col for col in df.columns
            if any(
                prot_col.startswith(col) and prot_col != col
                for prot_col in prot_attr_cols
            )
        ]
        if len(suspicious_cols) > 0:
            logging.warning(f"Removing suspicious columns {suspicious_cols} from the features.")
            df = df.drop(columns=suspicious_cols)
        
        return df

    def fit(self, X, y, sample_weight=None):

        # If no privileged group was set, use majority group
        if self.privileged_group is None:
            sensitive_attr = pd.Series(X[self.sensitive_feature_ids])
            self.privileged_group = sensitive_attr.value_counts(ascending=False).index[0]

        logging.info(f"preprocessing: using {self.sensitive_feature_ids}={self.privileged_group} as privileged group.")

        # Fit underlying preprocessing object
        self.preprocessing_obj.fit(
            X=self.data_to_aif360_preprocessor_compatible(X, self.sensitive_feature_ids),
            y=y,
            priv_group=self.privileged_group,
            sample_weight=sample_weight,
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessing_obj.transform(
            self.data_to_aif360_preprocessor_compatible(X, self.sensitive_feature_ids),
        )

    def fit_transform(self, X, y, **fit_params) -> pd.DataFrame:
        return self.fit(X, y, **fit_params).transform(X)


class AIF360GeneralPreprocessing(BaseEstimator, ClassifierMixin, TransformerMixin):
    """An sklearn-compatible wrapper class for general aif360 preprocessing
    algorithms under aif360.algorithms.preprocessing.*
    """
    pass # TODO
