import logging

import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset, BinaryLabelDataset


LABEL_COL_NAME = "label_"


def convert_df_to_aif360_compatible(
        X: pd.DataFrame,
        y: np.ndarray | pd.Series = None,
        sensitive: np.ndarray | pd.Series = None,
        protected_attribute_names: str | list[str] = None,
        label_col_name: str = LABEL_COL_NAME,
    ) -> BinaryLabelDataset:
    """Convert the given data to aif360-compatible format (w/ binary label)."""

    # All data must be float dtype
    df = X.astype(float)

    if isinstance(protected_attribute_names, str):
        protected_attribute_names = [protected_attribute_names]

    # Ensure protected attribute columns are part of the features
    assert all(col in df.columns for col in protected_attribute_names)
    # NOTE: `sensitive` will be ignored if provided

    # No other column's name should match the beginning of the protected attribute columns
    # (i.e., ensure no other columns are suspiciously named...)
    suspicious_cols = [
        col for col in df.columns
        if any(
            prot_col.startswith(col) and prot_col != col
            for prot_col in protected_attribute_names
        )
    ]
    if len(suspicious_cols) > 0:
        logging.warning(f"Removing suspicious columns {suspicious_cols} from the features.")
        df = df.drop(columns=suspicious_cols)

    # Add labels column even if one wasn't provided
    df = pd.concat(
        (
            df,
            pd.Series(
                data=y if y is not None else np.zeros(len(df)),
                index=df.index,
                name=label_col_name,
                dtype=float,
            ),
        ),
        axis=1,     # append on columns' axis
        ignore_index=False,
    )

    if sensitive is not None:
        logging.warning(
            f"convert_df_to_aif360_compatible: ignoring provided sensitive "
            f"attribute data (of type {type(sensitive)}), as the protected "
            f"attribute columns ({protected_attribute_names}) are part of the "
            f"dataset feature columns."
        )

    # Make `StandardDataset` object
    return StandardDataset(
        df=df,
        label_name=label_col_name,
        favorable_classes=[1],
        protected_attribute_names=protected_attribute_names,
        privileged_classes=[[0] for _col in protected_attribute_names],
    )

    # # Make `BinaryLabelDataset` object
    # return BinaryLabelDataset(
    #     df=df,
    #     label_names=[label_col_name],
    #     protected_attribute_names=protected_attribute_names,
    # )
