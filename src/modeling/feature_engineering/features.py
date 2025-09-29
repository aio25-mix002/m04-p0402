"""Feature engineering utilities for the heart disease dataset.

Feature engineering helps the model capture relationships between
variables that may not be obvious from the raw data.  In this module
we implement a handful of simple derived features, inspired by the
analysis in the original `m0402.ipynb`.  Each function is pure and
returns a new DataFrame with additional columns.
"""

from __future__ import annotations

import pandas as pd
from typing import Iterable

from ..utils.logging_utils import logger


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features to the DataFrame.

    Currently implemented ratios:

    - ``chol_per_age``: serum cholesterol divided by age.
    - ``trestbps_per_age``: resting blood pressure divided by age.

    The function safely handles missing values and avoids division by
    zero by leaving the ratio as NaN where ``age`` is zero or NaN.

    Parameters
    ----------
    df: pandas.DataFrame
        Input feature matrix.  Must contain columns ``chol``, ``trestbps``
        and ``age``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with additional ratio columns appended.
    """
    df = df.copy()
    if {"chol", "age"}.issubset(df.columns):
        df["chol_per_age"] = df["chol"] / df["age"].replace({0: pd.NA})
    else:
        logger.warning("Cannot add chol_per_age feature because required columns are missing")
    if {"trestbps", "age"}.issubset(df.columns):
        df["trestbps_per_age"] = df["trestbps"] / df["age"].replace({0: pd.NA})
    else:
        logger.warning(
            "Cannot add trestbps_per_age feature because required columns are missing"
        )
    return df


def apply_feature_engineering(df: pd.DataFrame, add_ratio_features_enabled: bool = True) -> pd.DataFrame:
    """Apply a sequence of feature engineering steps to the DataFrame.

    This helper wraps multiple feature transformations into a single
    function.  Additional steps can be added over time.  Each step
    operates on a copy of the input to avoid modifying the original
    DataFrame in place.

    Parameters
    ----------
    df: pandas.DataFrame
        The input data.
    add_ratio_features_enabled: bool
        Whether to compute the ratio features defined in
        :func:`add_ratio_features`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered features appended.
    """
    engineered = df.copy()
    if add_ratio_features_enabled:
        engineered = add_ratio_features(engineered)
    return engineered
