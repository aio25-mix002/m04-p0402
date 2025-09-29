"""Preprocessing and data splitting utilities.

This module provides helpers for inferring feature types, building a
preprocessing pipeline and splitting the dataset into train, validation
and test partitions.  The preprocessing pipeline handles missing
values, scales numeric features and one‑hot encodes categorical
variables.
"""

from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from ..utils.logging_utils import logger


def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns from a DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
        The input data frame containing features and target.
    target_col: str
        Name of the target column to exclude from feature type inference.

    Returns
    -------
    numeric_cols: list[str]
        Names of numeric features.
    categorical_cols: list[str]
        Names of categorical (discrete) features.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        dtype = df[col].dtype
        # Assume integer features with a small number of distinct values
        # are categorical; floats are numeric; booleans are treated as
        # categorical for simplicity.  This heuristic can be adjusted.
        if pd.api.types.is_numeric_dtype(dtype):
            # heuristically consider integer columns with <= 10 unique values as categorical
            if pd.api.types.is_integer_dtype(dtype) and df[col].nunique() <= 10:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    logger.info(
        f"Inferred {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features"
    )
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Create a column transformer for preprocessing numeric and categorical data.

    Numeric features are imputed with the median and scaled to unit
    variance.  Categorical features are imputed with the most frequent
    category and one‑hot encoded.  Unknown categories at inference
    time are ignored.

    Parameters
    ----------
    numeric_cols: list[str]
        Names of numeric features.
    categorical_cols: list[str]
        Names of categorical features.

    Returns
    -------
    ColumnTransformer
        A fitted or un‑fitted scikit‑learn transformer for preprocessing.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # Use sparse=False for compatibility with scikit-learn <1.2 (sparse_output was introduced later)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train/validation/test partitions.

    The dataset is first split into a temporary training set and a test set
    according to ``test_size``.  The temporary training set is then split
    into a final training and validation set such that the fraction of
    the **total** data allocated to validation is ``val_size``.  All
    splits are stratified by the target variable to maintain class
    proportions.

    Parameters
    ----------
    X: pandas.DataFrame
        Feature matrix.
    y: pandas.Series
        Target vector.
    test_size: float
        Fraction of data to allocate to the test set.
    val_size: float
        Fraction of data to allocate to the validation set **out of the total** dataset.
    random_state: int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(
        f"Splitting data: test_size={test_size}, val_size={val_size}, random_state={random_state}"
    )
    # First split off the test set
    X_temp_train, X_test, y_temp_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    # Compute the validation fraction relative to the remaining data
    # e.g. if val_size=0.25 and test_size=0.2, val_fraction=0.25/(1-0.2)=0.3125
    val_fraction = val_size / max(1.0 - test_size, 1e-8)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp_train,
        y_temp_train,
        test_size=val_fraction,
        stratify=y_temp_train,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
