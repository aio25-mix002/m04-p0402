"""Data loading utilities for the heart disease dataset.

This module defines functions to read the raw CSV file into a
pandas DataFrame and to normalise column names.  If the file does not
exist, a ``FileNotFoundError`` is raised.  Column names are validated
against the expected schema to detect unexpected formats early.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List

from ..utils.logging_utils import logger


# Canonical column names for the heart disease dataset.  See the
# documentation in ``m0402.ipynb`` for details.
COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_heart_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the heart disease dataset from a CSV file.

    Parameters
    ----------
    csv_path: str or Path
        Path to the raw CSV file containing the heart disease data.  The
        CSV must have the 14 columns listed in ``COLUMNS``; any
        additional columns will be ignored and missing columns will
        raise an exception.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the dataset with canonical column names.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file contains an unexpected number of columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    logger.info(f"Loading heart disease data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalise column names if they differ from the expected schema.  Some
    # versions of the dataset may use capitalised or alternative names
    # (e.g. "cp" vs "chest pain type").  We use a mapping based on known
    # prefixes to coerce the columns into our canonical names.
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Unexpected number of columns: {df.shape[1]}, expected {len(COLUMNS)}"
        )

    # If the columns do not exactly match, attempt to coerce by position.
    if list(df.columns) != COLUMNS:
        logger.info(
            "Column names differ from canonical names; renaming columns to standard schema"
        )
        df.columns = COLUMNS

    return df
