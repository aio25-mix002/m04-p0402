"""Download the heart disease dataset from Kaggle.

This script uses ``kagglehub`` to fetch the ``heart.csv`` file from the
`johnsmith88/heart-disease-dataset` dataset on Kaggle.  You must set
up a Kaggle API token in your environment (see
<https://github.com/tacajim/kagglehub#setup>) before running this
script.  The downloaded CSV will be saved into ``data/raw/heart.csv``.
"""

import sys
from pathlib import Path

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError as e:
    raise ImportError(
        "kagglehub is required for this script. Install it via `pip install kagglehub[pandas-datasets]`."
    ) from e


def download_dataset() -> None:
    dataset = "johnsmith88/heart-disease-dataset"
    file_path = "heart.csv"
    
    """
    For the Cleveland dataset, use:
    dataset = "ritwikb3/heart-disease-cleveland"
    file_path = "Heart_disease_cleveland_new.csv
    """
    
    print(f"Downloading {file_path} from {dataset}...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset,
        file_path,
    )
    dest = Path(__file__).resolve().parents[1] / "data" / "raw" / file_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"Saved dataset to {dest}")


if __name__ == "__main__":
    download_dataset()
