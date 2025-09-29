# Project

- [Project](#project)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [Step 1: Create the virtual environment](#step-1-create-the-virtual-environment)
    - [Step 2: Activate the virtual environment](#step-2-activate-the-virtual-environment)
    - [Step 3: Install dependencies](#step-3-install-dependencies)
  - [Preparation](#preparation)
    - [Download dataset (optional)](#download-dataset-optional)
  - [Run application](#run-application)
    - [Train the model via CLI](#train-the-model-via-cli)
    - [Explore via Jupyter Notebook](#explore-via-jupyter-notebook)
  - [Appendix](#appendix)
    - [Using UV](#using-uv)
    - [Code Linting](#code-linting)
    - [CI/CD](#cicd)


## Prerequisites

- Package management: [uv](https://docs.astral.sh/uv/getting-started/installation/)
  or a recent version of Python (3.9+) with `pip` installed.

## Setup

### Step 1: Create the virtual environment

Use [uv](https://github.com/astral-sh/uv) to create an isolated environment:

```bash
uv venv
```

### Step 2: Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

Unix/Mac:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

With **uv** (recommended):

```bash
uv sync
```

Alternatively, using **pip**:

```bash
pip install -r requirements.txt
```

The project requires common data‑science libraries such as `pandas`,
`numpy`, `scikit‑learn` and `joblib`.  If you use `uv`, these will be
installed automatically according to the lock file.

## Preparation

### Download dataset (optional)

This repository expects the raw CSV file `heart.csv` from the
[`johnsmith88/heart-disease-dataset`](https://kaggle.com/johnsmith88/heart-disease-dataset)
on Kaggle.  You can download it manually and place it into
`data/raw/heart.csv` or run the provided script (requires a Kaggle API
token):

```bash
python ./scripts/download_data.py
```

The script uses `kagglehub` to fetch the file and saves it into
`data/raw/`.  See the KaggleHub documentation for API setup details.

## Run application

### Train the model via CLI

To run the end‑to‑end training pipeline, execute the training script from
the project root:

```bash
python ./scripts/train.py
```

This will load the dataset, perform preprocessing, optionally apply
feature engineering, split the data into training/validation/test sets,
tune multiple classifiers (RandomForest, LogisticRegression and
KNN), evaluate them on the validation set and then report metrics on
the held‑out test set.  The best model and preprocessing pipeline are
serialized into the `artifacts/` directory.

### Explore via Jupyter Notebook

An exploratory data analysis notebook is provided in
`notebooks/heart_disease_eda.ipynb`.  You can open it locally in a
Jupyter environment or run it on Google Colab.  To open in Colab,
replace `<user>/<repo>` below with the path to your repository:

* [Open in Google Colab](https://colab.research.google.com/github/<user>/<repo>/blob/main/heart_disease_classification/notebooks/heart_disease_eda.ipynb)

## Appendix

### Using UV

**Install UV**

See the [official UV documentation](https://docs.astral.sh/uv/getting-started/installation/)
for platform‑specific installation instructions.

**Restore packages**

```bash
uv sync
```

**Add a new package**

```bash
uv add package-name
```

### Code Linting

We recommend using [ruff](https://github.com/astral-sh/ruff) for
light‑weight code analysis.

**Check code only**

```bash
ruff check
```

**Check and auto‑fix**

```bash
ruff check --fix
```

### CI/CD

The project includes a GitHub Actions workflow that runs linting
checks on pull requests and pushes to the main branch.  You can
customise the workflow in `.github/workflows/` to suit your needs.
