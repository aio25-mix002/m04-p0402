"""End‑to‑end training script for the heart disease classifier.

This script orchestrates the entire modelling pipeline: it loads the
configuration, reads the raw data, performs optional feature
engineering, splits the data, preprocesses numeric and categorical
features, tunes multiple classification models, evaluates them on a
validation set and finally reports the performance on the test set.  The
best model and preprocessor are saved into the ``artifacts/``
directory for later use.
"""

from __future__ import annotations

import os
import joblib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure the ``src`` directory is on the Python path when running this
# script directly.  Without this, imports like ``from src.configuration``
# will fail.  The path manipulation below appends ``../src`` relative
# to this file to ``sys.path``.
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from configuration.configuration_manager import ConfigurationManager  # type: ignore  # noqa
from modeling.data import (
    load_heart_data,
    infer_feature_types,
    build_preprocessor,
    split_data,
)
from modeling.feature_engineering import apply_feature_engineering  # type: ignore  # noqa
from modeling.models import HeartDiseaseClassifier  # type: ignore  # noqa
from modeling.utils.logging_utils import logger  # type: ignore  # noqa

import matplotlib
matplotlib.use("Agg")  # Use a non‑interactive backend for image generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main() -> None:
    # Load configuration
    config = ConfigurationManager.load()
    logger.info("Loaded configuration")

    # Load raw data
    df = load_heart_data(config.data.raw_file)

    # Optional feature engineering
    if config.feature_engineering.add_ratio_features:
        logger.info("Applying feature engineering (ratio features)")
        df = apply_feature_engineering(df, add_ratio_features_enabled=True)

    # Separate features and target
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Infer feature types
    numeric_cols, categorical_cols = infer_feature_types(df, target_col=target_col)
    logger.info(f"Numeric columns: {numeric_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")

    # Split data into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.random_state,
    )
    logger.info(
        f"Data split completed: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # ---------------------------------------------------------------------
    # Data visualisation
    # ---------------------------------------------------------------------
    # Create an output directory for visualisations
    viz_dir = Path("artifacts") / "visualisations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    try:
    # Plot histograms for each numeric feature separated by the target class
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            tmp_df = pd.DataFrame({col: X_train[col], "target": y_train})

            # Add hue + palette (needed in seaborn ≥0.14)
            sns.histplot(
                data=tmp_df,
                x=col,
                hue="target",
                kde=True,
                stat="density",
                common_norm=False,
                palette="Set2",   # choose any palette you like
                ax=ax
            )

            ax.set_title(f"Distribution of {col} by target")
            fig.tight_layout()
            fig.savefig(viz_dir / f"hist_{col}.png")
            plt.close(fig)

        # Plot correlation heatmap for numeric features
        if numeric_cols:
            corr = X_train[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                vmin=-1, vmax=1,
                center=0,
                ax=ax
            )
            ax.set_title("Correlation matrix of numeric features (training set)")
            fig.tight_layout()
            fig.savefig(viz_dir / "correlation_matrix.png")
            plt.close(fig)

    except Exception as e:
        logger.warning(f"Failed to generate data visualisations: {e}")


    # Build and fit the preprocessor on training data only
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    logger.info("Fitting preprocessing pipeline")
    X_train_pre = preprocessor.fit_transform(X_train)
    X_val_pre = preprocessor.transform(X_val)
    X_test_pre = preprocessor.transform(X_test)

    # Extract hyperparameter grid from configuration (pydantic wraps them in ModelHyperparams)
    param_grid = {}
    for name, hyperparams in config.model.params.items():
        # hyperparams may be a ModelHyperparams instance or a plain dict
        params_dict = (
            hyperparams.__root__ if hasattr(hyperparams, "__root__") else hyperparams
        )
        # Convert YAML None to Python None (already done by PyYAML) and skip if value is None
        clean_dict = {k: v for k, v in params_dict.items()}
        param_grid[name] = clean_dict

    # Train models
    clf = HeartDiseaseClassifier(
        random_state=config.random_state,
        algorithms=config.model.algorithms,
        param_grid=param_grid,
    )
    clf.train(X_train_pre, y_train.values, X_val_pre, y_val.values)

    # Retrieve the best model and evaluate on test set
    best_result = clf.best_model()
    test_metrics = clf.evaluate(best_result, X_test_pre, y_test.values)
    logger.info(
        f"Best model: {best_result.name} (params={best_result.params})\n"
        f"Validation accuracy: {best_result.val_accuracy:.4f}, AUC: {best_result.val_auc:.4f}\n"
        f"Test accuracy: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}"
    )

    # ---------------------------------------------------------------------
    # Summarise results across all trained models
    # ---------------------------------------------------------------------
    results_summary = []
    for res in clf.results:
        try:
            metrics = clf.evaluate(res, X_test_pre, y_test.values)
        except Exception:
            metrics = {"accuracy": float("nan"), "auc": float("nan")}
        results_summary.append(
            {
                "model": res.name,
                "val_accuracy": res.val_accuracy,
                "val_auc": res.val_auc,
                "test_accuracy": metrics["accuracy"],
                "test_auc": metrics["auc"],
            }
        )
    results_df = pd.DataFrame(results_summary)
    # Sort by validation AUC (descending) but place NaNs at the end
    results_df.sort_values(
        by=["val_auc"], ascending=False, inplace=True, na_position="last"
    )

    # Save summary to CSV
    results_path = Path("artifacts") / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved model results summary to {results_path}")

    # Plot validation AUC bar chart
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

# Seaborn ≥0.14 requires hue when using palette; we set hue=model and disable legend.
        sns.barplot(
            x="model", y="val_auc",
            data=results_df,
            hue="model",            # <-- add hue
            dodge=False,
            legend=False,           # <-- hide redundant legend
            palette="viridis",      # <-- now valid with hue
            ax=ax
        )

        ax.set_title("Validation AUC by model")
        ax.set_ylabel("Validation AUC")
        ax.set_xlabel("Model")

        # Avoid set_xticklabels() warning; use setp to rotate/align existing tick labels.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        fig.tight_layout()
        fig.savefig(Path("artifacts") / "val_auc_bar.png")
        plt.close(fig)

    except Exception as e:
        logger.warning(f"Failed to plot validation AUC bar chart: {e}")

    # Save artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / f"best_model_{best_result.name}.pkl"
    preproc_path = artifacts_dir / "preprocessor.pkl"
    joblib.dump(best_result.estimator, model_path)
    joblib.dump(preprocessor, preproc_path)
    logger.info(f"Saved best model to {model_path}")
    logger.info(f"Saved preprocessor to {preproc_path}")


if __name__ == "__main__":
    main()
