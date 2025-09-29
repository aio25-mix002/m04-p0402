"""Classifier orchestrator for heart disease prediction.

This module defines a ``HeartDiseaseClassifier`` class that trains one
or more machine‑learning models on preprocessed feature data.  It
supports hyper‑parameter tuning via scikit‑learn's ``GridSearchCV`` and
computes evaluation metrics on a validation set.  Supported
algorithms include random forests, logistic regression, k‑nearest
neighbours, decision trees, support vector machines (RBF kernel),
AdaBoost, gradient boosting, unsupervised K‑Means (with majority vote),
as well as third‑party boosters such as XGBoost and LightGBM (if
installed).  Additional models can be added by extending the
``_get_estimator`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from ..utils.logging_utils import logger


# -----------------------------------------------------------------------------
# Custom estimator definitions
# -----------------------------------------------------------------------------

class KMeansClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Simple wrapper to convert K‑Means clustering into a binary classifier.

    This classifier fits a :class:`~sklearn.cluster.KMeans` model to the
    training data and assigns a class label to each cluster based on the
    majority label of training points within that cluster.  Predictions for
    new samples are obtained by predicting the nearest cluster and then
    returning the corresponding majority class.  The ``predict_proba``
    method returns a one‑hot encoding of the predicted class to enable
    evaluation using metrics like ROC AUC.

    Parameters
    ----------
    n_clusters: int, optional
        The number of clusters to form.  Defaults to 2.  When used in
        conjunction with grid search, this value may be overridden by
        the hyperparameter grid.
    random_state: int or None, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_clusters: int = 2, random_state: int | None = None) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: KMeans | None = None
        self.mapping: Dict[int, int] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the KMeans model and derive cluster-to-class mapping.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data.
        y: ndarray of shape (n_samples,)
            Binary labels (0 or 1) corresponding to ``X``.

        Returns
        -------
        self: KMeansClassifierWrapper
            Fitted instance.
        """
        # Fit KMeans to the data
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        labels = self.kmeans.fit_predict(X)
        # Derive mapping from cluster id to majority class label
        mapping: Dict[int, int] = {}
        for c in np.unique(labels):
            cluster_indices = labels == c
            cluster_labels = y[cluster_indices]
            if len(cluster_labels) == 0:
                # Default to class 0 if no samples (should not happen)
                mapping[c] = 0
            else:
                # Majority class: round mean of cluster labels (ties -> class 1)
                mapping[c] = int(np.round(cluster_labels.mean()))
        self.mapping = mapping
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in ``X``.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            New samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted binary class labels.
        """
        if self.kmeans is None or self.mapping is None:
            raise ValueError("KMeansClassifierWrapper instance is not fitted yet.")
        clusters = self.kmeans.predict(X)
        return np.array([self.mapping.get(int(c), 0) for c in clusters])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in ``X``.

        This implementation returns a one‑hot encoding of the predicted
        class label, i.e. the probability for the predicted class is 1 and
        0 for the other class.  Although simplistic, this allows the
        downstream evaluation code to compute ROC AUC on the proxy scores.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            New samples.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Class probabilities for each sample.
        """
        preds = self.predict(X).astype(int)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute a decision function for each sample.

        Since K‑Means does not naturally provide a decision score, this method
        returns the predicted class label (0 or 1).  This allows code that
        falls back to ``decision_function`` when ``predict_proba`` is absent
        to work as intended.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            New samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Scores corresponding to the positive class (1).
        """
        return self.predict(X).astype(float)


@dataclass
class ModelResult:
    name: str
    estimator: Any
    params: Dict[str, Any]
    val_accuracy: float
    val_auc: float


class HeartDiseaseClassifier:
    """Train and evaluate multiple classification models.

    Parameters
    ----------
    random_state: int
        Random seed for reproducibility.
        algorithms: Iterable[str]
        Collection of algorithm names to train.  Supported values include
        ``"random_forest"``, ``"logistic_regression"``, ``"knn"``,
        ``"decision_tree"``, ``"svm"``/``"svm_rbf"``, ``"adaboost"``,
        ``"gradient_boosting"`` (and its variants ``_fe``, ``_dt``, ``_fe_dt``),
        ``"xgboost"`` and ``"lightgbm"`` (if the corresponding packages are
        installed), as well as ``"kmeans"`` for an unsupervised cluster‑based
        classifier.
    param_grid: dict[str, dict[str, Iterable]]
        Dictionary mapping algorithm names to dictionaries of hyperparameter
        grids for ``GridSearchCV``.
    cv_splits: int, optional
        Number of cross‑validation folds to use during tuning.  Defaults to 5.
    n_jobs: int, optional
        Number of parallel jobs to use.  ``-1`` uses all available cores.
    """

    def __init__(
        self,
        random_state: int,
        algorithms: Iterable[str],
        param_grid: Dict[str, Dict[str, Iterable]],
        cv_splits: int = 5,
        n_jobs: int = -1,
    ) -> None:
        self.random_state = random_state
        self.algorithms = list(algorithms)
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.n_jobs = n_jobs
        self.results: List[ModelResult] = []

    def _get_estimator(self, name: str):
        """
        Return an un‑fitted estimator corresponding to the specified algorithm name.

        Supported algorithm keys:

        - ``random_forest``: a :class:`RandomForestClassifier`.
        - ``logistic_regression``: a :class:`LogisticRegression` with the saga solver.
        - ``knn``: a :class:`KNeighborsClassifier`.
        - ``decision_tree`` or ``dt``: a :class:`DecisionTreeClassifier`.
        - ``svm`` or ``svm_rbf``: a :class:`SVC` with an RBF kernel and ``probability=True``.
        - ``adaboost``: an :class:`AdaBoostClassifier`.
        - ``xgboost``: an :class:`xgboost.XGBClassifier` (requires the ``xgboost`` package).
        - ``lightgbm``: an :class:`lightgbm.LGBMClassifier` (requires the ``lightgbm`` package).
        - ``kmeans``: a simple K‑Means classifier wrapper that converts cluster assignments to class labels.
        - ``gradient_boosting`` and its variants (``gradient_boosting_fe``, ``gradient_boosting_dt``,
          ``gradient_boosting_fe_dt``): a :class:`GradientBoostingClassifier`.

        Parameters
        ----------
        name: str
            The key identifying the algorithm to construct.

        Returns
        -------
        sklearn.base.BaseEstimator
            An unfitted estimator instance.
        """
        key = name.lower()
        if key == "random_forest":
            return RandomForestClassifier(random_state=self.random_state)
        if key == "logistic_regression":
            return LogisticRegression(
                max_iter=2000,
                solver="saga",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        if key == "knn":
            return KNeighborsClassifier()
        if key in {"decision_tree", "dt"}:
            return DecisionTreeClassifier(random_state=self.random_state)
        if key in {"svm", "svm_rbf"}:
            return SVC(kernel="rbf", probability=True, random_state=self.random_state)
        if key == "adaboost":
            return AdaBoostClassifier(random_state=self.random_state)
        if key == "kmeans":
            # Fallback cluster count will be overridden by parameter grid if provided
            return KMeansClassifierWrapper(n_clusters=2, random_state=self.random_state)
        if key.startswith("gradient_boosting"):
            return GradientBoostingClassifier(random_state=self.random_state)
        if key == "xgboost":
            try:
                from xgboost import XGBClassifier  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "XGBoost is not installed. Please install xgboost to use the xgboost algorithm."
                ) from exc
            return XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric="logloss",
                tree_method="hist",
                use_label_encoder=False,
            )
        if key == "lightgbm":
            try:
                from lightgbm import LGBMClassifier  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "LightGBM is not installed. Please install lightgbm to use the lightgbm algorithm."
                ) from exc
            return LGBMClassifier(random_state=self.random_state)
        raise ValueError(f"Unsupported algorithm: {name}")

    def _fit_model(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> GridSearchCV:
        """
        Construct and fit a ``GridSearchCV`` instance for the specified algorithm.

        Parameters
        ----------
        name: str
            Algorithm name (as configured in ``self.algorithms``).
        X_train: ndarray
            Training features.
        y_train: ndarray
            Training labels.

        Returns
        -------
        GridSearchCV
            A fitted ``GridSearchCV`` object containing the best estimator and parameters.
        """
        estimator = self._get_estimator(name)
        param_grid = self.param_grid.get(name, {}) or {}
        # If no hyperparameters specified, use a dummy grid to fit a single estimator
        if not param_grid:
            param_grid = {"dummy_param": [None]}
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        # Use accuracy scoring for unsupervised KMeans; otherwise use ROC AUC
        scoring = "accuracy" if name.lower() == "kmeans" else "roc_auc"
        gs = GridSearchCV(
            estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        logger.info(
            f"Starting grid search for {name} with {len(list(param_grid.keys()))} hyperparameters (scoring={scoring})"
        )
        gs.fit(X_train, y_train)
        logger.info(f"Finished grid search for {name}; best params: {gs.best_params_}")
        return gs

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train all configured models and evaluate them on a validation set.

        The results are stored in ``self.results`` sorted by decreasing
        validation AUC.
        """
        self.results = []
        for name in self.algorithms:
            try:
                gs = self._fit_model(name, X_train, y_train)
            except ImportError as exc:
                # Skip algorithms whose dependencies are not installed
                logger.warning(f"Skipping algorithm '{name}' due to missing dependency: {exc}")
                continue
            except Exception as exc:
                logger.warning(f"Failed to fit model for algorithm '{name}': {exc}")
                continue
            # Evaluate the best estimator on the validation set
            estimator = gs.best_estimator_
            # Some classifiers (e.g. KNN) may not expose predict_proba but do
            # expose decision_function; fallback accordingly
            if hasattr(estimator, "predict_proba"):
                probas = estimator.predict_proba(X_val)[:, 1]
            else:
                probas = estimator.decision_function(X_val)
            preds = estimator.predict(X_val)
            val_acc = accuracy_score(y_val, preds)
            try:
                val_auc = roc_auc_score(y_val, probas)
            except ValueError:
                val_auc = float("nan")
            result = ModelResult(
                name=name,
                estimator=estimator,
                params=gs.best_params_,
                val_accuracy=val_acc,
                val_auc=val_auc,
            )
            logger.info(
                f"Validation metrics for {name}: accuracy={val_acc:.4f}, AUC={val_auc:.4f}"
            )
            self.results.append(result)
        # Sort results by descending validation AUC (NaNs last)
        self.results.sort(key=lambda r: (-r.val_auc if not np.isnan(r.val_auc) else float("inf")))

    def best_model(self) -> ModelResult:
        """Return the model result with the highest validation AUC."""
        if not self.results:
            raise RuntimeError("No models have been trained yet")
        return self.results[0]

    def evaluate(self, model_result: ModelResult, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained estimator on the test set.

        Returns a dictionary containing accuracy and AUC.  If the test
        set contains only a single class, the AUC will be NaN.
        """
        estimator = model_result.estimator
        preds = estimator.predict(X_test)
        if hasattr(estimator, "predict_proba"):
            probas = estimator.predict_proba(X_test)[:, 1]
        else:
            probas = estimator.decision_function(X_test)
        test_acc = accuracy_score(y_test, preds)
        try:
            test_auc = roc_auc_score(y_test, probas)
        except ValueError:
            test_auc = float("nan")
        logger.info(
            f"Test metrics for {model_result.name}: accuracy={test_acc:.4f}, AUC={test_auc:.4f}"
        )
        return {"accuracy": test_acc, "auc": test_auc}
