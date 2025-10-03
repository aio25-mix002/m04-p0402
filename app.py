from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio features used during training.

    Adds ``chol_per_age`` and ``trestbps_per_age`` columns by dividing
    cholesterol and resting blood pressure by age.  Handles division by zero
    gracefully.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns ``chol``, ``trestbps`` and ``age``.

    Returns
    -------
    pandas.DataFrame
        Copy of input with ratio features appended.
    """
    df = df.copy()
    if {"chol", "age"}.issubset(df.columns):
        df["chol_per_age"] = df["chol"] / df["age"].replace({0: np.nan})
    if {"trestbps", "age"}.issubset(df.columns):
        df["trestbps_per_age"] = df["trestbps"] / df["age"].replace({0: np.nan})
    return df

def load_dataset() -> pd.DataFrame:
    """Load the heart disease dataset with absolute path resolution.

    Returns
    -------
    pandas.DataFrame
        The heart disease dataset including engineered ratio features if
        configured to do so.
    """
    # Determine project root and dataset path relative to this file
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir
    config_path = project_dir / "configs" / "config.yaml"
    # Insert src into sys.path to import project modules if available
    src_path = project_dir / "src"
    sys.path.insert(0, str(src_path))
    try:
        from configuration.configuration_manager import ConfigurationManager  # type: ignore
        from modeling.data import load_heart_data  # type: ignore
        from modeling.feature_engineering import apply_feature_engineering  # type: ignore
    except Exception:
        # If imports fail, fall back to reading CSV directly
        config = None
    else:
        # Load configuration
        config = ConfigurationManager.load(config_path)
        
    # If config loaded, use raw_file from config; else default path
    if config is not None:
        raw_rel = config.data.raw_file
        raw_path = project_dir / raw_rel
    else:
        raw_path = project_dir / "data" / "raw" / "heart_statlog_cleveland_hungary_final_cleaned.csv"
    # Read CSV
    df = pd.read_csv(raw_path)
    # Apply ratio features if config specified; fallback to always compute
    if config is not None and config.feature_engineering.add_ratio_features:
        df = apply_feature_engineering(df, add_ratio_features_enabled=True)  # type: ignore
    else:
        df = compute_ratio_features(df)
    return df

def set_split(df: pd.DataFrame, val=False):
    """Split train, validation, and test sets."""
    # Separate features and target
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    # Split into train/validation/test (80/10/10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    
    if val:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1111, stratify=y_train, random_state=42
        )  # 0.1111 * 0.9 ≈ 0.1
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    return (X_train, y_train), (X_test, y_test) 
    
    

def preprocessor(train_set: tuple, test_set: tuple, val_set: tuple=None) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features.

    Uses median imputation and standard scaling for numeric features, and
    most‑frequent imputation plus one‑hot encoding (dense) for categorical
    features.  Compatible with scikit‑learn 1.1.x.
    """
  
    X_train, y_train = train_set
    X_test, y_test = test_set
    if val_set:
        X_val, y_val = val_set
   # Identify numeric and categorical columns (integer features with <=10 unique values as categorical)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            if pd.api.types.is_integer_dtype(X_train[col]) and X_train[col].nunique() <= 10:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col) 
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # Đừng truyền sparse=False hay sparse_output=False
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
     
    raw_pipeline = Pipeline([
    ('preprocess', preprocess),
    ])
    # print('145 X_train ftr col ', X_train.columns)
    X_train_processed = raw_pipeline.fit_transform(X_train, y_train)
    X_test_processed = raw_pipeline.transform(X_test)
    
    if val_set:
        X_val_processed = raw_pipeline.transform(X_val)
        return (X_train_processed, y_train), (X_val_processed, y_val), (X_test_processed, y_test)
    
    return (X_train_processed, y_train), (X_test_processed, y_test)


def train_selected_model(algo: str, train_set: tuple, test_set: tuple, val_set: tuple = None) -> tuple[Pipeline, dict[str, float]]:
    """Train a machine‑learning model on the heart disease dataset.

    Parameters
    ----------
    algo : str
        Name of the algorithm to train.  Supported values: 'logistic_regression',
        'random_forest', 'knn', 'decision_tree', 'svm_rbf', 'adaboost',
        'gradient_boosting'.  These names correspond to keys in the
        configuration.
    df : pandas.DataFrame
        The full dataset including features and target (with ratio features).

    Returns
    -------
    tuple
        (best_pipeline, metrics) where ``best_pipeline`` is a fitted
        scikit‑learn pipeline (preprocessor + estimator) and ``metrics`` is a
        dictionary of test accuracy and AUC.
    """
    
    # Build preprocessing
    X_train, y_train = train_set
    # X_val, y_val = val_set
    X_test, y_test = test_set
    
    # Define algorithm and parameter grid
    algo = algo.lower()
    estimator: any
    param_grid: dict[str, list] = {}
    if algo == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, solver="saga")
        param_grid = {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        }
    elif algo == "random_forest":
        estimator = RandomForestClassifier(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
    elif algo == "knn":
        estimator = KNeighborsClassifier()
        param_grid = {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    elif algo in {"decision_tree", "dt"}:
        estimator = DecisionTreeClassifier(random_state=42)
        param_grid = {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__class_weight": [None, "balanced"],
        }
    elif algo in {"svm", "svm_rbf"}:
        estimator = SVC(kernel="rbf", probability=True, random_state=42)
        param_grid = {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": [0.01, 0.1, 1.0],
            "model__class_weight": [None, "balanced"],
        }
    elif algo == "adaboost":
        estimator = AdaBoostClassifier(random_state=42)
        param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0],
        }
    elif algo == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "model__n_estimators": [200, 500],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5],
        }
    elif algo in {"lightgbm", "lgbm", "lgbm_classifier"}:
        # pip install lightgbm
        estimator = LGBMClassifier(
            random_state=42,
            objective="binary",
            n_jobs=-1)
        param_grid = {
            "model__n_estimators": [200, 500, 800],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__num_leaves": [31, 63, 127],
            "model__max_depth": [-1, 5, 10],
            "model__min_child_samples": [10, 20, 40],
            "model__subsample": [0.7, 0.9, 1.0],          # bagging_fraction
            "model__colsample_bytree": [0.7, 0.9, 1.0],   # feature_fraction
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [0.0, 0.1, 1.0],
        }

    elif algo in {"xgboost", "xgb", "xgb_classifier"}:
        # pip install xgboost
        estimator = XGBClassifier(
            random_state=42,
            objective="binary:logistic",
            n_estimators=200,
            tree_method="hist",        # nhanh & ổn định
            eval_metric="auc",         # tránh warning
            n_jobs=-1,
            use_label_encoder=False
        )
        param_grid = {
            "model__n_estimators": [200, 500, 800],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 8],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__gamma": [0.0, 0.1, 1.0],
            # Nếu dữ liệu mất cân bằng, có thể thêm:
            # "model__scale_pos_weight": [1, 2, 5, 10],
        }
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    # Build full pipeline
    pipeline = Pipeline([
        # ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    # Grid search
    gs = GridSearchCV(
        estimator=pipeline,  
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    best_pipeline: Pipeline = gs.best_estimator_
    
    # Evaluate on validation set (not used but can be shown)
    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    if hasattr(best_pipeline["model"], "predict_proba"):
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = best_pipeline.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    metrics = {"accuracy": acc, "auc": auc}
    return best_pipeline, metrics


################################################################################
# Streamlit UI
################################################################################

def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Dashboard",
        page_icon="❤️",
        layout="wide",
    )

    st.title("❤️ Heart Disease Prediction & Exploration")
    st.write(
        "This dashboard allows you to explore the heart disease dataset, "
        "train different machine learning models and make predictions based on patient parameters."
    )

    # Tabs for different functionality
    tabs = st.tabs(["Overview", "Train Model", "Predict", "Visualisations"])

    # Load dataset once for overview and training
    @st.cache_data
    def get_dataset():
        return load_dataset()

    df = get_dataset()
    train_set, test_set = set_split(df, val=False)
    train_set_processed, test_set_processed = preprocessor(train_set, test_set, val_set=None)
    TARGET = 'target'
    ftr_list = list(df.columns)
    ftr_list.remove(TARGET)
    numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
    categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    
    ########################################
    # Overview Tab
    ########################################
    with tabs[0]:
        st.header("Data Overview")
        st.write(f"Number of samples: {len(df)}")
        st.write(f"Number of features (including engineered): {df.shape[1] - 1}")
        st.write("First five rows of the dataset:")
        st.dataframe(df.head())
        st.subheader("Target variable distribution")
        target_counts = df["target"].value_counts().rename({0: "No Disease", 1: "Disease"})
        st.bar_chart(target_counts)
        
        # EDA
        st.subheader("Exploratory Data Analysis on Train set")     
        
        # Gộp X_train và y_train để tiện vẽ
        X_train, y_train = train_set
        # st.markdown(f'X_train shape {X_train.shape}')
        X_train = pd.DataFrame(data=X_train, columns=ftr_list)  # Create pd.DataFrame from numpy array
        eda_df = X_train.copy()
        eda_df[TARGET] = y_train

        st.markdown('1. Distribution exploration on numeric features')
        # Các cột numeric muốn kiểm tra
        cols_to_plot = numeric_cols  # ['age','trestbps','chol','thalach','oldpeak']

        for col in cols_to_plot:
            plt.figure(figsize=(8,3))

            # Histplot
            plt.subplot(1,3,1)
            sns.histplot(data=eda_df, x=col, hue=TARGET, kde=True, element="step")
            plt.title(f"Histogram of {col} by {TARGET}")

            # Boxplot
            plt.subplot(1,3,2)
            sns.boxplot(data=eda_df, x=TARGET, y=col)
            plt.title(f"Boxplot of {col} by {TARGET}")

            # Violinplot
            plt.subplot(1,3,3)
            sns.violinplot(data=eda_df, x=TARGET, y=col)
            plt.title(f"Violinplot of {col} by {TARGET}")

            plt.tight_layout()
            st.pyplot(plt.gcf())  #plt.show()
            plt.close()  # Close the figure to free memory
            
        # 2. Histogram to identify skewness
        st.markdown('2. Transformations on numeric features')
        # Các cột muốn kiểm tra (lấy từ numeric_cols để đồng bộ biến đã có)
        eda_cols = [c for c in ['trestbps','chol','oldpeak','thalach'] if c in numeric_cols]

        # Lấy dữ liệu train cho các cột này
        eda_df = X_train[eda_cols].copy()

        # Log-transform (log1p an toàn với 0)
        for col in eda_cols:
            eda_df[f"{col}_log"] = np.log1p(eda_df[col])

        # Yeo-Johnson transform
        pt = PowerTransformer(method='yeo-johnson')
        eda_df[[f"{col}_yj" for col in eda_cols]] = pt.fit_transform(eda_df[eda_cols])

        # Vẽ histplot so sánh trước/sau biến đổi
        for col in eda_cols:
            plt.figure(figsize=(9,3))

            plt.subplot(1,3,1)
            sns.histplot(eda_df[col], kde=True)
            plt.title(f"Original {col} (skew={X_train[col].skew():.2f})")

            plt.subplot(1,3,2)
            sns.histplot(eda_df[f"{col}_log"], kde=True)
            plt.title(f"Log-transform {col}")

            plt.subplot(1,3,3)
            sns.histplot(eda_df[f"{col}_yj"], kde=True)
            plt.title(f"Yeo-Johnson {col}")

            plt.tight_layout()
            st.pyplot(plt.gcf()) 
            plt.close()

        # 3. Feature correlation matrix
        st.markdown('3. Correlation matrix among numeric features')
        plt.figure(figsize=(6, 5))
        # st.markdown(eda_df)
        corr_matrix = X_train[numeric_cols].corr(method='pearson')
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    vmin=-1, vmax=1, center=0, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title("Correlation matrix among numeric features")
        plt.tight_layout()
        # plt.savefig("correlation_matrix.pdf", bbox_inches="tight")
        st.pyplot(plt.gcf())  #plt.show()
        plt.close()
            
    ########################################
    # Train Model Tab
    ########################################
    with tabs[1]:
        st.header("Train a Model")
        st.write(
            "Select a machine learning algorithm from the dropdown and click **Train** to fit it on the dataset. "
            "A small hyper‑parameter search will be performed automatically."
        )
        algo_map = {
            "Logistic Regression": "logistic_regression",
            "Random Forest": "random_forest",
            "K‑Nearest Neighbours": "knn",
            "Decision Tree": "decision_tree",
            "Support Vector Machine (RBF)": "svm_rbf",
            "AdaBoost": "adaboost",
            "Gradient Boosting": "gradient_boosting",
        }
        algo_label = st.selectbox("Choose algorithm", list(algo_map.keys()))
        if st.button("Train", key="train_button"):
            algo_key = algo_map[algo_label]
            with st.spinner(f"Training {algo_label}..."):
                try:
                    pipeline, metrics = train_selected_model(algo_key, train_set_processed, test_set_processed)
                except ValueError as e:
                    st.error(str(e))
                    pipeline, metrics = None, None
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    pipeline, metrics = None, None
            if pipeline is not None:
                # Save to session state
                st.session_state["trained_pipeline"] = pipeline
                st.session_state["trained_algo"] = algo_label
                # Display metrics
                st.success(
                    f"Training complete! Test Accuracy: {metrics['accuracy']:.4f}, "
                    f"Test AUC: {metrics['auc']:.4f}"
                )
            else:
                st.info("No model was trained. Please select a supported algorithm.")

        # Optionally show which model is currently loaded
        if "trained_pipeline" in st.session_state:
            st.write(
                f"Current model in memory: {st.session_state['trained_algo']}"
            )
        else:
            st.write("No model has been trained yet.")

    ########################################
    # Prediction Tab
    ########################################
    with tabs[2]:
        st.header("Make a Prediction")
        if "trained_pipeline" not in st.session_state:
            st.info("Please train a model in the *Train Model* tab before making predictions.")
        else:
            st.write(
                "Enter patient parameters below.  All inputs should reflect values at rest or during exercise."
            )
            # Collect user input (same as before)
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 18, 100, 50)
                trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
                chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
            with col2:
                thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
                oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
                ca = st.slider("Major Vessels colored by Fluoroscopy", 0, 3, 0)
            with col3:
                sex_label = st.selectbox("Gender", ["Male", "Female"], index=0)
                cp_label = st.selectbox(
                    "Chest Pain Type",
                    ["Typical Angina", "Atypical Angina", "Non‑anginal Pain", "Asymptomatic"],
                    index=0,
                )
                fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], index=1)
                restecg_label = st.selectbox(
                    "Resting ECG Results",
                    ["Normal", "ST‑T Wave Abnormality", "Left Ventricular Hypertrophy"],
                    index=0,
                )
                exang_label = st.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0)
                slope_label = st.selectbox(
                    "Slope of the ST Segment", ["Upsloping", "Flat", "Downsloping"], index=0
                )
                thal_label = st.selectbox(
                    "Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], index=0
                )
            # Mapping to numeric
            sex_map = {"Male": 1, "Female": 0}
            cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non‑anginal Pain": 2, "Asymptomatic": 3}
            fbs_map = {"Yes": 1, "No": 0}
            restecg_map = {"Normal": 0, "ST‑T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            exang_map = {"No": 0, "Yes": 1}
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
            # On predict button
            if st.button("Predict", key="predict_button"):
                # Build DataFrame with all features used during training
                input_df = pd.DataFrame([
                    {
                        "age": age,
                        "sex": sex_map[sex_label],
                        "cp": cp_map[cp_label],
                        "trestbps": trestbps,
                        "chol": chol,
                        "fbs": fbs_map[fbs_label],
                        "restecg": restecg_map[restecg_label],
                        "thalach": thalach,
                        "exang": exang_map[exang_label],
                        "oldpeak": oldpeak,
                        "slope": slope_map[slope_label],
                        "ca": ca,
                        "thal": thal_map[thal_label],
                    }
                ])
                # Compute ratio features to match training
                input_df = compute_ratio_features(input_df)
                # Retrieve trained pipeline
                pipeline: Pipeline = st.session_state["trained_pipeline"]
                try:
                    # Predict probability and class
                    if hasattr(pipeline[-1], "predict_proba"):
                        prob = pipeline.predict_proba(input_df)[0][1]
                    else:
                        score = pipeline.decision_function(input_df)[0]
                        prob = 1 / (1 + np.exp(-score))
                    pred = pipeline.predict(input_df)[0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                else:
                    st.subheader("Prediction Results")
                    st.write(f"Probability of heart disease: **{prob:.2%}**")
                    st.write(f"Diagnosis: **{'Likely' if pred == 1 else 'Unlikely'}**")
                    if pred == 1:
                        st.warning(
                            "The model predicts a high likelihood of heart disease. "
                            "Please consult a healthcare professional for further evaluation."
                        )
                    else:
                        st.success(
                            "The model predicts a low likelihood of heart disease. "
                            "Keep up a healthy lifestyle and regular check‑ups."
                        )

    ########################################
    # Visualisations Tab
    ########################################
    with tabs[3]:
        st.header("Visualisations")
        vis_dir = Path(__file__).resolve().parent / "heart" / "artifacts" / "visualisations"
        if not vis_dir.exists():
            st.info("No pre‑computed visualisations found. Train the models via train.py to generate them.")
        else:
            images = list(vis_dir.glob("*.png"))
            if not images:
                st.info("No images found in visualisations directory.")
            else:
                # Display images in a grid
                cols = st.columns(3)
                for idx, img_path in enumerate(images):
                    with cols[idx % 3]:
                        st.image(str(img_path), caption=img_path.name)


if __name__ == "__main__":
    main()
