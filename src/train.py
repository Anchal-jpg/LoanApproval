from pathlib import Path
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "loan_approval_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "loan_pipeline.joblib"
MLFLOW_DB = PROJECT_DIR / "mlflow.db"


def load_training_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame.columns = frame.columns.str.strip()
    frame = frame.drop(columns=[column for column in ("loan_id",) if column in frame.columns])
    frame = frame.dropna(subset=["loan_status"])
    frame["loan_status"] = frame["loan_status"].str.strip().map(
        {"Approved": 1, "Rejected": 0}
    )
    if frame["loan_status"].isna().any():
        raise ValueError("loan_status contains an unsupported value")
    return frame


def build_pipeline(frame: pd.DataFrame) -> Pipeline:
    X = frame.drop(columns=["loan_status"])
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def train() -> dict:
    frame = load_training_data()
    X = frame.drop(columns=["loan_status"])
    y = frame["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("loan-approval")

    pipeline = build_pipeline(frame)
    with mlflow.start_run(run_name="logistic-regression") as run:
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = pipeline.predict(X_test)

        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(y_test, predictions, zero_division=0),
            "test_recall": recall_score(y_test, predictions, zero_division=0),
            "test_f1": f1_score(y_test, predictions, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, probabilities),
        }
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "test_size": 0.2,
                "class_weight": "balanced",
                "random_state": 42,
            }
        )
        mlflow.log_metrics(metrics)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.sklearn.log_model(
            pipeline,
            name="loan_approval_pipeline",
            registered_model_name="loan-approval",
            skops_trusted_types=["numpy.dtype"],
        )

    print(f"Run ID: {run.info.run_id}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    return {"run_id": run.info.run_id, "metrics": metrics}


if __name__ == "__main__":
    train()
