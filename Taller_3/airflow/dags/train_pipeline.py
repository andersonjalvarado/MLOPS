from __future__ import annotations

import os
from datetime import datetime

import joblib
import pandas as pd
from airflow.decorators import dag, task
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sqlalchemy import create_engine, text
from sklearn.decomposition import PCA
import urllib.request
import json


def get_mysql_engine():
    host = os.environ.get("DATA_DB_HOST", "mysql")
    user = os.environ.get("DATA_DB_USER", "penguin")
    password = os.environ.get("DATA_DB_PASSWORD", "penguin")
    dbname = os.environ.get("DATA_DB_NAME", "penguins_data")
    return create_engine(f"mysql+pymysql://{user}:{password}@{host}/{dbname}")


def get_mysql_url() -> str:
    host = os.environ.get("DATA_DB_HOST", "mysql")
    user = os.environ.get("DATA_DB_USER", "penguin")
    password = os.environ.get("DATA_DB_PASSWORD", "penguin")
    dbname = os.environ.get("DATA_DB_NAME", "penguins_data")
    return f"mysql+pymysql://{user}:{password}@{host}/{dbname}"


def read_sql_robustly(query: str) -> pd.DataFrame:
    """
    Reads data from MySQL using pure SQLAlchemy to bypass pandas's problematic
    DBAPI/connection detection logic within the Airflow task environment.
    """
    engine = get_mysql_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        columns = result.keys()
        return pd.DataFrame(rows, columns=columns)


ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/opt/artifacts")
CLEAN_TABLE = "penguins_clean"


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["penguins", "ml"],
    default_args={"owner": "airflow"},
    description="Entrenamiento del modelo y persistencia de artefactos",
)
def train_pipeline():
    @task(task_id="fit_preprocessor", doc_md="Ajusta preprocesador (scaler+OHE) y guarda artifact")
    def fit_preprocessor():
        df = read_sql_robustly(f"SELECT * FROM {CLEAN_TABLE}")

        feature_columns_numeric = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        feature_columns_categorical = ["island", "sex"]

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, feature_columns_numeric),
                ("cat", categorical_transformer, feature_columns_categorical),
            ]
        )

        X = df[feature_columns_numeric + feature_columns_categorical]
        preprocessor.fit(X)

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))

    @task(task_id="train_model", doc_md="Entrena RandomForest y guarda modelo y label encoder")
    def train_model():
        df = read_sql_robustly(f"SELECT * FROM {CLEAN_TABLE}")

        feature_columns_numeric = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        feature_columns_categorical = ["island", "sex"]

        X = df[feature_columns_numeric + feature_columns_categorical]
        y = df["species"].astype(str)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))

        # We split the original feature dataframe to keep it for visualization
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Now we transform the training data for the model
        X_train_processed = preprocessor.transform(X_train_orig)

        # Fit a PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_train_pca = pca.fit_transform(X_train_processed)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_processed, y_train)

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.joblib"))
        joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
        joblib.dump(pca, os.path.join(ARTIFACTS_DIR, "pca.joblib"))
        
        # Save the raw training data, labels, and PCA-transformed features for visualization
        training_data_for_viz = {
            "features_raw": X_train_orig,
            "features_pca": pd.DataFrame(X_train_pca, columns=["pca1", "pca2"]),
            "labels_encoded": y_train,
        }
        joblib.dump(
            training_data_for_viz, os.path.join(ARTIFACTS_DIR, "training_data_for_viz.joblib")
        )

    fit = fit_preprocessor()
    train = train_model()

    @task(task_id="warmup_api", doc_md="Llama al endpoint /warmup de la API para cargar artefactos")
    def warmup_api():
        api_url = os.environ.get("API_WARMUP_URL", "http://api:8000/warmup")
        try:
            with urllib.request.urlopen(api_url, timeout=10) as resp:
                body = resp.read()
                try:
                    data = json.loads(body)
                except Exception:
                    data = {"raw": body.decode("utf-8", errors="ignore")}
                print("Warmup response:", data)
        except Exception as exc:  # noqa: BLE001
            # No fallar el DAG por warmup, solo registrar
            print("Warmup failed:", exc)

    warm = warmup_api()

    fit >> train >> warm


train_pipeline()


