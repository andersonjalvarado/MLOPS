import pandas as pd
import numpy as np
import mlflow
import json
from sqlalchemy import create_engine
from datetime import datetime
from airflow.decorators import dag, task
from airflow.models import Variable
from sklearn.linear_model import LogisticRegression

DB_RAW_HOST = 'BD_raw'
DB_RAW_PORT = '5541'
DB_RAW_USER = 'Admin'
DB_RAW_PASSWORD = 'Admin'
DB_RAW_NAME = 'raw_data'
DB_RAW_TABLE = 'diabetic_data'

MLFLOW_TRACKING_URI = 'http://mlflow:5000'
MLFLOW_EXPERIMENT_NAME = 'diabetes_warm_start_training'
BATCH_SIZE = 15000
OFFSET_VARIABLE_KEY = "warm_start_offset"

def transform_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'])
    df.replace('?', np.nan, inplace=True)
    df['readmitted'] = df['readmitted'].replace({'>30': 'NO', '<30': 'YES'})
    df['readmitted'] = df['readmitted'].map({'NO': 0, 'YES': 1})
    df.dropna(subset=['readmitted'], inplace=True)
    df['race'].fillna(df['race'].mode()[0], inplace=True)
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col].fillna('Unknown', inplace=True)
        rare = df[col].value_counts()[df[col].value_counts() < 10].index
        df[col] = df[col].replace(rare, 'Other')
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True, dtype=float)
    return df

@dag(
    dag_id='3_warm_start_training',
    start_date=datetime(2025, 11, 2),
    schedule_interval=None,
    catchup=False,
    tags=['training', 'warm-start']
)
def warm_start_training_dag():

    @task
    def get_current_offset() -> int:
        return int(Variable.get(OFFSET_VARIABLE_KEY, default_var=0))

    @task
    def extract_and_transform_batch(offset: int) -> pd.DataFrame:
        url = f"postgresql://{DB_RAW_USER}:{DB_RAW_PASSWORD}@{DB_RAW_HOST}:{DB_RAW_PORT}/{DB_RAW_NAME}"
        engine = create_engine(url)
        q = f"SELECT * FROM {DB_RAW_TABLE} ORDER BY patient_nbr LIMIT {BATCH_SIZE} OFFSET {offset};"
        df = pd.read_sql(q, engine)
        if df.empty: raise ValueError("No hay nuevos datos.")
        return transform_data_logic(df)

    @task
    def get_latest_model_info(offset: int) -> dict:
        if offset == 0: return {"run_id": None}
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs.empty:
            return {"run_id": runs.iloc[0]["run_id"]}
        return {"run_id": None}

    @task
    def train_with_warm_start(df: pd.DataFrame, latest_model_info: dict, offset: int):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        X_new = df.drop('readmitted', axis=1)
        y_new = df['readmitted']
        prev_id = latest_model_info.get("run_id")

        with mlflow.start_run(run_name=f"Warm_start_run_offset_{offset}"):
            if prev_id:
                model_uri = f"runs:/{prev_id}/warm_start_model"
                client = mlflow.tracking.MlflowClient()
                local_path = client.download_artifacts(prev_id, "model_columns.json")
                with open(local_path, "r") as f:
                    prev_cols = json.load(f)["columns"]
                model = mlflow.sklearn.load_model(model_uri)
                model.warm_start = True
                cols_to_drop = [c for c in X_new.columns if c not in prev_cols]
                X_new.drop(columns=cols_to_drop, inplace=True, errors='ignore')
                X_new = X_new.reindex(columns=prev_cols, fill_value=0)
                mlflow.log_param("started_from_run", prev_id)
            else:
                model = LogisticRegression(max_iter=1000, random_state=74, warm_start=True)

            model.fit(X_new, y_new)
            acc = model.score(X_new, y_new)
            mlflow.log_params({"offset": offset, "warm_start_enabled": True})
            mlflow.log_metric("batch_accuracy", acc)
            mlflow.sklearn.log_model(model, "warm_start_model")
            mlflow.log_dict({"columns": list(X_new.columns)}, "model_columns.json")
        return "OK"

    @task
    def update_offset(offset: int):
        Variable.set(OFFSET_VARIABLE_KEY, str(offset + BATCH_SIZE))

    off = get_current_offset()
    info = get_latest_model_info(off)
    df = extract_and_transform_batch(off)
    res = train_with_warm_start(df, info, off)
    res >> update_offset(off)

warm_start_training_dag_instance = warm_start_training_dag()
