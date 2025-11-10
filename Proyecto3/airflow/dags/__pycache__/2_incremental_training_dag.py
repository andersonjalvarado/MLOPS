import pandas as pd
import numpy as np
import mlflow
from sqlalchemy import create_engine
from datetime import datetime
from airflow.decorators import dag, task
from airflow.models import Variable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Variables y Conexiones (Asegúrate de que estas coincidan con tu docker-compose.yml) ---
DB_RAW_HOST = 'BD_raw'
DB_RAW_PORT = '5541' # El puerto INTERNO que definiste en el 'command'
DB_RAW_USER = 'Admin'
DB_RAW_PASSWORD = 'Admin'
DB_RAW_NAME = 'raw_data'
DB_RAW_TABLE = 'diabetic_data'

MLFLOW_TRACKING_URI = 'http://mlflow:5000'
MLFLOW_EXPERIMENT_NAME = 'diabetes_incremental_training'
BATCH_SIZE = 15000
OFFSET_VARIABLE_KEY = "data_ingestion_offset"

# --- Lógica de Transformación ---
def transform_data_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'])
    df.replace('?', np.nan, inplace=True)
    df['readmitted'] = df['readmitted'].replace({'>30': 'NO', '<30': 'YES'})
    df['readmitted'] = df['readmitted'].map({'NO': 0, 'YES': 1})
    df.dropna(subset=['readmitted'], inplace=True)
    df['race'].fillna(df['race'].mode()[0], inplace=True)
    
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col].fillna('Unknown', inplace=True)
        # Una forma más segura de hacer esto para evitar errores con series vacías
        counts = df[col].value_counts()
        rare = counts[counts < 10].index
        if len(rare) > 0:
            df[col] = df[col].replace(rare, 'Other')
    
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True, dtype=float)
    return df

# --- Definición del DAG ---
@dag(
    dag_id='2_incremental_training', 
    description='Entrena un modelo con lotes de 15k datos de forma aislada.',
    start_date=datetime(2025, 11, 2), 
    schedule_interval=None, 
    catchup=False,
    tags=['training', 'incremental']
)
def incremental_training_dag():
    @task
    def get_current_offset() -> int:
        return int(Variable.get(OFFSET_VARIABLE_KEY, default_var=0))

    @task
    def extract_and_transform_batch(offset: int) -> pd.DataFrame:
        url = f"postgresql://{DB_RAW_USER}:{DB_RAW_PASSWORD}@{DB_RAW_HOST}:{DB_RAW_PORT}/{DB_RAW_NAME}"
        engine = create_engine(url)
        q = f"SELECT * FROM {DB_RAW_TABLE} ORDER BY patient_nbr LIMIT {BATCH_SIZE} OFFSET {offset};"
        df = pd.read_sql(q, engine)
        if df.empty:
            raise ValueError("No hay más datos para procesar. Reinicia la variable de offset a 0 si quieres empezar de nuevo.")
        return transform_data_logic(df)
        
    @task
    def train_on_batch(df: pd.DataFrame, offset: int):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        X, y = df.drop('readmitted', axis=1), df['readmitted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)
        
        with mlflow.start_run(run_name=f"Training_run_offset_{offset}"):
            model = LogisticRegression(max_iter=1000, random_state=74)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            
            # --- CAMBIO APLICADO ---
            mlflow.log_metric("batch_accuracy", acc)
            # --- FIN DEL CAMBIO ---

            mlflow.log_params({"offset": offset, "batch_size": BATCH_SIZE})
            mlflow.sklearn.log_model(model, "incremental_model")
        
        return "OK"
        
    @task
    def update_offset(offset: int):
        new_offset = offset + BATCH_SIZE
        Variable.set(OFFSET_VARIABLE_KEY, str(new_offset))
        print(f"Offset actualizado a {new_offset}")
        return new_offset

    # --- Flujo del DAG ---
    offset = get_current_offset()
    transformed_df = extract_and_transform_batch(offset)
    training_result = train_on_batch(transformed_df, offset)
    
    # La tarea de update_offset se ejecuta después del entrenamiento
    training_result >> update_offset(offset)

# --- Instancia del DAG ---
incremental_training_dag_instance = incremental_training_dag()