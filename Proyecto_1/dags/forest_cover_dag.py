"""
DAG para el pipeline de MLOps del proyecto Forest Cover Type.
Este DAG realiza:
1. Extracción de datos desde API externa
2. Almacenamiento incremental en PostgreSQL
3. Preprocesamiento de datos acumulados
4. Entrenamiento de modelos con GridSearchCV
5. Registro de experimentos en MLflow
"""

from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Airflow imports
from airflow.decorators import dag, task
from airflow.models import Variable

# Database imports
import psycopg2
from psycopg2.extras import execute_values

# HTTP client
import requests

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Configuración de constantes
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "http://10.43.100.103:8080")
POSTGRES_CONN = os.getenv("AIRFLOW_CONN_POSTGRES_BATCHES", 
                          "postgresql://batchuser:batchpass123@postgres-batches:5432/batches_db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
GROUP_NUMBER = 5
MAX_EXECUTIONS = 10

# Columnas del dataset
FEATURE_COLUMNS = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type'
]
TARGET_COLUMN = 'Cover_Type'
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}


@dag(
    dag_id='forest_cover_mlops_pipeline',
    default_args=default_args,
    description='Pipeline MLOps para clasificación de cobertura forestal',
    schedule='*/5 * * * *',  # Ejecutar cada 5 minutos
    start_date=datetime(2025, 10, 5),
    max_active_runs=1,
    catchup=False,
    tags=['mlops', 'forest-cover', 'classification'],
)
def forest_cover_pipeline():
    """
    DAG principal que orquesta todo el pipeline de MLOps.
    Se ejecuta manualmente para cada batch de datos.
    """

    @task
    def extract_data_from_api() -> Dict[str, Any]:
        """
        Tarea 1: Extrae datos desde la API externa.
        
        Returns:
            Dictionary con batch_number, group_number y los datos extraídos
        """
        try:
            # Verificar si ya se alcanzó el máximo de ejecuciones
            conn = psycopg2.connect(POSTGRES_CONN)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT batch_number) FROM data_batches WHERE group_number = %s", (GROUP_NUMBER,))
            batch_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            if batch_count >= MAX_EXECUTIONS:
                print(f"✓ Ya se alcanzó el máximo de {MAX_EXECUTIONS} ejecuciones")
                raise Exception("MaxExecutionsReached")
            
            # Realizar petición GET a la API
            url = f"{EXTERNAL_API_URL}/data"
            params = {"group_number": GROUP_NUMBER}
            
            print(f"Solicitando datos desde: {url} con group_number={GROUP_NUMBER}")
            response = requests.get(url, params=params, timeout=30)
            
            # Verificar si se alcanzó el límite de muestras (código 400)
            if response.status_code == 400:
                print("✓ API retornó código 400: Se alcanzó el número mínimo de muestras")
                raise Exception("MinimumSamplesReached")
            
            response.raise_for_status()
            
            # Parsear respuesta JSON
            api_data = response.json()
            
            batch_number = api_data.get('batch_number')
            data_rows = api_data.get('data', [])
            
            print(f"✓ Datos extraídos exitosamente")
            print(f"  - Batch number: {batch_number}")
            print(f"  - Group number: {api_data.get('group_number')}")
            print(f"  - Registros obtenidos: {len(data_rows)}")
            
            return {
                'batch_number': batch_number,
                'group_number': GROUP_NUMBER,
                'data': data_rows,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error al extraer datos de la API: {e}")
            raise
        except Exception as e:
            if "MaxExecutionsReached" in str(e) or "MinimumSamplesReached" in str(e):
                raise
            print(f"✗ Error inesperado: {e}")
            raise
    
    @task
    def store_batch_in_postgres(extracted_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Tarea 2: Almacena el batch extraído en PostgreSQL.
        
        Args:
            extracted_data: Datos extraídos de la API
            
        Returns:
            Dictionary con estadísticas de almacenamiento
        """
        try:
            conn = psycopg2.connect(POSTGRES_CONN)
            cursor = conn.cursor()
            
            batch_number = extracted_data['batch_number']
            group_number = extracted_data['group_number']
            data_rows = extracted_data['data']
            
            # Convertir datos a formato JSONB para PostgreSQL
            data_jsonb = json.dumps(data_rows)
            row_count = len(data_rows)
            
            # Insertar batch en la tabla data_batches
            insert_query = """
                INSERT INTO data_batches 
                (batch_number, group_number, data, row_count, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """
            
            cursor.execute(insert_query, (
                batch_number,
                group_number,
                data_jsonb,
                row_count,
                extracted_data['timestamp']
            ))
            
            inserted_id = cursor.fetchone()[0]
            
            # Actualizar o insertar en batch_status
            status_query = """
                INSERT INTO batch_status (batch_number, total_records, is_complete)
                VALUES (%s, %s, %s)
                ON CONFLICT (batch_number) 
                DO UPDATE SET 
                    total_records = batch_status.total_records + EXCLUDED.total_records,
                    last_updated = CURRENT_TIMESTAMP;
            """
            
            cursor.execute(status_query, (batch_number, row_count, False))
            
            conn.commit()
            
            # Obtener estadísticas actuales
            cursor.execute("""
                SELECT COUNT(DISTINCT batch_number), SUM(row_count)
                FROM data_batches
                WHERE group_number = %s;
            """, (group_number,))
            
            total_batches, total_records = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            print(f"✓ Batch almacenado en PostgreSQL")
            print(f"  - ID insertado: {inserted_id}")
            print(f"  - Batch number: {batch_number}")
            print(f"  - Registros en este batch: {row_count}")
            print(f"  - Total de batches acumulados: {total_batches}")
            print(f"  - Total de registros acumulados: {total_records}")
            
            return {
                'batch_number': batch_number,
                'records_in_batch': row_count,
                'total_batches': total_batches,
                'total_records': total_records
            }
            
        except Exception as e:
            print(f"✗ Error al almacenar en PostgreSQL: {e}")
            raise
    
    @task
    def load_accumulated_data(storage_stats: Dict[str, int]) -> pd.DataFrame:
        """
        Tarea 3: Carga todos los datos acumulados desde PostgreSQL.
        
        Args:
            storage_stats: Estadísticas del almacenamiento
            
        Returns:
            DataFrame con todos los datos acumulados
        """
        try:
            conn = psycopg2.connect(POSTGRES_CONN)
            
            # Cargar todos los batches almacenados
            query = """
                SELECT batch_number, data, timestamp
                FROM data_batches
                WHERE group_number = %s
                ORDER BY batch_number, timestamp;
            """
            
            cursor = conn.cursor()
            cursor.execute(query, (GROUP_NUMBER,))
            
            all_rows = []
            batches_loaded = set()
            
            for batch_num, data_json, timestamp in cursor.fetchall():
                batches_loaded.add(batch_num)
                # Parsear datos JSON
                #batch_data = json.loads(data_json)
                batch_data = data_json
                all_rows.extend(batch_data)
            
            cursor.close()
            conn.close()
            
            # Convertir a DataFrame
            df = pd.DataFrame(all_rows, columns=ALL_COLUMNS)
            
            # Convertir columnas numéricas
            for col in FEATURE_COLUMNS:
                if col in ['Wilderness_Area', 'Soil_Type']:
                    # Estas son categóricas, las codificaremos después
                    continue
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
            
            print(f"✓ Datos acumulados cargados desde PostgreSQL")
            print(f"  - Batches únicos cargados: {sorted(batches_loaded)}")
            print(f"  - Total de registros: {len(df)}")
            print(f"  - Forma del DataFrame: {df.shape}")
            print(f"  - Columnas: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error al cargar datos acumulados: {e}")
            raise
    
    @task
    def preprocess_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tarea 4: Preprocesa los datos para entrenamiento.
        
        Args:
            df: DataFrame con datos acumulados
            
        Returns:
            Dictionary con datos preprocesados y divididos
        """
        try:
            print("Iniciando preprocesamiento de datos...")
            
            # Eliminar filas con valores nulos
            df_clean = df.dropna()
            print(f"  - Registros después de eliminar nulos: {len(df_clean)}")
            
            # Codificar variables categóricas (Wilderness_Area y Soil_Type)
            # Usando Label Encoding para simplificar
            from sklearn.preprocessing import LabelEncoder
            
            le_wilderness = LabelEncoder()
            le_soil = LabelEncoder()
            
            df_clean['Wilderness_Area_Encoded'] = le_wilderness.fit_transform(df_clean['Wilderness_Area'])
            df_clean['Soil_Type_Encoded'] = le_soil.fit_transform(df_clean['Soil_Type'])
            
            # Preparar features y target
            feature_cols = [col for col in FEATURE_COLUMNS if col not in ['Wilderness_Area', 'Soil_Type']]
            feature_cols.extend(['Wilderness_Area_Encoded', 'Soil_Type_Encoded'])
            
            X = df_clean[feature_cols].values
            y = df_clean[TARGET_COLUMN].values
            
            # Dividir en train y test (80/20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"✓ Preprocesamiento completado")
            print(f"  - Features utilizadas: {len(feature_cols)}")
            print(f"  - Tamaño de entrenamiento: {X_train.shape}")
            print(f"  - Tamaño de prueba: {X_test.shape}")
            print(f"  - Clases en target: {np.unique(y)}")
            
            return {
                'X_train': X_train.tolist(),
                'X_test': X_test.tolist(),
                'y_train': y_train.tolist(),
                'y_test': y_test.tolist(),
                'feature_names': feature_cols,
                'n_samples': len(df_clean),
                'n_features': len(feature_cols)
            }
            
        except Exception as e:
            print(f"✗ Error en preprocesamiento: {e}")
            raise
    
    @task
    def train_and_log_models(preprocessed_data: Dict[str, Any], 
                            storage_stats: Dict[str, int]) -> Dict[str, Any]:
        """
        Tarea 5: Entrena modelos con GridSearchCV y registra en MLflow.
        
        Args:
            preprocessed_data: Datos preprocesados
            storage_stats: Estadísticas de almacenamiento
            
        Returns:
            Dictionary con resultados de entrenamiento
        """
        try:
            # Configurar MLflow
            import mlflow
            import mlflow.sklearn
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            experiment_name = "forest_cover_classification"
            
            # Crear o obtener experimento
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
            
            # Reconstruir arrays desde listas
            X_train = np.array(preprocessed_data['X_train'])
            X_test = np.array(preprocessed_data['X_test'])
            y_train = np.array(preprocessed_data['y_train'])
            y_test = np.array(preprocessed_data['y_test'])
            
            # Definir modelos a entrenar
            models = {
                'RandomForest': {
                    'estimator': RandomForestClassifier(random_state=42),
                    'params': {
                        'classifier__n_estimators': [50, 100],
                        'classifier__max_depth': [10, 20, None],
                        'classifier__min_samples_split': [2, 5]
                    }
                },
                'GradientBoosting': {
                    'estimator': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'classifier__n_estimators': [50, 100],
                        'classifier__learning_rate': [0.05, 0.1],
                        'classifier__max_depth': [3, 5]
                    }
                },
                'LogisticRegression': {
                    'estimator': LogisticRegression(max_iter=1000, random_state=42),
                    'params': {
                        'classifier__C': [0.1, 1.0, 10.0],
                        'classifier__solver': ['lbfgs', 'saga']
                    }
                }
            }
            
            results = {}
            best_model = None
            best_score = 0
            
            print(f"Entrenando {len(models)} modelos con GridSearchCV...")
            
            for model_name, model_config in models.items():
                print(f"\n{'='*50}")
                print(f"Modelo: {model_name}")
                print(f"{'='*50}")
                
                # Crear pipeline con escalado y clasificador
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model_config['estimator'])
                ])
                
                # GridSearchCV con validación cruzada
                grid_search = GridSearchCV(
                    pipeline,
                    model_config['params'],
                    cv=3,  # 3-fold cross validation
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Entrenar
                with mlflow.start_run(run_name=f"{model_name}_batch_{storage_stats['total_batches']}"):
                    
                    # Log de parámetros del experimento
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("total_batches", storage_stats['total_batches'])
                    mlflow.log_param("total_records", storage_stats['total_records'])
                    mlflow.log_param("n_features", preprocessed_data['n_features'])
                    mlflow.log_param("cv_folds", 3)
                    
                    # Entrenar modelo
                    grid_search.fit(X_train, y_train)
                    
                    # Mejores parámetros
                    print(f"  Mejores parámetros: {grid_search.best_params_}")
                    mlflow.log_params(grid_search.best_params_)
                    
                    # Predicciones
                    y_pred_train = grid_search.predict(X_train)
                    y_pred_test = grid_search.predict(X_test)
                    
                    # Métricas
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=3)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    print(f"  Train Accuracy: {train_accuracy:.4f}")
                    print(f"  Test Accuracy: {test_accuracy:.4f}")
                    print(f"  Test F1 (weighted): {test_f1_weighted:.4f}")
                    print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
                    
                    # Log de métricas
                    mlflow.log_metric("train_accuracy", train_accuracy)
                    mlflow.log_metric("test_accuracy", test_accuracy)
                    mlflow.log_metric("test_f1_weighted", test_f1_weighted)
                    mlflow.log_metric("cv_mean_score", cv_mean)
                    mlflow.log_metric("cv_std_score", cv_std)
                    
                    # Log del modelo
                    mlflow.sklearn.log_model(
                        grid_search.best_estimator_,
                        f"model_{model_name.lower()}",
                        registered_model_name=f"forest_cover_{model_name.lower()}"
                    )
                    
                    # Guardar resultados
                    results[model_name] = {
                        'test_accuracy': test_accuracy,
                        'test_f1_weighted': test_f1_weighted,
                        'cv_mean_score': cv_mean,
                        'best_params': grid_search.best_params_
                    }
                    
                    # Rastrear mejor modelo
                    if test_accuracy > best_score:
                        best_score = test_accuracy
                        best_model = model_name
            
            print(f"\n{'='*50}")
            print(f"✓ Entrenamiento completado")
            print(f"  Mejor modelo: {best_model} (Accuracy: {best_score:.4f})")
            print(f"{'='*50}")
            
            return {
                'results': results,
                'best_model': best_model,
                'best_score': best_score,
                'experiment_name': experiment_name,
                'mlflow_uri': MLFLOW_TRACKING_URI
            }
            
        except Exception as e:
            print(f"✗ Error en entrenamiento: {e}")
            raise
    
    @task
    def log_pipeline_summary(training_results: Dict[str, Any],
                            storage_stats: Dict[str, int]) -> None:
        """
        Tarea 6: Registra un resumen del pipeline completo.
        
        Args:
            training_results: Resultados del entrenamiento
            storage_stats: Estadísticas de almacenamiento
        """
        print("\n" + "="*70)
        print("RESUMEN DEL PIPELINE DE MLOPS")
        print("="*70)
        print(f"Batch procesado: #{storage_stats['batch_number']}")
        print(f"Total de batches acumulados: {storage_stats['total_batches']}/10")
        print(f"Total de registros: {storage_stats['total_records']}")
        print(f"\nMejor modelo: {training_results['best_model']}")
        print(f"Accuracy: {training_results['best_score']:.4f}")
        print(f"\nMLflow Tracking URI: {training_results['mlflow_uri']}")
        print(f"Experimento: {training_results['experiment_name']}")
        print("="*70)
        print(f"✓ Pipeline ejecutado exitosamente")
        print("="*70 + "\n")
    
    # Definir flujo del DAG
    api_data = extract_data_from_api()
    storage_stats = store_batch_in_postgres(api_data)
    accumulated_data = load_accumulated_data(storage_stats)
    preprocessed = preprocess_data(accumulated_data)
    training_results = train_and_log_models(preprocessed, storage_stats)
    log_pipeline_summary(training_results, storage_stats)


# Instanciar el DAG
forest_cover_dag = forest_cover_pipeline()