"""
Script principal de entrenamiento para clasificación de pingüinos Palmer
Este script implementa un pipeline completo de ML con las siguientes prácticas:
- Modular y extensible
- Configuración externa
- Logging completo
- Guardado de metadatos
"""

import logging
import argparse
import yaml
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# Imports de scikit-learn organizados por funcionalidad
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import del dataset
from palmerpenguins import load_penguins

# Configurar logging para el seguimiento de eventos
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()  # También mostrar en consola
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Clase responsable de toda la manipulación de datos.
    Separamos esta lógica para mantener el código organizado.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Inicializando procesador de datos")
    
    def load_data(self) -> pd.DataFrame:
        """
        Carga y prepara los datos iniciales.
        Incluye limpieza básica y feature engineering.
        """
        logger.info("Cargando dataset Palmer Penguins")
        
        # Cargar datos originales
        penguins = load_penguins()
        logger.info(f"Dataset cargado: {penguins.shape[0]} filas, {penguins.shape[1]} columnas")
        
        # Remover filas sin especie (target)
        initial_size = len(penguins)
        penguins = penguins.dropna(subset=['species'])
        logger.info(f"Removidas {initial_size - len(penguins)} filas sin especie")
        
        # Feature engineering: crear características derivadas
        penguins = self._create_features(penguins)
        
        # Logging de distribución de clases
        species_counts = penguins['species'].value_counts()
        logger.info(f"Distribución de especies: {species_counts.to_dict()}")
        
        return penguins
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características adicionales que pueden mejorar el modelo.
        """
        df = df.copy()
        
        # Ratio del pico: puede ser indicativo de la especie
        df['bill_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
        
        # Masa corporal en kg (más interpretable)
        df['body_mass_kg'] = df['body_mass_g'] / 1000
        
        # Índice de masa corporal del pingüino (masa/longitud aleta)
        df['mass_flipper_ratio'] = df['body_mass_g'] / df['flipper_length_mm']
        
        logger.info("Características adicionales creadas: bill_ratio, body_mass_kg, mass_flipper_ratio")
        return df
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Crea el pipeline de preprocesamiento.
        Esta separación permite reutilizar la misma lógica en entrenamiento e inferencia.
        """
        numeric_features = self.config['model']['features']['numeric']
        categorical_features = self.config['model']['features']['categorical']
        
        # Pipeline para características numéricas
        # Primero imputamos valores faltantes, luego normalizamos
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Mediana es robusta a outliers
            ('scaler', StandardScaler())  # Normalización necesaria para regresión logística
        ])
        
        # Pipeline para características categóricas
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Usar la categoría más común
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding
        ])
        
        # Combinar ambos transformadores
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        logger.info(f"Pipeline de preprocesamiento creado para {len(numeric_features)} características numéricas y {len(categorical_features)} categóricas")
        return preprocessor

class ModelTrainer:
    """
    Clase responsable del entrenamiento y evaluación del modelo.
    Diseñada para ser extensible a diferentes algoritmos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Inicializando entrenador de modelo")
    
    def create_model(self) -> Pipeline:
        """
        Crea el modelo según la configuración.
        Factory pattern para fácil extensión a nuevos algoritmos.
        """
        model_type = self.config['model']['type']
        
        # Diccionario de modelos disponibles
        models = {
            'logistic': LogisticRegression(
                max_iter=1000, 
                random_state=self.config['data']['random_state']
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['data']['random_state']
            ),
            'svm': SVC(
                probability=True,  # Necesario para predict_proba en la API
                random_state=self.config['data']['random_state']
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Modelo '{model_type}' no soportado. Opciones: {list(models.keys())}")
        
        # Crear pipeline completo: preprocesamiento + modelo
        data_processor = DataProcessor(self.config)
        preprocessor = data_processor.create_preprocessing_pipeline()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', models[model_type])
        ])
        
        logger.info(f"Modelo '{model_type}' creado exitosamente")
        return pipeline
    
    def train_with_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Entrena el modelo con optimización de hiperparámetros si está habilitada.
        GridSearchCV encuentra la mejor combinación de parámetros.
        """
        pipeline = self.create_model()
        
        if not self.config['hyperparameter_tuning']['enabled']:
            logger.info("Entrenando modelo sin optimización de hiperparámetros")
            pipeline.fit(X_train, y_train)
            return pipeline
        
        logger.info("Iniciando optimización de hiperparámetros")
        
        # Obtener parámetros según el tipo de modelo
        model_type = self.config['model']['type']
        param_key = f"{model_type}_params"
        
        if param_key not in self.config['hyperparameter_tuning']:
            logger.warning(f"No se encontraron parámetros para {model_type}, entrenando con valores por defecto")
            pipeline.fit(X_train, y_train)
            return pipeline
        
        # Preparar grid de parámetros con prefijo del pipeline
        param_grid = {}
        for param, values in self.config['hyperparameter_tuning'][param_key].items():
            param_grid[f'classifier__{param}'] = values
        
        # Configurar GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=self.config['hyperparameter_tuning']['cv_folds'],
            scoring='accuracy',
            n_jobs=-1,  # Usar todos los procesadores disponibles
            verbose=1   # Mostrar progreso
        )
        
        # Entrenar
        logger.info(f"Probando {len(param_grid)} combinaciones de parámetros")
        grid_search.fit(X_train, y_train)
        
        # Logging de resultados
        logger.info(f"Mejor puntuación: {grid_search.best_score_:.4f}")
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evalúa el modelo entrenado y genera métricas completas.
        """
        logger.info("Evaluando modelo en conjunto de prueba")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Validación cruzada en el conjunto de entrenamiento
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),  # Convertir a lista para JSON
            'cross_validation_mean': cv_scores.mean(),
            'cross_validation_std': cv_scores.std(),
            'class_distribution': y_test.value_counts().to_dict()
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return metrics

class MLPipeline:
    """
    Clase principal que orquesta todo el proceso de entrenamiento.
    Esta es la fachada que simplifica el uso de todas las demás clases.
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_directories()
        logger.info("Pipeline de ML inicializado")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración desde archivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Archivo de configuración {config_path} no encontrado")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error al parsear configuración YAML: {e}")
            raise
    
    def _setup_directories(self):
        """Crea directorios necesarios si no existen"""
        dirs_to_create = [
            self.config['output']['model_dir'],
            self.config['output']['log_dir']
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio asegurado: {dir_path}")
    
    def run(self) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Ejecuta el pipeline completo de entrenamiento.
        Retorna el modelo entrenado y las métricas.
        """
        logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO ===")
        
        # 1. Cargar y procesar datos
        data_processor = DataProcessor(self.config)
        penguins = data_processor.load_data()
        
        # 2. Preparar características y target
        feature_cols = (
            self.config['model']['features']['numeric'] + 
            self.config['model']['features']['categorical']
        )
        
        # Filtrar solo las columnas que existen en el dataset
        available_cols = [col for col in feature_cols if col in penguins.columns]
        missing_cols = [col for col in feature_cols if col not in penguins.columns]
        
        if missing_cols:
            logger.warning(f"Columnas no encontradas en el dataset: {missing_cols}")
        
        X = penguins[available_cols]
        y = penguins['species']
        
        logger.info(f"Características utilizadas: {available_cols}")
        logger.info(f"Shape de datos: X{X.shape}, y{y.shape}")
        
        # 3. División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y if self.config['data']['stratify'] else None
        )
        
        logger.info(f"División train/test: {len(X_train)} entrenamiento, {len(X_test)} prueba")
        
        # 4. Entrenamiento
        trainer = ModelTrainer(self.config)
        model = trainer.train_with_hyperparameter_tuning(X_train, y_train)
        
        # 5. Evaluación
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        # 6. Guardar modelo y metadatos
        self._save_artifacts(model, metrics, feature_cols)
        
        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
        return model, metrics
    
    def _save_artifacts(self, model: Pipeline, metrics: Dict[str, Any], feature_cols: list):
        """
        Guarda el modelo entrenado y toda la información relevante.
        Esta información es crucial para reproducibilidad y debugging.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = Path(self.config['output']['model_dir'])
        
        # Guardar modelo
        model_path = model_dir / f'penguin_model_{timestamp}.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado en: {model_path}")
        
        # También guardar como "latest" para fácil uso en la API
        latest_path = model_dir / 'penguin_model_latest.joblib'
        joblib.dump(model, latest_path)
        logger.info(f"Modelo también guardado como: {latest_path}")
        
        # Guardar metadatos completos
        if self.config['output']['save_metadata']:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.config['model']['type'],
                'feature_columns': feature_cols,
                'metrics': metrics,
                'config': self.config,
                'sklearn_version': joblib.__version__,
                'model_path': str(model_path)
            }
            
            metadata_path = model_dir / f'penguin_model_{timestamp}_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Metadatos guardados en: {metadata_path}")

def main():
    """Función principal que maneja argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Entrena modelo de clasificación de pingüinos Palmer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Ruta al archivo de configuración YAML'
    )
    
    args = parser.parse_args()
    
    try:
        # Crear y ejecutar pipeline
        pipeline = MLPipeline(args.config)
        model, results = pipeline.run()
        
        print("\n" + "="*50)
        print("RESUMEN DE ENTRENAMIENTO")
        print("="*50)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"CV Score: {results['cross_validation_mean']:.4f} (+/- {results['cross_validation_std'] * 2:.4f})")
        print("Modelo guardado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()