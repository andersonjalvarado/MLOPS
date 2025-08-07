"""
API FastAPI para clasificación de especies de pingüinos Palmer.
Esta API proporciona endpoints para predicción individual y por lotes,
con validación completa de datos y manejo de errores.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELOS DE DATOS PYDANTIC ====================

class PenguinFeatures(BaseModel):
    """
    Modelo de datos para las características de entrada del pingüino.
    Pydantic se encarga automáticamente de la validación y documentación.
    """
    island: str = Field(
        ..., 
        description="Isla donde fue observado el pingüino",
        example="Biscoe"
    )
    bill_length_mm: float = Field(
        ..., 
        ge=25.0, le=65.0,  # Greater equal 25, less equal 65
        description="Longitud del pico en milímetros",
        example=39.1
    )
    bill_depth_mm: float = Field(
        ..., 
        ge=10.0, le=25.0,
        description="Profundidad del pico en milímetros", 
        example=18.7
    )
    flipper_length_mm: float = Field(
        ..., 
        ge=170.0, le=240.0,
        description="Longitud de la aleta en milímetros",
        example=181.0
    )
    body_mass_g: float = Field(
        ..., 
        ge=2500.0, le=6500.0,
        description="Masa corporal en gramos",
        example=3750.0
    )
    sex: str = Field(
        ..., 
        description="Sexo del pingüino",
        example="male"
    )
    
    @validator('island')
    def validate_island(cls, v):
        """Validador personalizado para isla"""
        valid_islands = ['Biscoe', 'Dream', 'Torgersen']
        if v not in valid_islands:
            raise ValueError(f'La isla debe ser una de: {valid_islands}')
        return v
    
    @validator('sex')
    def validate_sex(cls, v):
        """Validador personalizado para sexo"""
        valid_sexes = ['male', 'female']
        if v.lower() not in valid_sexes:
            raise ValueError(f'El sexo debe ser uno de: {valid_sexes}')
        return v.lower()
    
    class Config:
        """Configuración del modelo Pydantic"""
        schema_extra = {
            "example": {
                "island": "Biscoe",
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "sex": "male"
            }
        }

class PredictionResponse(BaseModel):
    """Modelo de respuesta para predicciones individuales"""
    species: str = Field(description="Especie predicha del pingüino")
    confidence: float = Field(description="Confianza de la predicción (0-1)")
    probabilities: Dict[str, float] = Field(description="Probabilidades para cada especie")
    input_features: Dict[str, Any] = Field(description="Características de entrada utilizadas")
    model_version: str = Field(description="Versión del modelo utilizado")
    prediction_timestamp: str = Field(description="Timestamp de la predicción")

class BatchPredictionRequest(BaseModel):
    """Modelo para predicciones por lotes"""
    penguins: List[PenguinFeatures] = Field(
        ..., 
        description="Lista de pingüinos para clasificar",
        min_items=1,
        max_items=100  # Limitar el tamaño del lote
    )

class BatchPredictionResponse(BaseModel):
    """Modelo de respuesta para predicciones por lotes"""
    predictions: List[PredictionResponse]
    batch_size: int = Field(description="Número de predicciones procesadas")
    processing_time_ms: float = Field(description="Tiempo de procesamiento en milisegundos")

class HealthResponse(BaseModel):
    """Modelo de respuesta para health check"""
    status: str = Field(description="Estado del servicio")
    model_loaded: bool = Field(description="Si el modelo está cargado")
    model_version: Optional[str] = Field(description="Versión del modelo cargado")
    uptime_seconds: float = Field(description="Tiempo de actividad en segundos")
    timestamp: str = Field(description="Timestamp actual del servidor")

class ModelInfo(BaseModel):
    """Información detallada del modelo"""
    model_type: str
    feature_columns: List[str]
    target_classes: List[str]
    training_timestamp: Optional[str]
    accuracy: Optional[float]

# ==================== GESTIÓN DEL MODELO ====================

class ModelManager:
    """
    Clase singleton para gestionar el modelo ML.
    Centraliza la carga, predicciones y metadatos del modelo.
    """
    
    def __init__(self):
        self.model = None
        self.metadata = None
        self.model_version = "unknown"
        self.load_timestamp = None
        self.feature_columns = None
        
    def load_model(self, model_path: str = "models/penguin_model_latest.joblib"):
        """
        Carga el modelo entrenado y sus metadatos.
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                logger.error(f"Archivo de modelo no encontrado: {model_path}")
                return False
            
            # Cargar modelo
            self.model = joblib.load(model_path)
            self.load_timestamp = datetime.now()
            logger.info(f"Modelo cargado desde: {model_path}")
            
            # Intentar cargar metadatos
            self._load_metadata(model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            return False
    
    def _load_metadata(self, model_path: Path):
        """Intenta cargar metadatos del modelo"""
        try:
            # Buscar archivo de metadatos
            metadata_pattern = model_path.parent / f"{model_path.stem}_metadata.json"
            
            if metadata_pattern.exists():
                with open(metadata_pattern, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                    self.model_version = self.metadata.get('timestamp', 'unknown')
                    self.feature_columns = self.metadata.get('feature_columns', [])
                logger.info("Metadatos del modelo cargados")
            else:
                logger.warning("No se encontraron metadatos del modelo")
                
        except Exception as e:
            logger.warning(f"No se pudieron cargar metadatos: {e}")
    
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado y listo"""
        return self.model is not None
    
    def predict_single(self, features: PenguinFeatures) -> PredictionResponse:
        """
        Realiza predicción para un solo pingüino.
        """
        if not self.is_loaded():
            raise HTTPException(status_code=503, detail="Modelo no está cargado")
        
        try:
            # Convertir a DataFrame
            input_data = self._prepare_input_data([features])
            
            # Realizar predicción
            prediction = self.model.predict(input_data)[0]
            probabilities = self.model.predict_proba(input_data)[0]
            
            # Obtener nombres de clases
            classes = self.model.classes_
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            
            return PredictionResponse(
                species=prediction,
                confidence=float(max(probabilities)),
                probabilities=prob_dict,
                input_features=features.dict(),
                model_version=self.model_version,
                prediction_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
    
    def predict_batch(self, features_list: List[PenguinFeatures]) -> List[PredictionResponse]:
        """
        Realiza predicciones para múltiples pingüinos.
        Optimizado para procesamiento por lotes.
        """
        if not self.is_loaded():
            raise HTTPException(status_code=503, detail="Modelo no está cargado")
        
        try:
            # Preparar datos de entrada
            input_data = self._prepare_input_data(features_list)
            
            # Realizar predicciones
            predictions = self.model.predict(input_data)
            probabilities = self.model.predict_proba(input_data)
            classes = self.model.classes_
            
            # Crear respuestas
            results = []
            timestamp = datetime.now().isoformat()
            
            for i, (pred, probs, features) in enumerate(zip(predictions, probabilities, features_list)):
                prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
                
                results.append(PredictionResponse(
                    species=pred,
                    confidence=float(max(probs)),
                    probabilities=prob_dict,
                    input_features=features.dict(),
                    model_version=self.model_version,
                    prediction_timestamp=timestamp
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en predicción por lotes: {e}")
            raise HTTPException(status_code=500, detail=f"Error en predicción por lotes: {str(e)}")
    
    def _prepare_input_data(self, features_list: List[PenguinFeatures]) -> pd.DataFrame:
        """
        Prepara los datos de entrada para el modelo.
        Incluye feature engineering que coincida con el entrenamiento.
        """
        # Convertir a DataFrame
        data_dicts = [f.dict() for f in features_list]
        df = pd.DataFrame(data_dicts)
        
        # Feature engineering (debe coincidir exactamente con train.py)
        df['bill_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
        df['body_mass_kg'] = df['body_mass_g'] / 1000
        df['mass_flipper_ratio'] = df['body_mass_g'] / df['flipper_length_mm']
        
        return df
    
    def get_model_info(self) -> ModelInfo:
        """Retorna información detallada del modelo"""
        if not self.is_loaded():
            raise HTTPException(status_code=503, detail="Modelo no está cargado")
        
        return ModelInfo(
            model_type=self.metadata.get('model_type', 'unknown') if self.metadata else 'unknown',
            feature_columns=self.feature_columns or [],
            target_classes=list(self.model.classes_) if hasattr(self.model, 'classes_') else [],
            training_timestamp=self.metadata.get('timestamp') if self.metadata else None,
            accuracy=self.metadata.get('metrics', {}).get('accuracy') if self.metadata else None
        )

# Crear instancia global del gestor de modelos
model_manager = ModelManager()

# ==================== APLICACIÓN FASTAPI ====================

# Crear aplicación con metadatos completos
app = FastAPI(
    title="Palmer Penguins Species Classifier API",
    description="""
    API para clasificación de especies de pingüinos Palmer basada en características físicas.
    
    Esta API utiliza un modelo de machine learning entrenado con el dataset Palmer Penguins
    para predecir la especie (Adelie, Chinstrap, o Gentoo) basándose en:
    - Dimensiones del pico (longitud y profundidad)
    - Longitud de la aleta
    - Masa corporal
    - Isla de origen
    - Sexo
    
    ## Características principales:
    - Predicción individual y por lotes
    - Validación robusta de entrada
    - Métricas de confianza
    - Health checks
    - Documentación automática
    """,
    version="1.0.0",
    contact={
        "name": "Equipo 5 - MLOPS",
        "email": "andersonjalvarado@javeriana.edu.co"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Configurar CORS para permitir acceso desde navegadores
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Variable global para tracking de uptime
app_start_time = datetime.now()

# ==================== EVENTOS DE CICLO DE VIDA ====================

@app.on_event("startup")
async def startup_event():
    """
    Evento que se ejecuta al iniciar la aplicación.
    Carga el modelo y configura el entorno.
    """
    logger.info("Iniciando aplicación FastAPI")
    
    # Asegurar que el directorio de modelos existe
    Path("models").mkdir(exist_ok=True)
    
    # Intentar cargar el modelo
    model_loaded = model_manager.load_model()
    
    if model_loaded:
        logger.info("✅ Aplicación iniciada exitosamente con modelo cargado")
    else:
        logger.warning("⚠️ Aplicación iniciada pero sin modelo cargado")

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la aplicación"""
    logger.info("Cerrando aplicación FastAPI")

# ==================== ENDPOINTS ====================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "Palmer Penguins Species Classifier API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_manager.is_loaded(),
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint para monitoreo.
    """
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded() else "degraded",
        model_loaded=model_manager.is_loaded(),
        model_version=model_manager.model_version if model_manager.is_loaded() else None,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Obtiene información detallada del modelo actual"""
    return model_manager.get_model_info()

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_species(features: PenguinFeatures):
    """
    Predice la especie de un pingüino basándose en sus características físicas.
    
    Retorna la especie predicha junto con las probabilidades para cada clase
    y el nivel de confianza de la predicción.
    """
    logger.info(f"Predicción solicitada para pingüino de isla {features.island}")
    return model_manager.predict_single(features)

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_species_batch(request: BatchPredictionRequest):
    """
    Predice las especies de múltiples pingüinos en una sola petición.
    Optimizado para procesamiento por lotes.
    """
    start_time = datetime.now()
    
    logger.info(f"Predicción por lotes solicitada para {len(request.penguins)} pingüinos")
    
    # Realizar predicciones
    predictions = model_manager.predict_batch(request.penguins)
    
    # Calcular tiempo de procesamiento
    end_time = datetime.now()
    processing_time_ms = (end_time - start_time).total_seconds() * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        batch_size=len(predictions),
        processing_time_ms=processing_time_ms
    )

# ==================== MANEJO DE ERRORES ====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Manejo personalizado de errores de validación"""
    logger.error(f"Error de validación: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Error de validación", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejo general de excepciones no capturadas"""
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Error interno del servidor", "detail": "Por favor contacte al administrador"}
    )

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    import uvicorn
    
    # Configuración para desarrollo
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8989,
        log_level="info",
        reload=True  # Auto-reload en desarrollo
    )