import os
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any

# Configurar MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

app = FastAPI(
    title="ML Inference API",
    description="API para inferencia con modelos de MLflow",
    version="1.0.0"
)

# Variable global para el modelo cargado
loaded_model = None
model_info = {}

class PredictionRequest(BaseModel):
    """Esquema para solicitud de predicción"""
    data: Dict[str, Any]

class PredictionResponse(BaseModel):
    """Esquema para respuesta de predicción"""
    prediction: Any
    model_name: str
    model_version: str

def load_model():
    """Carga el modelo desde MLflow"""
    global loaded_model, model_info
    
    try:
        client = mlflow.MlflowClient()
        
        # Intentar cargar modelo desde Model Registry con stage Production
        model_names = ["forest_cover_randomforest", "forest_cover_gradientboosting", 
                      "forest_cover_logisticregression", "production_model"]
        
        model_loaded = False
        
        # Intentar primero con modelos en Production
        for model_name in model_names:
            try:
                versions = client.get_latest_versions(model_name, stages=["Production"])
                if versions:
                    model_version = versions[0]
                    model_uri = f"models:/{model_name}/Production"
                    loaded_model = mlflow.pyfunc.load_model(model_uri)
                    
                    model_info = {
                        "name": model_name,
                        "version": model_version.version,
                        "stage": "Production",
                        "run_id": model_version.run_id
                    }
                    
                    print(f"Modelo cargado exitosamente desde Registry: {model_info}")
                    model_loaded = True
                    break
            except Exception as e:
                continue
        
        # Si no hay modelo en Production, cargar el último run del experimento
        if not model_loaded:
            experiment = client.get_experiment_by_name("forest_cover_classification")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.test_accuracy DESC"],
                    max_results=1
                )
                
                if runs:
                    best_run = runs[0]
                    # Buscar el artifact path del modelo
                    artifacts = client.list_artifacts(best_run.info.run_id)
                    model_artifact = None
                    for artifact in artifacts:
                        if artifact.path.startswith("model_"):
                            model_artifact = artifact.path
                            break
                    
                    if model_artifact:
                        model_uri = f"runs:/{best_run.info.run_id}/{model_artifact}"
                        loaded_model = mlflow.pyfunc.load_model(model_uri)
                        
                        model_info = {
                            "name": "latest_best_model",
                            "version": "from_run",
                            "stage": "None",
                            "run_id": best_run.info.run_id,
                            "accuracy": best_run.data.metrics.get("test_accuracy", 0)
                        }
                        
                        print(f"Modelo cargado desde mejor run: {model_info}")
                        model_loaded = True
        
        return model_loaded
        
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Evento de inicio: cargar modelo"""
    print("Iniciando Inference API...")
    if not load_model():
        print("Advertencia: No se pudo cargar el modelo al inicio")

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "model_loaded": loaded_model is not None,
        "model_info": model_info if loaded_model else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint de predicción"""
    if loaded_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. Intente recargar."
        )
    
    try:
        # Convertir datos a DataFrame
        df = pd.DataFrame([request.data])
        
        # Realizar predicción
        prediction = loaded_model.predict(df)
        
        return PredictionResponse(
            prediction=prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            model_name=model_info.get("name", "unknown"),
            model_version=model_info.get("version", "unknown")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )

@app.post("/reload-model")
async def reload_model():
    """Endpoint para recargar el modelo"""
    if load_model():
        return {
            "status": "success",
            "message": "Modelo recargado exitosamente",
            "model_info": model_info
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Error al recargar modelo"
        )

@app.get("/model-info")
async def get_model_info():
    """Endpoint para obtener información del modelo actual"""
    if loaded_model is None:
        return {"status": "no_model_loaded"}
    
    return {
        "status": "loaded",
        "model_info": model_info
    }