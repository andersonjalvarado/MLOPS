# api/api.py
from flask import Flask, request, jsonify
import mlflow
import os
import pandas as pd 

# Configuración para conectar a MLflow y MinIO
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000' # o la IP del host
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000' # o la IP del host
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

app = Flask(__name__)

# Cargar el modelo desde el registro de MLflow
# Lo cargamos al inicio para que esté listo para las predicciones
model_name = "PenguinClassifierModel"
model_stage = "None" # Puedes usar "Staging" o "Production" si lo has transicionado en la UI

try:
    print(f"Cargando modelo '{model_name}' en etapa '{model_stage}'...")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no pudo ser cargado."}), 500

    # Obtener datos del request
    data = request.json
    
    # El modelo espera un DataFrame de pandas
    try:
        # ejemplo de payload: {"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181.0, "body_mass_g": 3750.0}
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)
        
        return jsonify({"species_prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
