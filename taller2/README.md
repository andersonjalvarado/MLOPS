# Taller: Desarrollo en Contenedores

## Descripción
En este taller configuramos un entorno de desarrollo usando Docker.
La idea es tener dos servicios funcionando en paralelo:

Un notebook de Jupyter para entrenar modelos.
Una API que pueda usar esos modelos para hacer predicciones.

Ambos servicios comparten la carpeta donde se guardan los modelos, así que lo que se entrena en Jupyter está disponible en la API.

## Estructura del proyecto
├── docker-compose.yml   # Archivo para levantar los servicios
├── api/                 # Carpeta con el código de la API
│   ├── Dockerfile
│   └── main.py
├── jupyter/             # Carpeta con el entorno Jupyter
│   ├── Dockerfile
│   └── notebooks/
│       └── UsarModelo.ipynb
├── sharedmodels/        # Carpeta compartida para modelos
│   └── penguin_model.pkl

## Requisitos previos
- Tener Docker instalado.  
- Tener Docker Compose instalado.

Verificar la instalacion o componentes con:
docker --version
docker-compose --version

## Ejecución
1. Construir y levantar los servicios:
   docker-compose up --build

2. Servicios disponibles:
   - **API**: `http://localhost:8000`  
     - Endpoint raíz: `GET /`  
     - Predicciones: `POST /predict` con JSON de entrada.  

     Ejemplo de request:
     curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"X": [5.1, 3.5, 1.4, 0.2, 0.5, 1, 0]}'

   - **Jupyter Notebook**: `http://localhost:8888`  
     (El token de acceso se muestra en consola al iniciar el contenedor).  

3. Para detener todo:
   docker-compose down

## Uso
- En el notebook puedes entrenar modelos y guardarlos en sharedmodels.
- La API los lee directamente para responder a las solicitudes de predicción. 

Ejemplo en notebook:

import pickle

with open("../sharedmodels/penguin_model.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict([[5.1, 3.5, 1.4, 0.2, 0.5, 1, 0]])
print(pred)

## Conclusiones
- Con Docker logramos un entorno fácil de reproducir y portable.
- Separar servicios (notebooks y API) hace que el flujo de trabajo sea más organizado.
- El modelo entrenado puede usarse tanto desde el notebook como desde la API, ideal para proyectos de machine learning que luego pasan a producción. 
