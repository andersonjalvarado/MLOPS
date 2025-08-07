# Taller 1 - MLOPS

Utilice la librería [palmerpenguins](https://pypi.org/project/palmerpenguins/) para descargar los datos.

- Cree un archivo en Python consuma estos datos y realice las dos etapas, procesamiento de datos y creación de modelo. Considere usar como guía las sub-etapas listadas.
- Cree un API usando FastAPI permita hacer inferencia al modelo entrenado.
- Cree la imagen del contenedor con el API creada. Exponga el API en puerto 8989.

![nivel 0](img/lvl0.svg)

Bono.

El proceso de entrenamiento de un modelo busca encontrar el mejor modelo y ajustarlo a los datos, este proceso de experimentación en ocasiones resulta en multiples modelos con muy buenos resultados. Como bono entregue en el API un método adicional que permita seleccionar cual modelo será usado en el proceso de inferencia.


# Solución
## Descripción del Proyecto

Este es un proyecto MLOps de **Clasificación de Especies de Pingüinos de Palmer** que implementa un pipeline completo de aprendizaje automático, desde el procesamiento de datos hasta el despliegue de una API. El proyecto clasifica especies de pingüinos (Adelie, Chinstrap, Gentoo) en función de características físicas utilizando el conjunto de datos Palmer Penguins.

## Comandos Clave

### Flujo de Trabajo de Desarrollo

- **Configurar entorno**: `./run.sh setup` (crea venv, instala dependencias)
    
- **Entrenar modelo**: `./run.sh train` (usa config.yaml, guarda en models/)
    
- **Ejecutar API localmente**: `./run.sh api` (inicia FastAPI en el puerto 8989)
    
- **Ver estado del proyecto**: `./run.sh status`
    

### Despliegue con Docker

- **Construir contenedor**: `./run.sh docker-build`
    
- **Ejecutar contenedor**: `./run.sh docker-start` (expone la API en el puerto 8989)
    

### Comandos Directos de Python (con venv activado)

- **Entrenar con configuración personalizada**: `python train.py --config config.yaml`
    
- **Ejecutar API**: `cd app && python -m uvicorn main:app --host 0.0.0.0 --port 8989 --reload`
    

## Descripción de la Arquitectura

### Componentes Principales

1. **Procesamiento de Datos** (`train.py:DataProcessor`): Carga, limpia y realiza ingeniería de características sobre el conjunto de datos Palmer Penguins
    
2. **Entrenamiento del Modelo** (`train.py:ModelTrainer`): Soporta regresión logística, bosque aleatorio y SVM con ajuste de hiperparámetros
    
3. **Pipeline de ML** (`train.py:MLPipeline`): Orquesta todo el flujo de trabajo de entrenamiento
    
4. **Aplicación FastAPI** (`app/main.py`): API lista para producción con validación, manejo de errores y monitoreo
    

### Sistema de Configuración

- **Configuración principal**: `config.yaml` - Controla tipo de modelo, características, ajuste de hiperparámetros y opciones de salida
    
- **Tipos de modelo**: `logistic`, `random_forest`, `svm`
    
- **Ingeniería de características**: Crea automáticamente `bill_ratio`, `body_mass_kg`, `mass_flipper_ratio`
    

### Pipeline de Datos

- Carga el conjunto de datos Palmer Penguins usando la librería `palmerpenguins`
    
- Maneja valores faltantes con estrategias apropiadas (mediana para numéricos, moda para categóricos)
    
- Aplica estandarización a características numéricas y codificación one-hot a categóricas
    
- Soporta división de entrenamiento/prueba con estratificación
    

### Gestión de Modelos

- Los modelos se guardan como `penguin_model_latest.joblib` para uso de la API
    
- Los metadatos se guardan como JSON con métricas de entrenamiento, configuración y marcas de tiempo
    
- Soporta versionamiento de modelos con nombres de archivo con timestamp
    

### Funcionalidades de la API

- **Predicción individual**: `POST /predict` con validación de `PenguinFeatures`
    
- **Predicción por lotes**: `POST /predict/batch` (máximo 100 pingüinos)
    
- **Verificación de estado**: `GET /health` para monitoreo
    
- **Información del modelo**: `GET /model/info` para metadatos del modelo
    
- **Validación completa**: Modelos Pydantic con validadores personalizados para islas y sexo
    

## Detalles Importantes de Implementación

### Consistencia en Ingeniería de Características

La API (`app/main.py:_prepare_input_data`) debe replicar exactamente la ingeniería de características del entrenamiento (`train.py:_create_features`):

- `bill_ratio = bill_length_mm / bill_depth_mm`
    
- `body_mass_kg = body_mass_g / 1000`
    
- `mass_flipper_ratio = body_mass_g / flipper_length_mm`
    

### Estructura del Pipeline del Modelo

Los modelos se guardan como objetos completos de Pipeline de sklearn que contienen:

1. Transformador de columnas con preprocesamiento separado para numéricos y categóricos
    
2. El clasificador entrenado (logistic/random_forest/svm)
    

### Configuración de Docker

- Build multi-etapa para tamaño de imagen optimizado
    
- Ejecución como usuario no-root por seguridad
    
- Integración del endpoint de verificación de estado
    
- Montaje de volúmenes para modelos y registros
    

## Estructura de Directorios

```
├── app/                   # Aplicación FastAPI
│   ├── main.py            # Aplicación principal de la API
├── models/                # Modelos entrenados y metadatos
├── logs/                  # Registros del entrenamiento y la API  
├── train.py               # Pipeline de entrenamiento de ML
├── config.yaml            # Configuración de entrenamiento
├── requirements.txt       # Dependencias de Python
├── Dockerfile             # Definición del contenedor
└── run.sh                 # Script de automatización
```

## Notas de Desarrollo

- El proyecto utiliza una arquitectura modular basada en clases para facilitar el mantenimiento
    
- Se implementa logging completo durante el entrenamiento y en la API
    
- Todos los modelos soportan `predict_proba()` para puntajes de confianza
    
- La API incluye middleware CORS y manejo de errores completo
    
- Los rangos de validación de características se basan en las características del conjunto de datos Palmer Penguins
    
- El ajuste de hiperparámetros usa GridSearchCV con validación cruzada de 5 pliegues
    
