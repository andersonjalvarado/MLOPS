# Proyecto 3: MLOps Diabetes Readmission
### Desarrollado por:
##### Anderson Alvarado Rubio, David Moreno Arias y Juan Diego Peña Mayorga

## Sección 1: Explicacion del proyecto
Este proyecto implementa un sistema completo de MLOps para predecir la readmisión hospitalaria de pacientes con diabetes. Cubre todo el ciclo de vida del modelo: desde la ingesta de datos y limpieza, hasta entrenamientos, registro en MLflow, despliegue como microservicio en Kubernetes y monitorización en tiempo real.

Arquitectura:

- Infraestructura MLOps (fuera de Kubernetes):
  - Airflow: Orquesta pipelines de datos y entrenamiento.
  - MLflow: Seguimiento de experimentos y registro central de modelos.
  - MinIO: Almacenamiento de artefactos compatible con S3.
  - PostgreSQL: Base de datos para Airflow, MLflow y datos crudos/limpios.

- Infraestructura de despliegue (dentro de Kubernetes):
  - API FastAPI: Servicio de inferencia que carga el mejor modelo.
  - UI Streamlit: Interfaz gráfica para interactuar con la API.
  - Observabilidad: Prometheus y Grafana.
  - Pruebas de carga: Locust para evaluar rendimiento.

Flujo de uso y detalle de este README:

1. Instalar herramientas necesarias.
2. Levantar servicios Airflow, MLflow, MinIO, PostgreSQL.
3. Desplegar API y UI en Kubernetes.
4. Ejecutar pipelines de Airflow y registrar modelos en MLflow.
5. Hacer predicciones a través de UI o API.
6. Monitorizar la API y ejecutar pruebas de carga.
7. Apagar y limpiar todo el sistema al finalizar.

## Sección 2: Instalación y preparación del entorno

Antes de levantar el proyecto, necesitamos instalar las herramientas necesarias. Esta sección asume que estás usando Ubuntu 24.04, pero los pasos se pueden adaptar a otras distribuciones.

Herramientas necesarias:

- Docker & Docker Compose
- k3d (Kubernetes ligero en Docker)
- kubectl
- Helm
- Python 3.11+ y pipx

Instalación:

```bash
# Actualizar paquetes e instalar dependencias base
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release docker-compose-plugin pipx

# Inicializar pipx para instalar herramientas de Python
pipx ensurepath

# Iniciar y habilitar Docker
sudo systemctl enable docker
sudo systemctl start docker

# Instalar k3d (Kubernetes dentro de Docker)
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Instalar kubectl
sudo snap install kubectl --classic

# Instalar Helm
sudo snap install helm --classic
```

Preparar proyecto y hosts locales:

```bash
# Ir a la raíz del proyecto
cd /Raiz/Documentos/MLOPS/Proyecto3

# Agregar dominios locales para la API y UI
sudo bash -c 'echo "127.0.0.1   api.localhost ui.localhost" >> /etc/hosts'

# Verificar que Docker está corriendo
sudo systemctl start docker
```

Con esto, tu entorno está listo para levantar todos los servicios del proyecto.  

## Sección 3: Levantar infraestructura MLOps

Esta sección levanta todos los servicios centrales del proyecto que no dependen de Kubernetes: Airflow, MLflow, MinIO y PostgreSQL. Se usan contenedores Docker gestionados por Docker Compose.

Pasos:

```bash
# Ir a la raíz del proyecto
cd /Raiz/Documentos/MLOPS/Proyecto3

# Levantar servicios en segundo plano y construir imágenes
docker-compose up -d --build

# Inicializar la base de datos de Airflow
docker-compose run --rm airflow-webserver db init

# Verificar estado de todos los contenedores
docker-compose ps
```

Notas importantes:

- La primera vez que se inicialice Airflow, algunos contenedores pueden reiniciarse varias veces. Esto es normal, por ende se debe esperar hasta que todos estén en estado `Up` o `Up (healthy)`.
- Airflow estará accesible en `http://localhost:8080`, MLflow en `http://localhost:5000` y MinIO en `http://localhost:9001`.
- Las credenciales por defecto para MinIO y AirFlow son:
  - Usuario: Admin
  - Contraseña: SuperSecret

## Sección 4: Ejecutar pipelines en Airflow y MLflow

Esta sección explica cómo ejecutar los DAGs de Airflow para procesar los datos, entrenar modelos de manera incremental y registrar el mejor modelo en MLflow.

Ejecutar DAG #1: Ingesta de datos crudos

```bash
# Abrir Airflow en el navegador: http://localhost:8080
# Activar y ejecutar manualmente el DAG llamado: 1_ingest_raw_data (Ejecutar una sola vez)
# Este DAG carga el archivo Diabetes.csv en la base de datos RAW_DATA de PostgreSQL
```

Ejecutar DAG #2: Entrenamiento incremental y DAG #3: Entrtenamiento con inicio en caliente.

```bash
# Activar y ejecutar manualmente los DAGs llamados: 2_incremental_training y 3_warm_start_training
# Primera ejecución:
#   - Entrena los Modelos desde cero con el primer lote de 15,000 registros
#   - Registra el modelo en MLflow bajo los experimento: diabetes_incremental_trainin g y diabetes_warm_start_training
```

El proceso de entrenamiento de cada uno de los DAGs 2 y 3 se debe repetir 6 veces (un total de 7 ejecuciones por DAG).  
*Es importante dejar la anterior ejecucion de cada DAG antes de enviar una nueva*


Una vez terminadas las 7 ejecuciones se debe registrar el mejor modelo en MLflow:

```bash
# Abrir MLflow en el navegador: http://localhost:5000
# Entrar al experimento: diabetes_warm_start_training
# Buscar la run con mejor 'batch_accuracy'
# Ir a Artifacts -> Register Model
# Nombre del modelo: diabetes-readmission-predictor
# Cambiar el stage a: Production
```
##### Muy importante si o si que el modelo se llame registrado se llame ****diabetes-readmission-predictor**** 

Notas:

- La API FastAPI desplegada en Kubernetes cargará automáticamente el modelo que esté en Production.
- Se puede verificar que la API está activa con:
```bash
curl -s http://api.localhost:8081"
```

## Sección 5: Despliegue en Kubernetes (API y UI)

En esta sección se despliegan la API FastAPI y la UI Streamlit dentro de un clúster de Kubernetes usando k3d. También se importan las imágenes Docker al clúster y se aplican los manifiestos YAML.

Pasos:

```bash
# Crear un clúster k3d llamado "mlops" y mapear el puerto 8081
k3d cluster create mlops --api-port 6443 -p "8081:80@loadbalancer"

# Configurar kubectl para usar el nuevo clúster
k3d kubeconfig merge mlops --kubeconfig-merge-default

# Verificar que el clúster está listo
kubectl get nodes
```

Construir e importar imágenes Docker:

```bash
# API FastAPI
cd /Raiz/Documentos/MLOPS/Proyecto3/api
docker build -t diabetes-api:latest .
k3d image import diabetes-api:latest -c mlops

# UI Streamlit
cd /Raiz/Documentos/MLOPS/Proyecto3/streamlit_ui
docker build -t diabetes-ui:latest .
k3d image import diabetes-ui:latest -c mlops

# Volver a la raíz del proyecto
cd /Raiz/Documentos/MLOPS/Proyecto3
```

Desplegar aplicaciones en Kubernetes:

```bash
# Desplegar API (Deployment, Service, Ingress)
kubectl apply -f ./api/k8s/

# Desplegar UI (Deployment, Service, Ingress)
kubectl apply -f ./streamlit_ui/k8s/

# Verificar que los pods estén corriendo
kubectl get pods -w
```

Notas importantes:

- Esperar a que todos los pods estén en estado `Running` y `READY 1/1`.
- La UI estará disponible en `http://ui.localhost:8081` y la API en `http://api.localhost:8081`.


### Sección 6: Realizar predicciones

Una vez que el mejor modelo esté en estado Production en MLflow, se pueden hacer predicciones usando la UI Streamlit o directamente la API FastAPI.

1. Predicción usando la UI Streamlit:

```bash
# Abrir en el navegador:
http://ui.localhost:8081

# Completar el formulario con los datos del paciente y haz clic en "Realizar Predicción"
```

2. Predicción usando la API FastAPI con curl:

```bash
curl -X POST "http://api.localhost:8081/predict" \
-H "Content-Type: application/json" \
-d '{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[70-80)",
  "time_in_hospital": 5,
  "num_lab_procedures": 40,
  "num_procedures": 1,
  "num_medications": 15,
  "diag_1": "250.83",
  "diag_2": "401",
  "diag_3": "250.01"
}'
```

Notas:

- La respuesta será un JSON con la predicción de readmisión del paciente.
- Se puede repetir el proceso para diferentes pacientes o datasets.
- La UI y la API están conectadas al mismo modelo en Produccion.

### Sección 7: Monitorización y pruebas de carga

Esta sección permite monitorizar la API en tiempo real y evaluar su rendimiento usando Prometheus, Grafana y Locust.

Desplegar Prometheus y Grafana:

```bash
# Añadir repositorios de Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Crear namespace monitoring si no existe
kubectl create namespace monitoring || echo "Namespace monitoring ya existe"

# Desplegar Prometheus
kubectl apply -f ./monitoring/prometheus.yaml

# Desplegar Grafana
kubectl apply -f ./monitoring/grafana.yaml

```

Verificar que ambos pods estén corriendo:

```bash
kubectl get pods -n monitoring
```

Cuando ambos estén en estado `Running`, hacer un port-forward para acceder a Grafana y Prometeus desde el navegador: (En terminales diferentes)

```bash
kubectl port-forward --address 0.0.0.0 svc/prometheus-service 9090:9090 -n monitoring
kubectl port-forward --address 0.0.0.0 svc/grafana-service 3000:3000 -n monitoring
```

Notas:


- URL: http://localhost:9090  
- Es posible probar y aegurar la conectividad de las APIs (Status/Targets)

- URL: http://localhost:3000  
- Usuario: `Admin`  
- Contraseña: `SuperSecret`
- Configura la fuente de datos Prometheus en Grafana con:
```bash
URL: http://prometheus-server.monitoring.svc.cluster.local
```


Pruebas de carga con Locust:

```bash
# Instalar Locust si no está instalado
pipx install locust || echo "Locust ya instalado"

# Navegar a la carpeta del test
cd /Raiz/Documentos/MLOPS/Proyecto3/locust_test/

# Ejecutar Locust
locust -f locustfile.py 
```

Notas:

- Abrir la UI de Locust: `http://localhost:8089`  
- Introducir el número de usuarios y la tasa de spawn, luego hacer clic en "Start Swarming".  
- Mientras Locust está corriendo, Grafana mostrará métricas de latencia, peticiones por segundo y errores en tiempo real.

### Sección 8: Apagado y limpieza completa

Cuando hayas terminado de usar el sistema, es importante apagar y limpiar todos los servicios y recursos para liberar memoria y espacio.

```bash
# Ir a la raíz del proyecto
cd /Raiz/Documentos/MLOPS/Proyecto3

# 1. Apagar y eliminar el clúster de Kubernetes
k3d cluster delete mlops

# 2. Detener los contenedores Docker Compose (Airflow, MLflow, MinIO, PostgreSQL)
docker-compose down

# 3. Limpieza total de volúmenes persistentes (opcional dado que esto eliminará todas las bases de datos y modelos guardados)
docker-compose down -v
```
