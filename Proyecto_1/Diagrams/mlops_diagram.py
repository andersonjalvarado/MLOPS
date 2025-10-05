from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.container import Docker
from diagrams.onprem.database import Mysql, PostgreSQL
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.mlops import Mlflow
from diagrams.programming.framework import Fastapi
from diagrams.custom import Custom

# Configuración del grafo para simular ambiente distribuido en local
graph_attr = {
    "layout": "dot",
    "compound": "true",
    "splines": "ortho",
    "pad": "0.5",
    "nodesep": "0.8",
    "ranksep": "1.0"
}

with Diagram(name="MLOps Architecture - Simulated Distributed Environment", 
             direction='LR', 
             show=False, 
             graph_attr=graph_attr,
             filename="mlops_architecture"):

    # Simular API Externa (10.43.100.103:8080)
    with Cluster("External Production API\n(10.43.100.103:8080)", 
                 direction='TB', 
                 graph_attr={"bgcolor": "#E8F4F8", "style": "dashed"}):
        external_api = Fastapi("Data Provider API\n(Batches cada 5 min)")

    # Máquina Local - Ambiente MLOps
    with Cluster("Local Machine - MLOps Environment\n(Docker Compose)", 
                 direction='TB',
                 graph_attr={"bgcolor": "#F5F5F5"}):

        # Capa de Datos
        with Cluster("Data Layer", 
                     direction='LR', 
                     graph_attr={"bgcolor": "#FFE5CC"}):
            
            with Cluster("PostgreSQL\n(Batch Storage)", 
                         graph_attr={"bgcolor": "#D0E8F2"}):
                postgres_batches = PostgreSQL("batches_db\n(Historical Data)")
            
            with Cluster("MySQL\n(MLflow Metadata)", 
                         graph_attr={"bgcolor": "#FFD4D4"}):
                mysql_metadata = Mysql("mlflow_db\n(Experiments & Runs)")
            
            with Cluster("MinIO\n(Object Storage)", 
                         graph_attr={"bgcolor": "#E0D4FF"}):
                minio_storage = Custom("Artifact Store\n(Models & Files)", 
                                      "./images/MINIO_wordmark.png")

        # Capa de MLOps
        with Cluster("MLOps Layer", 
                     direction='TB', 
                     graph_attr={"bgcolor": "#D4F4DD"}):
            
            with Cluster("Airflow\n(Orchestration)", 
                         graph_attr={"bgcolor": "#B8E6D5"}):
                airflow_scheduler = Airflow("Scheduler\n(DAGs)")
            
            with Cluster("MLflow\n(Tracking & Registry)", 
                         graph_attr={"bgcolor": "#F9B7D0"}):
                mlflow_tracking = Mlflow("Tracking Server")
                mlflow_registry = Mlflow("Model Registry")

        # Capa de Inference
        with Cluster("Inference Layer", 
                     direction='TB', 
                     graph_attr={"bgcolor": "#FFF4CC"}):
            inference_api = Fastapi("Inference API\n(Model Serving)")
        
        # Capa de UI
        with Cluster("User Interface", 
                     direction='TB', 
                     graph_attr={"bgcolor": "#E8D4F8"}):
            streamlit_ui = Custom("Streamlit UI\n(Port 8503)", 
                                 "./images/streamlit.png") if False else Fastapi("Streamlit UI")

    # Flujo de datos desde API Externa
    external_api >> Edge(label="GET /batch/5\n(every 30s)", 
                         color="#2E86AB", 
                         style="bold") >> airflow_scheduler

    # Flujo de Airflow
    airflow_scheduler >> Edge(label="Store batches", 
                              color="#A23B72") >> postgres_batches
    
    airflow_scheduler >> Edge(label="Train & Log", 
                              color="#F18F01") >> mlflow_tracking

    # Flujo de MLflow
    mlflow_tracking >> Edge(label="Metadata", 
                            color="#C73E1D") >> mysql_metadata
    
    mlflow_tracking >> Edge(label="Register Model", 
                            color="#6A4C93") >> mlflow_registry
    
    mlflow_registry >> Edge(label="Store Artifacts", 
                            color="#8338EC") >> minio_storage
    
    mlflow_tracking >> Edge(label="Artifacts", 
                            color="#8338EC", 
                            style="dashed") >> minio_storage

    # Flujo de Inference
    postgres_batches >> Edge(label="Load Training Data", 
                             color="#06A77D", 
                             style="dotted") >> mlflow_tracking
    
    mlflow_registry >> Edge(label="Load Model", 
                            color="#FF006E") >> inference_api
    
    # Flujo de UI
    streamlit_ui >> Edge(label="Get Data", 
                        color="#9D4EDD", 
                        style="dashed") >> external_api
    
    streamlit_ui >> Edge(label="Predict", 
                        color="#3A0CA3") >> inference_api
    
    streamlit_ui >> Edge(label="View Experiments", 
                        color="#F72585", 
                        style="dotted") >> mlflow_registry