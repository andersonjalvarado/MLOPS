"""
Interfaz gráfica con Streamlit para el sistema MLOps de clasificación de cobertura forestal.
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Forest Cover MLOps",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URLs de los servicios
EXTERNAL_API_URL = "http://10.43.100.103:8080"
INFERENCE_API_URL = "http://inference-api:8000"
MLFLOW_URL = "http://mlflow:5000"
AIRFLOW_URL = "http://airflow-webserver:8080"

# Título principal
st.title("🌲 Sistema MLOps - Clasificación de Cobertura Forestal")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Selecciona una opción:",
    ["🏠 Inicio", "📊 Obtener Datos", "🤖 Predicción", "📈 Estado del Sistema"]
)

# ==================== PÁGINA INICIO ====================
if page == "🏠 Inicio":
    st.header("Bienvenido al Sistema MLOps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Información del Sistema")
        st.info("""
        Este sistema de MLOps automatiza el proceso de:
        - Extracción de datos de cobertura forestal
        - Almacenamiento incremental en PostgreSQL
        - Entrenamiento de modelos de clasificación
        - Registro de experimentos en MLflow
        - Inferencia con modelos en producción
        """)
        
        st.subheader("🎯 Características del Dataset")
        st.write("""
        **Variables de entrada:**
        - Elevation, Aspect, Slope
        - Distancias a hidrología, carreteras, puntos de fuego
        - Hillshade (9am, Noon, 3pm)
        - Wilderness Area, Soil Type
        
        **Variable objetivo:**
        - Cover_Type (7 clases de cobertura forestal)
        """)
    
    with col2:
        st.subheader("🔧 Servicios Disponibles")
        
        # Estado de servicios
        services_status = {}
        
        with st.spinner("Verificando servicios..."):
            try:
                r = requests.get(f"{INFERENCE_API_URL}/health", timeout=5)
                services_status["Inference API"] = "🟢 Activo" if r.status_code == 200 else "🔴 Inactivo"
            except:
                services_status["Inference API"] = "🔴 Inactivo"
            
            try:
                r = requests.get(f"{MLFLOW_URL}/health", timeout=5)
                services_status["MLflow"] = "🟢 Activo" if r.status_code == 200 else "🔴 Inactivo"
            except:
                services_status["MLflow"] = "🔴 Inactivo"
            
            try:
                r = requests.get(f"{EXTERNAL_API_URL}/data?group_number=5", timeout=5)
                services_status["API Externa"] = "🟢 Activo" if r.status_code in [200, 400] else "🔴 Inactivo"
            except:
                services_status["API Externa"] = "🔴 Inactivo"
        
        for service, status in services_status.items():
            st.write(f"**{service}:** {status}")
        
        st.markdown("---")
        
        st.subheader("🔗 Enlaces Rápidos")
        st.markdown(f"- [MLflow UI]({MLFLOW_URL})")
        st.markdown(f"- [Airflow UI]({AIRFLOW_URL})")
        st.markdown(f"- [API Docs](http://localhost:8000/docs)")

# ==================== PÁGINA OBTENER DATOS ====================
elif page == "📊 Obtener Datos":
    st.header("📊 Extracción de Datos desde API Externa")
    
    st.write("Consulta datos de la API externa de cobertura forestal.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        group_number = st.number_input("Número de Grupo", min_value=1, max_value=10, value=5)
    
    with col2:
        st.write("")
        st.write("")
        fetch_button = st.button("🔄 Obtener Datos", use_container_width=True)
    
    if fetch_button:
        with st.spinner("Obteniendo datos de la API externa..."):
            try:
                url = f"{EXTERNAL_API_URL}/data"
                params = {"group_number": group_number}
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 400:
                    st.error("❌ Error 400: Se alcanzó el número mínimo de muestras")
                    st.json(response.json())
                elif response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"✅ Datos obtenidos exitosamente del Batch #{data['batch_number']}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Grupo", data['group_number'])
                    col2.metric("Batch", data['batch_number'])
                    col3.metric("Registros", len(data['data']))
                    
                    # Mostrar datos en tabla
                    st.subheader("Vista de Datos")
                    
                    columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                              'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                              'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                              'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 
                              'Soil_Type', 'Cover_Type']
                    
                    df = pd.DataFrame(data['data'], columns=columns)
                    st.dataframe(df.head(100), use_container_width=True, height=400)
                    
                    # Estadísticas
                    st.subheader("Estadísticas Básicas")
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("❌ Error: Timeout al conectar con la API externa")
            except requests.exceptions.ConnectionError:
                st.error("❌ Error: No se pudo conectar con la API externa")
            except Exception as e:
                st.error(f"❌ Error inesperado: {str(e)}")

# ==================== PÁGINA PREDICCIÓN ====================
elif page == "🤖 Predicción":
    st.header("🤖 Predicción de Cobertura Forestal")
    
    tab1, tab2 = st.tabs(["Predicción Manual", "Información del Modelo"])
    
    with tab1:
        st.subheader("Ingresa los valores para predicción")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            elevation = st.number_input("Elevation", min_value=0, max_value=5000, value=2500)
            aspect = st.number_input("Aspect", min_value=0, max_value=360, value=180)
            slope = st.number_input("Slope", min_value=0, max_value=90, value=15)
            h_dist_hydro = st.number_input("Horizontal Distance to Hydrology", 
                                          min_value=0, max_value=2000, value=200)
        
        with col2:
            v_dist_hydro = st.number_input("Vertical Distance to Hydrology", 
                                          min_value=-500, max_value=500, value=0)
            h_dist_road = st.number_input("Horizontal Distance to Roadways", 
                                         min_value=0, max_value=10000, value=1000)
            hillshade_9am = st.number_input("Hillshade 9am", min_value=0, max_value=255, value=200)
            hillshade_noon = st.number_input("Hillshade Noon", min_value=0, max_value=255, value=220)
        
        with col3:
            hillshade_3pm = st.number_input("Hillshade 3pm", min_value=0, max_value=255, value=150)
            h_dist_fire = st.number_input("Horizontal Distance to Fire Points", 
                                         min_value=0, max_value=10000, value=1500)
            wilderness = st.selectbox("Wilderness Area", 
                                     ["Rawah", "Neota", "Commanche", "Cache"])
            soil = st.text_input("Soil Type", value="C2702")
        
        if st.button("🎯 Realizar Predicción", use_container_width=True):
            with st.spinner("Realizando predicción..."):
                try:
                    # Preparar datos
                    from sklearn.preprocessing import LabelEncoder
                    
                    # Encoders (valores de ejemplo - en producción deberían ser los mismos del entrenamiento)
                    wilderness_map = {"Rawah": 0, "Neota": 1, "Commanche": 2, "Cache": 3}
                    
                    # Crear diccionario de features
                    prediction_data = {
                        "Elevation": elevation,
                        "Aspect": aspect,
                        "Slope": slope,
                        "Horizontal_Distance_To_Hydrology": h_dist_hydro,
                        "Vertical_Distance_To_Hydrology": v_dist_hydro,
                        "Horizontal_Distance_To_Roadways": h_dist_road,
                        "Hillshade_9am": hillshade_9am,
                        "Hillshade_Noon": hillshade_noon,
                        "Hillshade_3pm": hillshade_3pm,
                        "Horizontal_Distance_To_Fire_Points": h_dist_fire,
                        "Wilderness_Area_Encoded": wilderness_map.get(wilderness, 0),
                        "Soil_Type_Encoded": hash(soil) % 100  # Simplificación
                    }
                    
                    # Hacer predicción
                    response = requests.post(
                        f"{INFERENCE_API_URL}/predict",
                        json={"data": prediction_data},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Predicción realizada exitosamente")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Tipo de Cobertura Predicho", 
                                    f"Clase {result['prediction'][0]}")
                        
                        with col2:
                            st.info(f"**Modelo:** {result['model_name']}\n\n"
                                  f"**Versión:** {result['model_version']}")
                        
                    elif response.status_code == 503:
                        st.error("❌ Modelo no disponible. Intenta recargar el modelo.")
                    else:
                        st.error(f"❌ Error en predicción: {response.json()['detail']}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ No se pudo conectar con el servicio de Inference")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.subheader("Información del Modelo Actual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Obtener Info del Modelo"):
                try:
                    response = requests.get(f"{INFERENCE_API_URL}/model-info", timeout=5)
                    
                    if response.status_code == 200:
                        info = response.json()
                        
                        if info['status'] == 'loaded':
                            st.success("✅ Modelo cargado")
                            st.json(info['model_info'])
                        else:
                            st.warning("⚠️ No hay modelo cargado")
                    else:
                        st.error("❌ Error al obtener información")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("♻️ Recargar Modelo"):
                with st.spinner("Recargando modelo..."):
                    try:
                        response = requests.post(f"{INFERENCE_API_URL}/reload-model", timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            st.json(result['model_info'])
                        else:
                            st.error(f"❌ Error: {response.json()['detail']}")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# ==================== PÁGINA ESTADO DEL SISTEMA ====================
elif page == "📈 Estado del Sistema":
    st.header("📈 Estado del Sistema MLOps")
    
    st.subheader("Servicios")
    
    # Matriz de estado de servicios
    services = {
        "Inference API": INFERENCE_API_URL + "/health",
        "MLflow": MLFLOW_URL + "/health",
        "API Externa (Producción)": EXTERNAL_API_URL + "/data?group_number=5"
    }
    
    status_data = []
    
    for service_name, url in services.items():
        try:
            start_time = time.time()
            response = requests.get(url, timeout=5)
            latency = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code in [200, 400]:
                status = "🟢 Activo"
                status_code = response.status_code
            else:
                status = "🟡 Degradado"
                status_code = response.status_code
        except:
            status = "🔴 Inactivo"
            status_code = "N/A"
            latency = "N/A"
        
        status_data.append({
            "Servicio": service_name,
            "Estado": status,
            "Código": status_code,
            "Latencia (ms)": latency
        })
    
    df_status = pd.DataFrame(status_data)
    st.dataframe(df_status, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Información del modelo actual
    st.subheader("Modelo en Producción")
    
    try:
        response = requests.get(f"{INFERENCE_API_URL}/model-info", timeout=5)
        
        if response.status_code == 200:
            info = response.json()
            
            if info['status'] == 'loaded':
                col1, col2, col3 = st.columns(3)
                
                model_info = info['model_info']
                
                col1.metric("Nombre", model_info.get('name', 'N/A'))
                col2.metric("Versión", model_info.get('version', 'N/A'))
                col3.metric("Stage", model_info.get('stage', 'N/A'))
                
                st.info(f"**Run ID:** {model_info.get('run_id', 'N/A')}")
            else:
                st.warning("⚠️ No hay modelo cargado en el servicio de Inference")
        else:
            st.error("❌ No se pudo obtener información del modelo")
            
    except Exception as e:
        st.error(f"❌ Error al consultar modelo: {str(e)}")
    
    st.markdown("---")
    
    # URLs de acceso
    st.subheader("Enlaces de Administración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### MLflow")
        st.markdown(f"🔗 [Abrir MLflow UI]({MLFLOW_URL})")
        st.caption("Tracking de experimentos y Model Registry")
    
    with col2:
        st.markdown("### Airflow")
        st.markdown(f"🔗 [Abrir Airflow UI]({AIRFLOW_URL})")
        st.caption("Orquestación de pipelines")

# Footer
st.markdown("---")
st.caption("Sistema MLOps - Clasificación de Cobertura Forestal | Octubre 2025")