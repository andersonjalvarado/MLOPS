# Informe de implementación y retos (por etapas)

## 0) Objetivo y alcance
Pipeline ML de extremo a extremo para clasificar pingüinos con Airflow (orquestación, metadatos Postgres), MySQL (datos), FastAPI (inferencia y UI), empaquetado con Docker Compose, explicabilidad (k‑NN + PCA) y artefactos compartidos en `artifacts/`.

---
## 1) Scaffolding y contenedores
- Issues:
  - Comando de creación de usuario en Airflow fallaba (`exit 127`).
- Causa:
  - Comillas/escapes en el `docker-compose.yml` al usar comandos multiline.
- Fix:
  - Unificar a una sola línea: `bash -c "airflow db migrate && airflow users create ..."`.

---
## 2) Dependencias de Airflow y base de imagen
- Issues:
  - ImportErrors (`airflow.decorators`, `pandas`, `sqlalchemy`, `pyarrow`).
  - Mensajes de NumPy 2.x incompatible con binarios compilados (pyarrow/pandas).
  - Error SQLAlchemy en migración: `executemany_mode: 'values'`.
- Causas:
  - Desalineación con pines de la imagen oficial; NumPy 2 rompe binarios.
- Fixes:
  - Imagen custom basada en `apache/airflow:2.8.3-python3.11`.
  - `airflow/requirements.txt`: `SQLAlchemy==1.4.52`, `numpy<2.0` y deps requeridas.
  - Evitar sobreescribir dependencias pinneadas por Airflow.

---
## 3) Ingesta de datos (dataset)
- Issues:
  - `seaborn.load_dataset("penguins")` fallaba por red (`Connection reset by peer`).
- Causa:
  - Restricciones/inspección TLS hacia `raw.githubusercontent.com` en algunas redes.
- Fix (definitivo):
  - Usar `palmerpenguins` (PyPI) como fuente principal y fallback a `seaborn` si está disponible.
  - Sin credenciales ni Kaggle; sin dependencia de internet en runtime.

---
## 4) Persistencia MySQL y NaN
- Issues:
  - `pymysql.err.ProgrammingError: nan can not be used with MySQL`.
  - Problemas con `pandas.to_sql`/`read_sql` (`cursor`/`TypeError`).
- Causas:
  - MySQL no acepta NaN; rutas internas de pandas/DBAPI inestables en runtime de Airflow.
- Fixes:
  - Limpieza con `df.dropna(inplace=True)` antes de escribir.
  - Escritura robusta con SQLAlchemy Core (`write_dataframe`: define tabla + `insert()` bulk).
  - Lectura robusta `read_sql_robustly` (SQLAlchemy → DataFrame) evitando `pd.read_sql` directo.

---
## 5) Entrenamiento y artefactos
- Issues:
  - API fallaba al arrancar sin artefactos; luego `ModuleNotFoundError: dill` al cargar.
- Causas:
  - Carga en `startup` (acoplada al estado); `joblib` puede requerir `dill`.
- Fixes:
  - Carga perezosa en `/predict` + endpoint `/warmup`.
  - DAG `train_pipeline` llama a `/warmup` al finalizar.
  - Añadir `dill` a `api/requirements.txt`.

---
## 6) Explicabilidad y UX
- Objetivo:
  - Dar contexto y confianza al usuario.
- Solución:
  - k‑NN sobre datos de entrenamiento para 5 vecinos + confianza (proporción que coincide con la predicción).
  - PCA (2 componentes) entrenado en `train_pipeline`; UI con Chart.js mostrando distribución, vecinos resaltados y el nuevo punto.

---
## 7) Permisos en volumen `artifacts/`
- Issues:
  - `PermissionError: /opt/artifacts/*.joblib` en entrenamientos.
- Causa:
  - UID de contenedor (50000) vs. owner del host.
- Fixes:
  - Opción rápida: `sudo chown -R 50000:0 artifacts && sudo chmod -R 775 artifacts`.
  - Alternativa: `.env` con `AIRFLOW_UID=$(id -u)` para alinear UID de Airflow con el host.

---
## 8) Compose y buenas prácticas
- Decisiones:
  - Hornear dependencias en imágenes (no `PIP_ADDITIONAL_REQUIREMENTS`).
  - Variables de entorno compartidas (`DATA_DB_*`, `ARTIFACTS_DIR`).
  - Separación de DAGs por responsabilidad: `data_pipeline` y `train_pipeline`.
  - Volumen `artifacts/` como bus de artefactos entre Airflow y API.

---
## 9) Git y versionado
- Issues:
  - Artefactos `.joblib` quedaron versionados inicialmente.
- Fix:
  - `.gitignore` para `artifacts/*` (manteniendo `.gitkeep`) y `__pycache__/`.
  - Limpieza con `git rm --cached` y convención de no versionar generados.

---
## 10) Despliegue en VM
- Acciones/claves:
  - `git reset --hard origin/main` para alinear el repo.
  - Rebuild de imágenes cuando cambian deps (`docker compose build airflow-*`).
  - Test de red desde contenedores (`exec -T ... python -c/urlopen`).
  - Ajuste de permisos de `artifacts/` si se requiere.

---
## Lecciones aprendidas
- Alinear versiones con la imagen base de Airflow y fijar pines críticos (SQLAlchemy/NumPy).
- Preferir SQLAlchemy Core cuando pandas↔DBAPI resulta frágil en runtime.
- Evitar dependencias de internet en ejecución (palmerpenguins > seaborn remoto).
- Diseñar APIs resilientes al estado del modelo (lazy load + warmup) y pipelines que automaticen preparación.
- Gestionar permisos de volúmenes y/o usar `AIRFLOW_UID`.

---
## Operación resumida
1. `docker compose up -d --build` (y permisos de `artifacts/` si aplica).
2. Airflow UI → `data_pipeline` → `train_pipeline` (warmup automático).
3. UI/API en `http://localhost:8000`.

Para detalles de inicio, permisos y resolución de problemas ver `README.md` (Puesta en marcha desde 0, Troubleshooting).
