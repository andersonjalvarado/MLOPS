# Imagen base optimizada con Python 3.11
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 mlflow && \
    mkdir -p /app/mlflowdb && \
    chown -R mlflow:mlflow /app

RUN apt-get update && apt-get install -y --no-install-recommends netcat-traditional && rm -rf /var/lib/apt/lists/*

# Cambiar al usuario no-root
USER mlflow
WORKDIR /app

# Actualizar pip y instalar dependencias de Python
RUN pip install --user --no-cache-dir \
    mlflow==2.16.2 \
    boto3==1.35.21 \
    pymysql==1.1.1 \
    cryptography==43.0.1 \
    psycopg2-binary==2.9.9

# Agregar binarios de usuario al PATH
ENV PATH="/home/mlflow/.local/bin:${PATH}"

# Exponer puerto del servidor MLflow
EXPOSE 5000

# Script de entrada para esperar MySQL
COPY --chown=mlflow:mlflow entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Comando de inicio
ENTRYPOINT ["/app/entrypoint.sh"]
# CMD ["mlflow", "server", \
#      "--host", "0.0.0.0", \
#      "--port", "5000", \
#      "--backend-store-uri", "mysql+pymysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}", \
#      "--default-artifact-root", "s3://mlflows3/artifacts", \
#      "--serve-artifacts"]
