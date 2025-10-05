# Imagen base oficial de Airflow
FROM apache/airflow:2.10.2-python3.11

# Cambiar a usuario root para instalaciones
USER root

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    netcat-traditional \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cambiar de nuevo al usuario airflow
USER airflow

# Copiar archivo de requirements
COPY requirements.txt /requirements.txt

ENV TMPDIR=/home/airflow/tmp
RUN mkdir -p $TMPDIR

# Instalar dependencias de Python para el proyecto
RUN pip install --no-cache-dir --prefer-binary -r /requirements.txt

# Configurar el directorio de trabajo
WORKDIR /opt/airflow

# Copiar script de inicialización
COPY --chown=airflow:root entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
