#!/bin/bash
set -e

# Extraer componentes de la conexión a la base de datos
DB_HOST=$(echo $AIRFLOW__DATABASE__SQL_ALCHEMY_CONN | sed -n 's/.*@\([^:]*\):.*/\1/p')
DB_PORT=$(echo $AIRFLOW__DATABASE__SQL_ALCHEMY_CONN | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

# Esperar a que PostgreSQL esté disponible
echo "Esperando PostgreSQL en ${DB_HOST}:${DB_PORT}..."
while ! nc -z ${DB_HOST} ${DB_PORT}; do
  sleep 2
done
echo "PostgreSQL está disponible!"

# Esperar a que Redis esté disponible (para Celery)
REDIS_HOST=$(echo $AIRFLOW__CELERY__BROKER_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
REDIS_PORT=$(echo $AIRFLOW__CELERY__BROKER_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

echo "Esperando Redis en ${REDIS_HOST}:${REDIS_PORT}..."
while ! nc -z ${REDIS_HOST} ${REDIS_PORT}; do
  sleep 2
done
echo "Redis está disponible!"

# Inicializar la base de datos de Airflow (solo para webserver)
if [ "$1" = "webserver" ]; then
  echo "Inicializando base de datos de Airflow..."
  airflow db migrate
  
  # Crear usuario admin si no existe
  airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123 || true
fi

# Ejecutar el comando de Airflow
exec airflow "$@"
