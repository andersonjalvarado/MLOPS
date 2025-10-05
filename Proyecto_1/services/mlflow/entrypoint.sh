#!/bin/bash
# set -e

# # Esperar a que MySQL esté disponible
# echo "Esperando a que MySQL esté disponible..."
# until nc -z -v -w30 ${DB_HOST} ${DB_PORT}; do
#   echo "Esperando MySQL en ${DB_HOST}:${DB_PORT}..."
#   sleep 2
# done

# echo "MySQL está disponible. Iniciando MLflow..."

# # Ejecutar el comando proporcionado
# exec "$@"


#!/bin/bash
set -e

# Esperar a que MySQL esté disponible
echo "Esperando a que MySQL esté disponible..."
until nc -z -v -w30 ${DB_HOST} ${DB_PORT}; do
  echo "Esperando MySQL en ${DB_HOST}:${DB_PORT}..."
  sleep 2
done

echo "MySQL está disponible. Iniciando MLflow..."

# Construir URI dinámicamente con las variables de entorno
BACKEND_URI="mysql+pymysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "$BACKEND_URI" \
    --default-artifact-root "s3://mlflows3/artifacts" \
    --serve-artifacts
