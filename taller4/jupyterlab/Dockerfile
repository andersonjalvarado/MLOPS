# jupyterlab/Dockerfile
FROM jupyter/scipy-notebook:latest

# Instalar las librerías necesarias para el taller
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
