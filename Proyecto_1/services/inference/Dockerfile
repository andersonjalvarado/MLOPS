# Imagen base optimizada
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd -m -u 1000 inference && \
    mkdir -p /app && \
    chown -R inference:inference /app

# Cambiar al usuario no-root
USER inference
WORKDIR /app

# Copiar requirements
COPY --chown=inference:inference requirements.txt .

# Instalar dependencias de Python
RUN pip install --user --no-cache-dir -r requirements.txt

# Agregar binarios de usuario al PATH
ENV PATH="/home/inference/.local/bin:${PATH}"

# Copiar código de la aplicación
COPY --chown=inference:inference . .

# Exponer puerto
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
