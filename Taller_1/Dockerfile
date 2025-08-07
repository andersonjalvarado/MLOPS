# ==================== STAGE 1: BUILDER ====================
# Usamos una imagen con herramientas de compilación para instalar dependencias
FROM python:3.12-slim as builder

# Metadatos del contenedor
LABEL maintainer="andersonjalvarado@javeriana.edu.co"
LABEL description="Palmer Penguins Classification API"
LABEL version="1.0.0"

# Configurar directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar librerias
# gcc y g++ son necesarios para compilar NumPy, SciPy, scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements.txt primero
COPY requirements.txt .

# Crear wheels de todas las dependencias
# Los wheels son paquetes precompilados que se instalan más rápido
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# ==================== STAGE 2: PRODUCTION ====================
# Imagen final más pequeña sin herramientas de compilación
FROM python:3.12-slim

# Configurar variables de entorno para optimización de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Crear directorio de trabajo
WORKDIR /app

# Instalar solo dependencias de runtime necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar wheels desde el stage builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Crear usuario no-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Crear directorios necesarios
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app

# Copiar código de la aplicación
COPY --chown=appuser:appuser ./app /app/app
COPY --chown=appuser:appuser ./models /app/models

# Cambiar al usuario no-root
USER appuser

# Exponer puerto de la aplicación
EXPOSE 8989

# Healthcheck para verificar que la aplicación esté funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8989/health || exit 1

# Comando para iniciar la aplicación
# Usamos exec form para manejar correctamente las señales del sistema
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8989", "--workers", "1"]