#!/bin/bash

# Script para automatizar la instalación y ejecución del proyecto
# Palmer Penguins Classifier
# Versión simplificada sin Docker Compose para desarrollo inicial

set -e  # Salir si cualquier comando falla

# Colores para output más claro - esto mejora la experiencia del desarrollador
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funciones para imprimir mensajes con colores
# Estas funciones hacen que el output sea más legible y profesional
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Función para mostrar ayuda - simplificada para esta versión
show_help() {
    echo "Palmer Penguins Classifier - Script de automatización"
    echo ""
    echo "Uso: ./run.sh [COMMAND]"
    echo ""
    echo "Comandos disponibles:"
    echo "  setup        - Instalar dependencias y crear entorno virtual"
    echo "  train        - Entrenar el modelo de ML"
    echo "  api          - Ejecutar la API en modo desarrollo"
    echo "  docker-build - Construir imagen de Docker"
    echo "  docker-start - Ejecutar contenedor Docker individual"
    echo "  clean        - Limpiar archivos temporales"
    echo "  help         - Mostrar esta ayuda"
    echo ""
    echo "Flujo recomendado para desarrollo:"
    echo "  1. ./run.sh setup      # Configurar entorno"
    echo "  2. ./run.sh train      # Entrenar modelo"
    echo "  3. ./run.sh api        # Probar API localmente"
    echo ""
    echo "Para deployment con Docker:"
    echo "  1. ./run.sh docker-build   # Construir imagen"
    echo "  2. ./run.sh docker-start   # Ejecutar contenedor"
    echo ""
}

# Función para setup inicial
setup() {
    print_status "Configurando entorno de desarrollo..."
    
    # Verificar Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 no está instalado"
        print_status "Por favor instala Python 3.8 o superior"
        exit 1
    fi
    
    # Mostrar versión de Python para debugging
    python_version=$(python3 --version)
    print_status "Usando $python_version"
    
    # Crear entorno virtual si no existe
    if [ ! -d "venv" ]; then
        print_status "Creando entorno virtual..."
        python3 -m venv venv
    else
        print_status "Entorno virtual ya existe, reutilizando..."
    fi
    
    # Activar entorno virtual
    source venv/bin/activate
    
    # Actualizar pip - Evitar problemas de compatibilidad
    print_status "Actualizando pip..."
    pip install --upgrade pip
    
    # Instalar dependencias
    print_status "Instalando dependencias de requirements.txt..."
    pip install -r requirements.txt
    
    # Crear directorios necesarios
    mkdir -p models logs
    
    print_success "Setup completado exitosamente"
    print_status "Para activar el entorno virtual manualmente: source venv/bin/activate"
}

# Función para entrenar modelo
train() {
    print_status "Iniciando entrenamiento del modelo..."
    
    # Validar que el setup se haya ejecutado
    if [ ! -d "venv" ]; then
        print_error "Entorno virtual no encontrado"
        print_status "Ejecuta primero: ./run.sh setup"
        exit 1
    fi
    
    # Validar que el archivo de configuración existe
    if [ ! -f "config.yaml" ]; then
        print_error "Archivo config.yaml no encontrado"
        print_status "Asegúrate de que config.yaml esté en el directorio raíz"
        exit 1
    fi
    
    source venv/bin/activate
    
    print_status "Usando configuración: config.yaml"
    python train.py --config config.yaml
    
    # Verificar que el modelo se haya guardado correctamente
    if [ -f "models/active_model.json" ]; then
        print_success "Entrenamiento completado - Modelo guardado exitosamente"
        print_status "Modelo disponible en: models/active_model.json"
    else
        print_error "El entrenamiento falló - No se encontró el modelo generado"
        exit 1
    fi
}

# Función para ejecutar API en desarrollo
api() {
    print_status "Iniciando API en modo desarrollo..."
    
    # Validar entorno virtual
    if [ ! -d "venv" ]; then
        print_error "Entorno virtual no encontrado. Ejecuta primero: ./run.sh setup"
        exit 1
    fi
    
    source venv/bin/activate
    
    # Verificar que existe un modelo entrenado
    if [ ! -f "models/active_model.json" ]; then
        print_warning "No se encontró modelo entrenado"
        print_status "Ejecutando entrenamiento automáticamente..."
        train
    fi
    
    # Verificar que el directorio de la app existe
    if [ ! -d "app" ]; then
        print_error "Directorio 'app' no encontrado"
        print_status "Asegúrate de estar en el directorio raíz del proyecto"
        exit 1
    fi
    
    print_success "Iniciando servidor FastAPI..."
    print_status "🌐 API disponible en: http://localhost:8989"
    print_status "📚 Documentación interactiva: http://localhost:8989/docs"
    print_status "🔍 Health check: http://localhost:8989/health"
    print_status ""
    print_status "Presiona Ctrl+C para detener el servidor"
    
    # Cambiar al directorio app y ejecutar
    cd app && python -m uvicorn main:app --host 0.0.0.0 --port 8989 --reload
}

# Función para construir imagen Docker
docker_build() {
    print_status "Construyendo imagen de Docker..."
    
    # Verificar que Docker está disponible
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado"
        print_status "Instala Docker desde: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Verificar que tenemos un modelo para incluir en la imagen
    if [ ! -f "models/active_model.json" ]; then
        print_warning "No se encontró modelo entrenado"
        print_status "El contenedor se construirá pero necesitarás entrenar un modelo"
        print_status "Puedes entrenar con: ./run.sh setup && ./run.sh train"
    fi
    
    # Construir imagen con tag específico y latest
    print_status "Construyendo imagen con tag 'penguin-classifier:latest'..."
    docker build -t penguin-classifier:latest .
    
    # Mostrar información de la imagen construida
    image_size=$(docker images penguin-classifier:latest --format "table {{.Size}}" | tail -n +2)
    print_success "Imagen Docker construida exitosamente"
    print_status "Tag: penguin-classifier:latest"
    print_status "Tamaño: $image_size"
    print_status "Para ejecutar: ./run.sh docker-start"
}

# Función para ejecutar contenedor individual
docker_start() {
    print_status "Ejecutando contenedor Docker..."
    
    # Verificar que la imagen existe
    if ! docker images penguin-classifier:latest --format "table {{.Repository}}" | grep -q penguin-classifier; then
        print_error "Imagen Docker no encontrada"
        print_status "Construye la imagen primero con: ./run.sh docker-build"
        exit 1
    fi
    
    # Verificar si el contenedor ya está existente
    if docker ps -a | grep -q penguin-api; then
        print_warning "Contenedor 'penguin-api' ya existe"
        print_status "Deteniendo y eliminando contenedor existente..."
        docker rm penguin-api
    fi


    # Detener contenedor existente si está corriendo
    existing_container=$(docker ps -q -f name=penguin-api)
    if [ ! -z "$existing_container" ]; then
        print_status "Deteniendo contenedor existente..."
        docker stop penguin-api
        docker rm penguin-api
    fi
    
    print_status "Iniciando nuevo contenedor..."
    print_status "Nombre del contenedor: penguin-api"
    print_status "Puerto: 8989"
    
    # Ejecutar contenedor con montaje de volúmenes para logs y modelos
    docker run -d \
        --name penguin-api \
        -p 8989:8989 \
        -v "$(pwd)/models:/app/models:ro" \
        -v "$(pwd)/logs:/app/logs:rw" \
        --restart unless-stopped \
        penguin-classifier:latest
    
    # Esperar un poco para que el contenedor inicie
    print_status "Esperando que el contenedor inicie..."
    sleep 5
    
    # Verificar que el contenedor esté corriendo
    if docker ps | grep -q penguin-api; then
        print_success "Contenedor iniciado exitosamente"
        print_status "🌐 API disponible en: http://localhost:8989"
        print_status "📚 Documentación: http://localhost:8989/docs"
        print_status ""
        print_status "Para ver logs: docker logs -f penguin-api"
        print_status "Para detener: docker stop penguin-api"
        print_status "Para eliminar: docker rm penguin-api"
    else
        print_error "Error al iniciar el contenedor"
        print_status "Verifica los logs con: docker logs penguin-api"
        exit 1
    fi
}

# Función para limpiar archivos temporales
clean() {
    print_status "Limpiando archivos temporales y cache..."
    
    # Limpiar archivos Python compilados
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    
    # Limpiar logs antiguos (más de 5 días)
    if [ -d "logs" ]; then
        find logs/ -name "*.log" -mtime +5 -delete 2>/dev/null || true
        print_status "Logs antiguos eliminados"
    fi
    
    # Limpiar contenedores Docker parados (opcional)
    if command -v docker &> /dev/null; then
        print_status "¿Limpiar también contenedores Docker parados? (y/N)"
        read -t 10 -n 1 response || response="n"
        echo
        if [[ "$response" =~ ^[Yy]$ ]]; then
            docker container prune -f
            print_status "Contenedores Docker limpiados"
        fi
    fi
    
    print_success "Limpieza completada"
}

# Función para mostrar el estado del proyecto
status() {
    print_status "Estado del proyecto Palmer Penguins Classifier"
    echo
    
    # Verificar entorno virtual
    if [ -d "venv" ]; then
        print_success "✅ Entorno virtual: Configurado"
    else
        print_error "❌ Entorno virtual: No configurado (ejecuta: ./run.sh setup)"
    fi
    
    # Verificar modelo
    if [ -f "models/active_model.json" ]; then
        model_size=$(du -h models/active_model.json | cut -f1)
        print_success "✅ Modelo ML: Entrenado (tamaño: $model_size)"
    else
        print_error "❌ Modelo ML: No entrenado (ejecuta: ./run.sh train)"
    fi
    
    # Verificar Docker
    if command -v docker &> /dev/null; then
        if docker images penguin-classifier:latest --format "table {{.Repository}}" | grep -q penguin-classifier; then
            print_success "✅ Imagen Docker: Construida"
        else
            print_warning "⚠️ Imagen Docker: No construida (ejecuta: ./run.sh docker-build)"
        fi
        
        if docker ps | grep -q penguin-api; then
            print_success "✅ Contenedor: Ejecutándose"
        else
            print_status "ℹ️ Contenedor: Detenido"
        fi
    else
        print_warning "⚠️ Docker: No instalado"
    fi
}

# Procesar argumentos de línea de comandos
case "${1:-help}" in
    setup)
        setup
        ;;
    train)
        train
        ;;
    api)
        api
        ;;
    docker-build)
        docker_build
        ;;
    docker-start)
        docker_start
        ;;
    status)
        status
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Comando no reconocido: $1"
        echo
        show_help
        exit 1
        ;;
esac