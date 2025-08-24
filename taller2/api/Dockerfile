FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

# Copia el código de la API
COPY main.py .

# Instala dependencias (añade requirements.txt si tienes más)
RUN uv pip install --system fastapi uvicorn[standard] joblib scikit-learn==1.6.1 pandas

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]