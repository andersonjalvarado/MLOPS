FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 streamlit && \
    mkdir -p /app && \
    chown -R streamlit:streamlit /app

USER streamlit
WORKDIR /app

COPY --chown=streamlit:streamlit requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

ENV PATH="/home/streamlit/.local/bin:${PATH}"

COPY --chown=streamlit:streamlit app.py .

EXPOSE 8503

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8503/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]