FROM apache/airflow:2.8.3-python3.11

USER airflow

COPY --chown=airflow:root requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt


