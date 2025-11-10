import pandas as pd
from sqlalchemy import create_engine, text
from airflow.decorators import dag, task
from datetime import datetime

DB_RAW_HOST = 'BD_raw'
DB_RAW_PORT = '5541'
DB_RAW_USER = 'Admin'
DB_RAW_PASSWORD = 'Admin'
DB_RAW_NAME = 'raw_data'

DATA_PATH = '/opt/airflow/data/Diabetes.csv'
TABLE_NAME = 'diabetic_data'
CHUNK_SIZE = 15000

@dag(
    dag_id='1_ingest_raw_data',
    start_date=datetime(2025, 11, 2),
    schedule_interval=None,
    catchup=False,
    tags=['ingestion'],
)
def ingest_raw_data_dag():

    @task
    def clear_raw_data_table():
        connection_url = f"postgresql://{DB_RAW_USER}:{DB_RAW_PASSWORD}@{DB_RAW_HOST}:{DB_RAW_PORT}/{DB_RAW_NAME}"
        engine = create_engine(connection_url)
        try:
            with engine.connect() as connection:
                connection.execute(text(f'TRUNCATE TABLE "{TABLE_NAME}" RESTART IDENTITY;'))
        except Exception as e:
            print(f"No se pudo limpiar la tabla: {e}")

    @task
    def ingest_csv_to_postgres():
        connection_url = f"postgresql://{DB_RAW_USER}:{DB_RAW_PASSWORD}@{DB_RAW_HOST}:{DB_RAW_PORT}/{DB_RAW_NAME}"
        engine = create_engine(connection_url)
        csv_iterator = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)
        batch_num = 1
        for chunk in csv_iterator:
            if_exists_strategy = 'replace' if batch_num == 1 else 'append'
            chunk.to_sql(TABLE_NAME, engine, if_exists=if_exists_strategy, index=False)
            print(f"Lote {batch_num} insertado con éxito.")
            batch_num += 1
        return f"Proceso finalizado. {batch_num - 1} lotes procesados."

    clear_task = clear_raw_data_table()
    ingest_task = ingest_csv_to_postgres()
    clear_task >> ingest_task

ingest_raw_data_dag_instance = ingest_raw_data_dag()
