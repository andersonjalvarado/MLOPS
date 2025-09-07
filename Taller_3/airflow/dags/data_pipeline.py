from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import seaborn as sns
from airflow.decorators import dag, task
from sqlalchemy import create_engine, text, Table, Column, MetaData
from sqlalchemy import String, Float, Integer, Boolean, DateTime


def get_mysql_engine():
    host = os.environ.get("DATA_DB_HOST", "mysql")
    user = os.environ.get("DATA_DB_USER", "penguin")
    password = os.environ.get("DATA_DB_PASSWORD", "penguin")
    dbname = os.environ.get("DATA_DB_NAME", "penguins_data")
    return create_engine(f"mysql+pymysql://{user}:{password}@{host}/{dbname}")


def get_mysql_url() -> str:
    host = os.environ.get("DATA_DB_HOST", "mysql")
    user = os.environ.get("DATA_DB_USER", "penguin")
    password = os.environ.get("DATA_DB_PASSWORD", "penguin")
    dbname = os.environ.get("DATA_DB_NAME", "penguins_data")
    return f"mysql+pymysql://{user}:{password}@{host}/{dbname}"


def read_sql_robustly(query: str) -> pd.DataFrame:
    """
    Reads data from MySQL using pure SQLAlchemy to bypass pandas's problematic
    DBAPI/connection detection logic within the Airflow task environment.
    """
    engine = get_mysql_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        columns = result.keys()
        return pd.DataFrame(rows, columns=columns)


def write_dataframe(engine, table_name: str, df: pd.DataFrame) -> None:
    metadata = MetaData()

    # Map pandas dtypes to SQLAlchemy types
    columns: list[Column] = []
    for col_name, dtype in df.dtypes.items():
        if str(dtype).startswith("float"):
            col_type = Float()
        elif str(dtype).startswith("int"):
            col_type = Integer()
        elif str(dtype) == "bool":
            col_type = Boolean()
        elif "datetime" in str(dtype):
            col_type = DateTime()
        else:
            # Fallback for object/category
            col_type = String(length=255)
        columns.append(Column(col_name, col_type))

    table = Table(table_name, metadata, *columns)

    with engine.begin() as conn:
        # Drop if exists, then create fresh (REPLACE semantics)
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
        metadata.create_all(conn, tables=[table])
        # Insert rows in chunks
        records = df.to_dict(orient="records")
        if records:
            conn.execute(table.insert(), records)


RAW_TABLE = "penguins_raw"
CLEAN_TABLE = "penguins_clean"


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["penguins", "data"],
    default_args={"owner": "airflow"},
    description="Carga y limpieza básica de datos de Penguins en MySQL",
)
def data_pipeline():
    @task(task_id="clear_database", doc_md="Borra tablas RAW y CLEAN de MySQL si existen")
    def clear_database():
        engine = get_mysql_engine()
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {RAW_TABLE}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {CLEAN_TABLE}"))

    @task(task_id="load_raw", doc_md="Carga dataset penguins a tabla RAW (palmerpenguins fallback seaborn)")
    def load_raw_data():
        try:
            from palmerpenguins import load_penguins  # type: ignore
            df = load_penguins()
        except Exception:
            df = sns.load_dataset("penguins")
        # WORKAROUND for MySQL NaN handling: drop rows with any NaN before insertion
        df.dropna(inplace=True)
        engine = get_mysql_engine()
        write_dataframe(engine, RAW_TABLE, df)

    @task(task_id="basic_clean", doc_md="Elimina filas con NA en columnas clave y persiste a CLEAN")
    def basic_clean():
        # Using the DB URL string directly forces pandas to manage the connection,
        # which seems to bypass the environment issues with SQLAlchemy object detection.
        df = read_sql_robustly(f"SELECT * FROM `{RAW_TABLE}`")

        # This second dropna is redundant if load_raw already cleaned the data,
        # but ensures the contract of `penguins_clean` is met.
        df = df.dropna(subset=[
            "species",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "island",
            "sex",
        ])

        engine = get_mysql_engine()
        write_dataframe(engine, CLEAN_TABLE, df)

    clear = clear_database()
    load = load_raw_data()
    clean = basic_clean()

    clear >> load >> clean


data_pipeline()


