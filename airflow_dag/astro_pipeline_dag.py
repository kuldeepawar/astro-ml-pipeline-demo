# airflow_dag/astro_pipeline_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from pathlib import Path

REPO_DIR = '{{ params.repo_dir }}'

default_args = {
    'owner': 'kuldeep',
    'depends_on_past': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='astro_ml_pipeline',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    preprocess = BashOperator(
        task_id='preprocess',
        bash_command=f'python {Path(REPO_DIR)/"src"/"data_pipeline.py"}',
        params={'repo_dir': '/path/to/astro-ml-pipeline-demo'}
    )

    train = BashOperator(
        task_id='train',
        bash_command=f'python {Path(REPO_DIR)/"src"/"model_pipeline.py"}',
        params={'repo_dir': '/path/to/astro-ml-pipeline-demo'}
    )

    preprocess >> train
