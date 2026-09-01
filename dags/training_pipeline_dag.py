"""Airflow DAG for the urgency triage model training pipeline.

Manually triggered (schedule=None). Runs the same scripts used for local
development: prepare_data.py (clean + split the dataset) followed by
train.py (fit the preprocessor and classifier, evaluate, save artifacts).
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="triage_model_training",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    prepare_data_task = BashOperator(
        task_id="prepare_data",
        bash_command="python scripts/prepare_data.py",
        retries=1,
    )

    train_model_task = BashOperator(
        task_id="train_model",
        bash_command="python scripts/train.py",
        retries=1,
    )

    prepare_data_task >> train_model_task
