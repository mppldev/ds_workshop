"""
Code that goes along with the Airflow tutorial located at:
https://github.com/apache/airflow/blob/master/airflow/example_dags/tutorial.py
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
from functools import reduce


default_args = {
    'owner': 'Airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG('soccer_pipeline_final', default_args=default_args, schedule_interval=timedelta(days=1))

# t1, t2 and t3 are examples of tasks created by instantiating operators

t1 = BashOperator(
    task_id="check_folder",
    bash_command="cp -r /usr/local/airflow/data ../",
    dag=dag
)


t2_cmd = "; ".join(reduce(
    lambda acc, y: [*acc, f"python /usr/local/airflow/code/scraping.py {y} -o ../data"],
    range(2011, 2019),
    []
))
t2 = BashOperator(
    task_id="scrape_for_data",
    bash_command=t2_cmd,
    dag=dag
)

t3 = BashOperator(
    task_id="verify_data",
    bash_command="ls ../data/",
    dag=dag
)

t4 = BashOperator(
    task_id="prepare_data",
    bash_command="python /usr/local/airflow/code/prepare.py -o ../data -i ../data",
    dag=dag
)

t5 = BashOperator(
    task_id="verify_data_again",
    bash_command="ls ../data/",
    dag=dag
)

t6 = BashOperator(
    task_id="fit_model",
    bash_command="python /usr/local/airflow/code/model.py -o ../models -i ../data",
    dag=dag
)



dag >> t1 >> t2 >> t3 >> t4 >> t5 >> t6