from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
import time

default_args = {
    'owner': 'Airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'email': ['josmoreira@deloitte.pt'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

def line_counter(**context):
    with open("/tmp/output.txt", "r") as f:
        lines = f.readlines()
    return len(lines), lines

def print_number_result(**context):
    
    count_result, lines = context["ti"].xcom_pull(key=None, task_ids="line_counter")
    print(f"The result nr is {count_result}")

def print_result_lines(**context):
    count_result, lines = context["ti"].xcom_pull(key=None, task_ids="line_counter")
    print(f"The result lines are {lines}")

dag = DAG('useless_pipeline', default_args=default_args, schedule_interval=timedelta(days=1))

task_1 = BashOperator(
    task_id="ls",
    bash_command="ls . > /tmp/output.txt",
    dag=dag)

task_2 = PythonOperator(
    task_id="line_counter",
    python_callable=line_counter,
    dag=dag,
    xcom_push=True)

task_3 = PythonOperator(
    task_id="print_nr_result",
    python_callable=print_number_result,
    dag=dag,
    provide_context=True)

task_4 = PythonOperator(
    task_id="print_result_lines",
    python_callable=print_result_lines,
    dag=dag,
    provide_context=True)

dag >> task_1 >> task_2 >> [task_3, task_4]