import datetime
import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.sql_sensor import SqlSensor

from sqlalchemy.orm import Session
from db.database import Session_Event, Session_Source
from db.models import MachineData, MachinePredict


def get_last_processed_time(**kwargs):
    with Session_Source() as session_source:
        state = session_source.query(MachineData).order_by(MachineData.timestamp.desc()).first()
        
    if state:
        return state.timestamp
    else:
        return datetime.datetime.min  


def request_for_detection(**kwargs):
    # data example : {'data': [['M23918', 'M ', 297.2, 308.6, 1576, 39.0, 100.0, 0, 0, 0, 0, 0]]}
    
    api_url = "http://172.17.0.1:8000/v0.1/api/predict"  # Corrected URL
    
    with Session_Source() as session_source:
        new_data_statement = session_source.query(MachineData).filter(MachineData.timestamp > kwargs['last_processed_time']).statement
        
        x_data = pd.read_sql(new_data_statement, session_source.connection())
    
    feat = x_data.drop("machine_failure", axis=1)
    times = feat["timestamp"]
    
    x_list = [v.iloc[2:].to_list() for i, v in feat.iterrows()]
    x = {"data": x_list}
    
    response = requests.post(
        url=api_url,
        json=x,
        headers={"Content-Type": "application/json"}
    ).json()
    
    predictions = response["prediction"]
    time_list = [state.strftime('%Y-%m-%d %X') for state in times]
    
    return {'times': time_list, 'predictions': predictions}


def insert_event_data(**kwargs):
    LABELS = {0: "normal", 1: "failure"}
    detection_result = kwargs['ti'].xcom_pull(task_ids='request_for_detection')
    
    predictions = detection_result['predictions']
    times = detection_result['times']
    
    for time, pred in zip(times, predictions):
        data = MachinePredict(source_time= datetime.datetime.strptime(time, '%Y-%m-%d %X') - datetime.timedelta(hours=9),
                              predict_time=datetime.datetime.utcnow(),
                              predict_label = str(LABELS[pred]),
                              predict_value = str(pred))
        
        with Session_Event() as session_event:
            session_event.add(data)
            session_event.commit()
            session_event.refresh(data)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.datetime(2024, 8, 31),
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

# Define the DAG
with DAG('Anomaly_event_detector',
          description='Anomaly Event Detector Example',
          default_args=default_args,
          schedule_interval=datetime.timedelta(seconds=30), # Sensor trigger
          catchup=False) as dag:


    get_last_processed_time_task = PythonOperator(
        task_id='get_last_processed_time',
        python_callable=get_last_processed_time,
        provide_context=True,
        dag=dag,
    )

    # SqlSensor 설정
    wait_for_new_data = SqlSensor(
        task_id='wait_for_new_data',
        conn_id='source-postgres',  # Airflow Connection에서 설정된 Postgres 연결 ID
        sql="SELECT COUNT(*) FROM machine_data WHERE timestamp > '{{ task_instance.xcom_pull(task_ids='get_last_processed_time') }}' LIMIT 1;",
        mode='poke',
        poke_interval=1,  # 1초마다 쿼리 실행
        timeout=60,  # 최대 1분 동안 대기
        dag=dag
        )
    
    
    # 이상 감지 작업 PythonOperator 정의
    anomaly_request = PythonOperator(
        task_id='request_for_detection',
        python_callable=request_for_detection,
        provide_context=True,
        op_kwargs={'last_processed_time': '{{ task_instance.xcom_pull(task_ids="get_last_processed_time") }}'},
        dag=dag,
    )
    
    insert_event_data_task = PythonOperator(
        task_id="insert_event_data",
        python_callable=insert_event_data,
        provide_context=True,
        dag=dag,
    )
    


# 작업 순서 정의
get_last_processed_time_task >>  wait_for_new_data >> anomaly_request >> insert_event_data_task