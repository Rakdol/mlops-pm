
## Airflow Setup 
- 공식홈페이지 참조 - https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

## Fetching docker-compose file
```bash
$ curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.10.0/docker-compose.yaml'
```

- postgres port 로컬호스트 포트 수정 
```bash
services:
  postgres:
    image: postgres:14
    ports:
      - 5437:5432
```

## Setting airflow directory and user
```bash
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
```


## Initialize databse
```bash
docker compose up airflow-init
```

## Running Airflow
```bash
docker compose up -d
```


## adding dependencies via requirements.txt file
```bash
# docker-compose.yaml
#image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:2.10.0}
build: .
```
- Create Dockerfile in the same folder your docker-compose.yaml file is with content similar to:

```bash
# Dockerfile
FROM apache/airflow:2.10.0
ADD requirements.txt .
RUN pip install apache-airflow==2.10.0 -r requirements.txt
```
