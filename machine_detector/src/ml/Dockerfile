FROM python:3.10-slim


ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=1234
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=machinedb
ENV POSTGRES_SERVER=http://172.17.0.1

ENV PROJECT_DIR=/mlflow/projects
ENV CODE_DIR=/mlflow/projects/code
ENV MLFLOW_TRACKING_URI=http://172.17.0.1:5001
ENV MLFLOW_S3_ENDPOINT_URL=http://172.17.0.1:9000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=miniostorage

WORKDIR /${PROJECT_DIR}

COPY requirements.txt /${PROJECT_DIR}/requirements.txt

RUN apt-get update && apt-get install -y \
    apt-utils gcc \
    && pip install --upgrade pip setuptools wheel \
    && pip install  --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /${CODE_DIR}