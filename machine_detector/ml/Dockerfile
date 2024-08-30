FROM python:3.10-slim


ENV PROJECT_DIR=/mlflow/projects
ENV CODE_DIR=/mlflow/projects/code


WORKDIR /${PROJECT_DIR}

COPY requirements.txt /${PROJECT_DIR}/requirements.txt

RUN apt-get update && apt-get install -y \
    apt-utils gcc \
    && pip install --upgrade pip setuptools wheel \
    && pip install  --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /${CODE_DIR}