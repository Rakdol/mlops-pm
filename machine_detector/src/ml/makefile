ABSOLUTE_PATH := $(shell pwd)
BASE_IMAGE_NAME := detection_system
IMAGE_VERSION := 0.0.1

DOCKERFILE := Dockerfile
include .env

build-dev:
	pip install -r requirements.txt

build-env:
	docker build --no-cache -t ${BASE_IMAGE_NAME}:${IMAGE_VERSION} -f ${DOCKERFILE} .

run-train:
	export MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL} && \
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && \
	export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} && \
	export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} && \
	mlflow run . --env-manager=local

	