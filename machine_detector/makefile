BASE_IMAGE_NAME := machine-detector

ABSOLUTE_PATH := $(shell pwd)

DOCKERFILE := Dockerfile
IMAGE_VERSION := 0.0.1

MODEL_PORT := 8000


build-env:
	docker build --no-cache -t ${BASE_IMAGE_NAME}:${IMAGE_VERSION} -f ${DOCKERFILE} .


model-deploy:
	docker run -d -p ${MODEL_PORT}:8000 ${BASE_IMAGE_NAME}:${IMAGE_VERSION}