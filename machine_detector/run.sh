#!/bin/bash

set -eu

HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-4}
UVICORN_WORKER=${UVICORN_WORKER:-"uvicorn.workers.UvicornWorker"}
LIMIT_MAX_REQUESTS=${LIMIT_MAX_REQUESTS:-65536}
MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-2048}
GRACEFUL_TIMEOUT=${GRACEFUL_TIMEOUT:-10}
APP_NAME=${APP_NAME:-"src.api.app:app"}

gunicorn ${APP_NAME} \
    -b ${HOST}:${PORT} \
    -w ${WORKERS} \
    -k ${UVICORN_WORKER} \
    --max-requests ${LIMIT_MAX_REQUESTS} \
    --max-requests-jitter ${MAX_REQUESTS_JITTER} \
    --graceful-timeout ${GRACEFUL_TIMEOUT}
