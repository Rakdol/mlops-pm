import sys
from pathlib import Path

import boto3
from src.configurations import ObjectStoreConfigurations

from logging import getLogger

logger = getLogger(__name__)

try:
    bucket_client = boto3.client(
        "s3",
        endpoint_url=ObjectStoreConfigurations.mlflow_s3_endpoint,
        aws_access_key_id=ObjectStoreConfigurations.aws_access_key_id,
        aws_secret_access_key=ObjectStoreConfigurations.aws_secret_access_key,
    )
    logger.info(f"Minio Client successfully connected")

except Exception as e:
    bucket_client = None
    logger.debug(f"Minio Client Exceptions {e}")
    raise ConnectionError("Minio Client not connected")
