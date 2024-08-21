import sys
from pathlib import Path

import boto3
from src.configurations import ObjectStoreConfigurations

sys.path.append(str(Path(__file__).resolve().parents[4]))

from utils.exception import CustomException
from utils.logger import logging

try:
    minio_client = boto3.client(
        "s3",
        endpoint_url=ObjectStoreConfigurations.mlflow_s3_endpoint,
        aws_access_key_id=ObjectStoreConfigurations.aws_access_key_id,
        aws_secret_access_key=ObjectStoreConfigurations.aws_secret_access_key,
    )
    logging.info(f"Minio Client successfully connected")

except Exception as e:
    minio_client = None
    logging.debug(f"Minio Client Exceptions {e}")
    raise CustomException(e, sys)
