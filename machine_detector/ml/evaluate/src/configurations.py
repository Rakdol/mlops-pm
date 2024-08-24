from pathlib import Path
import os
from logging import getLogger

from src.constants import PLATFORM_ENUM

logger = getLogger(__name__)


class PlatformConfigurations:
    platform = os.getenv("PLATFORM", PLATFORM_ENUM.DOCKER.value)
    if not PLATFORM_ENUM.has_value(platform):
        raise ValueError(
            f"PLATFORM must be one of {[v.value for v in PLATFORM_ENUM.__members__.values()]}"
        )


class DataConfigurations:

    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    ORIGIN_FILE = "machine failure.csv"
    LOCAL_FILE_PATH = "/opt/data/raw/total_data.csv"


class FeatureConfigurations:
    TARGET = "machine_failure"
    NUM_FEATURES = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]

    BIN_FEATURES = ["twf", "hdf", "pwf", "osf", "rnf"]  # Binary
    CAT_FEATURES = ["machine_type"]
    DROP_FEATURES = ["id", "UDI"]

    DOMAIN_FEATURES = [
        "product_id",
        "tool_wear",
        "torque",
        "rotational_speed",
        "air_temperature",
        "process_temperature",
    ]

    TRANS_FEATURES = [
        "air_process_diff",
        "speed_power",
        "torque_power",
        "tool_process",
        "temp_ratio",
    ]

    FRAME_FEATURES = (
        CAT_FEATURES + NUM_FEATURES + BIN_FEATURES + TRANS_FEATURES + ["product_id_num"]
    )

    LABELS = {0: "normal", 1: "failure"}


class TrainConfigurations:
    TRAIN_PREFIX = "train"
    TRAIN_NAME = "train_dataset.csv"
    VALID_PREFIX = "validation"
    VALID_NAME = "validation_dataset.csv"
    TEST_PREFIX = "test"
    TEST_NAME = "test_dataset.csv"
    PIPE_PREFIX = "pipe"
    PIPE_NAME = "pipe.joblib"


class ModelConfigurations:
    ONNX_FILE_NAME = "machine_rf_0.onnx"


class ObjectStoreConfigurations:
    mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "miniostorage")


logger.info(f"{PlatformConfigurations.__name__}: {PlatformConfigurations.__dict__}")
logger.info(f"{DataConfigurations.__name__}: {DataConfigurations.__dict__}")
logger.info(f"{FeatureConfigurations.__name__}: {FeatureConfigurations.__dict__}")
logger.info(f"{TrainConfigurations.__name__}: {TrainConfigurations.__dict__}")
logger.info(
    f"{ObjectStoreConfigurations.__name__}: {ObjectStoreConfigurations.__dict__}"
)
