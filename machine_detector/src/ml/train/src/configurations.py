from pathlib import Path
import os


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


class ModelConfigurations:
    pass


class ObjectStoreConfigurations:
    mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "miniostorage")
