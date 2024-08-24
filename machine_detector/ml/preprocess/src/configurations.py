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


class DBConfigurations:
    postgres_username = os.getenv("POSTGRES_USER", "admin")
    postgres_password = os.getenv("POSTGRES_PASSWORD", 1234)
    postgres_port = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db = os.getenv("POSTGRES_DB", "machinedb")
    postgres_server = os.getenv(
        "POSTGRES_SERVER", "172.17.0.1"
    )  # default docker bridge
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"


logger.info(f"{PlatformConfigurations.__name__}: {PlatformConfigurations.__dict__}")
logger.info(f"{DataConfigurations.__name__}: {DataConfigurations.__dict__}")
logger.info(f"{FeatureConfigurations.__name__}: {FeatureConfigurations.__dict__}")
