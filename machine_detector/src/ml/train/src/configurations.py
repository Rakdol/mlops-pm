from pathlib import Path
import os

PAKEGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv(dotenv_path=str(PAKEGE_ROOT.parent))


class DataConfigurations:
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    ORIGIN_FILE = "machine failure.csv"
    LOCAL_FILE_PATH = str(PAKEGE_ROOT.parent.parent / "data" / "raw" / TRAIN_FILE)


class FeatureConfigurations:
    TARGET = "Machine failure"
    NUM_FEATURES = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    BIN_FEATURES = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # Binary
    CAT_FEATURES = ["Type"]
    DROP_FEATURES = ["id", "UDI"]

    DOMAIN_FEATURES = [
        "Product ID",
        "Tool wear [min]",
        "Torque [Nm]",
        "Rotational speed [rpm]",
        "Air temperature [K]",
        "Process temperature [K]",
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


class DBConfigurations:
    postgres_username = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_port = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_server = os.getenv("POSTGRES_SERVER")
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"


class ObjectStoreConfigurations:
    mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "miniostorage")


class APIConfigurations:
    title = os.getenv("API_TITLE", "Failure Detector")
    description = os.getenv("API_DESCRIPTION", "machine learning system")
    version = os.getenv("API_VERSION", "0.1")
