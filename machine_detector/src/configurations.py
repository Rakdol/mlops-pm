import os


class APIConfigurations:
    title = os.getenv("API_TITLE", "Machine Detector Service")
    description = os.getenv("API_DESCRIPTION", "Machine Detector Service API")
    version = os.getenv("API_VERSION", "0.1")


class DBConfigurations:
    postgres_username = os.getenv("POSTGRES_USER", "admin")
    postgres_password = os.getenv("POSTGRES_PASSWORD", 1234)
    postgres_port = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db = os.getenv("POSTGRES_DB", "machinedb")
    # postgres_server = os.getenv(
    #     "POSTGRES_SERVER", "localhost"
    # )  # default docker bridge
    # postgres_server = os.getenv(
    #     "POSTGRES_SERVER", "172.17.0.1"
    # )  # default docker bridge
    postgres_server = os.getenv(
        "POSTGRES_SERVER", "host.docker.internal" # for macOs
    )  # default docker bridge
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"


class FeatureConfigurations:

    RAW_FEATURES = [
        "product_id",
        "machine_type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
    ]

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


class ModelConfigurations:
    PIPELINE_DIRECTORY = os.getenv("PIPELINE_DIRECTORY", "./model/")
    PIPE_FILE_NAME = os.getenv("PIPE_FILE_NAME", "machine_input_pipeline_0.joblib")
    MODEL_DIRECTORY = os.getenv("MODEL_DIRECTORY", "./model/")
    ONNX_FILE_NAME = os.getenv("ONNX_FILE_NAME", "machine_rf_0.onnx")


class ObjectStoreConfigurations:
    mlflow_s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "miniostorage")
