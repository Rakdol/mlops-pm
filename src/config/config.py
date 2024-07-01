import pathlib
import os

PACKAGE_ROOT = pathlib.Path(__file__).parent.parent.parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
ORIGIN_FILE = "machine failure.csv"

MODEL_NAME = "model.pkl"
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "artifacts/models")

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
