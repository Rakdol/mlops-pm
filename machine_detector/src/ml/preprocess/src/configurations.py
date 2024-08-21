from pathlib import Path
import os

PAKEGE_ROOT = Path(__file__).resolve()
print(PAKEGE_ROOT)

parent_dir = Path(__file__).resolve().parents[2]


class DataConfigurations:

    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    ORIGIN_FILE = "machine failure.csv"
    LOCAL_FILE_PATH = str(parent_dir / "data" / "raw" / TRAIN_FILE)


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
