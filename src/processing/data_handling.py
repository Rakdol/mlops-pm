import os
import sys
import pandas as pd
from pathlib import Path
import joblib

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))
# print("PACKAGE_ROOT: ", PACKAGE_ROOT)

from src.config import config


def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data


# Serialization
def save_pipeline(pipeline_to_save, model_name=None):
    if model_name is None:
        save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    else:
        save_path = os.path.join(config.SAVE_MODEL_PATH, model_name)

    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")


# Deserialization
def load_pipeline(pipeline_to_load, model_name=None):
    if model_name is None:
        save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    else:
        save_path = os.path.join(config.SAVE_MODEL_PATH, model_name)

    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded
