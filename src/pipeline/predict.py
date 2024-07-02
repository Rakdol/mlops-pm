import os
import sys
import pandas as pd
from pathlib import Path
import joblib

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from src.config import config
from src.processing.data_handling import load_pipeline

model_pipeline = load_pipeline(config.MODEL_NAME)


def make_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = model_pipeline.predict(data)
    result = {"prediction": pred}
    return result


if __name__ == "__main__":
    make_predictions()
