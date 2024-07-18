import os
import sys
import json
from pathlib import Path
from logging import getLogger
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_DIR = os.path.join(PACKAGE_ROOT.parent, "config")
load_dotenv(os.path.join(BASE_DIR, ".env"))

logger = getLogger(__name__)

from src.config import config
from src.app.schemas import MachineData, MachineFailure
from pipeline import input_pipeline


class Data(BaseModel):
    data: List[List[Any]] = [
        [6, "L56736", "L", 299.6, 311, 1413, 42.9, 156, 0, 0, 0, 0, 0]
    ]


class Claasifier(object):

    def __init__(self, model_filepath: str, label_filepath: str):

        self.model_filepath: str = model_filepath
        self.label_filepath: str = label_filepath
        self.classifier = None
        self.label: Dict[str, str] = {}
        self.input_name: str = ""
        self.output_name: str = ""

        self.load_model()
        self.load_label()

    def load_model(self):
        logger.info(f"load model in {self.model_filepath}")
        self.classifier = mlflow.sklearn.load_model(model_uri=self.model_filepath)
        self.input_name = list(MachineData.model_fields)
        self.output_name = list(MachineFailure.model_fields)
        logger.info(f"initialized model")

    def load_label(self):
        logger.info(f"load label in {self.label_filepath}")
        with open(self.label_filepath, "r") as f:
            self.label = json.load(f)
        logger.info(f"label: {self.label}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:

        prediction = self.classifier.predict_proba(data)
        logger.info(f"predict {prediction}")

        return prediction

    def predict_label(self, data: pd.DataFrame) -> str:
        prediction = self.classifier.predict(data)
        argmax = int(np.argmax(np.array(prediction)))
        return self.label[str(argmax)]


if __name__ == "__main__":
    classifier = Claasifier(
        model_filepath=config.MODEL_PATH,
        label_filepath=config.LABEL_PATH,
    )

    data = Data().data

    np_data = pd.DataFrame(
        np.array(data).reshape(1, -1),
        columns=[
            "id",
            "Product ID",
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
        ],
    )
    df = np_data.astype(
        {
            "id": int,
            "Product ID": str,
            "Type": str,
            "Air temperature [K]": float,
            "Process temperature [K]": float,
            "Rotational speed [rpm]": int,
            "Torque [Nm]": float,
            "Tool wear [min]": int,
            "TWF": int,
            "HDF": int,
            "PWF": int,
            "OSF": int,
            "RNF": int,
        }
    )
    print(classifier.predict_label(df))
