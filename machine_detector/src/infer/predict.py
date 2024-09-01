import os
from typing import List, Union
import numpy as np
import pandas as pd
import onnxruntime as rt
from pydantic import BaseModel

from src.configurations import (
    FeatureConfigurations,
    ModelConfigurations,
)
from src.utils.logger import logging


class Data(BaseModel):

    # "product_id", "machine_type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "twf", "hdf", "pwf", "osf","rnf",
    data: List[List[Union[str, int, float]]] = [
        ["L49624", "L ", 299.2, 308.6, 1267, 40.4, 76.0, 0, 0, 0, 0, 0]
    ]


class Classifier(object):
    def __init__(
        self,
        model_directory: str,
        onnx_file_name: str,
        providers: List[str] = ["CPUExecutionProvider"],
    ):

        self.onnx_session = self.load_model(
            model_directory=model_directory,
            onnx_file_name=onnx_file_name,
            providers=providers,
        )
        self.labels = FeatureConfigurations.LABELS
        self.columns = FeatureConfigurations.RAW_FEATURES

    def load_model(
        self, model_directory: str, onnx_file_name: str, providers: List[str]
    ):
        onnx_directory = os.path.join(model_directory, onnx_file_name)
        logging.info("... onnx model loading ...")
        sess = rt.InferenceSession(onnx_directory, providers=providers)
        logging.info("... onnx model loaded ...")

        return sess

    def convert_onnx_input(
        self,
        x: pd.DataFrame,
        cols: list = [
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ],
    ):

        inputs = {c: x[c].values for c in x.columns}

        for c in cols:
            if c == "rotational_speed" or c == "tool_wear":
                inputs[c] = inputs[c].astype(np.int64)
            else:
                inputs[c] = inputs[c].astype(np.float32)
        for k in inputs:
            inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))

        return inputs

    def predict(self, x: list):
        x = pd.DataFrame(x, columns=self.columns)
        inputs = self.convert_onnx_input(
            x[FeatureConfigurations.CAT_FEATURES + FeatureConfigurations.NUM_FEATURES]
        )

        logging.info("... model is predicting ...")
        pred = self.onnx_session.run(None, inputs)
        logging.info("... model has been predicted ...")

        return [p for p in pred[0].tolist()]

    def predict_label(self, x: list):
        x = pd.DataFrame(x, columns=self.columns)
        inputs = self.convert_onnx_input(
            x[FeatureConfigurations.CAT_FEATURES + FeatureConfigurations.NUM_FEATURES]
        )
        logging.info("... model is predicting for label ...")
        pred = self.onnx_session.run(None, inputs)
        logging.info("... model has been predicted for label ...")
        return [str(self.labels[p]) for p in pred[0].tolist()]


classifier = Classifier(
    model_directory=ModelConfigurations().MODEL_DIRECTORY,
    onnx_file_name=ModelConfigurations().ONNX_FILE_NAME,
)
logging.info("Classifier sucessfully loaded")
