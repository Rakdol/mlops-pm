import os
import sys
from typing import List, Dict, Union, Optional

import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt
from pydantic import BaseModel

from pathlib import Path


from src.configurations import (
    FeatureConfigurations,
    ModelConfigurations,
)
from src.utils.logger import logging


pipe_path = str(Path(__file__).resolve().parents[0])
sys.path.append(pipe_path)

from src.transformers import get_input_pipeline


class Data(BaseModel):
    data: List[List[Union[str, int, float]]] = [
        ["L49624", "L ", 299.2, 308.6, 1267, 40.4, 76.0, 0, 0, 0, 0, 0]
    ]

class Classifier(object):
    def __init__(
        self,
        pipeline_directory: str,
        pipe_file_name: str,
        model_directory: str,
        onnx_file_name: Optional[str],
        providers: List[str] = ["CPUExecutionProvider"],
    ):
        self.preprocess_pipeline = self.load_pipeline(
            pipeline_directory, pipe_file_name
        )
        self.onnx_session = self.load_model(
            model_directory=model_directory,
            onnx_file_name=onnx_file_name,
            providers=providers,
        )
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_label_name = self.onnx_session.get_outputs()[0].name
        self.labels = FeatureConfigurations.LABELS
        self.columns = FeatureConfigurations.RAW_FEATURES

    def load_pipeline(self, pipeline_directory: str, pipe_file_name: str):
        pipe_path = os.path.join(pipeline_directory, pipe_file_name)

        # pipe_path = mlflow.artifacts.download_artifacts(
        #     os.path.join("s3://mlflow", pipeline_directory)
        # )
        logging.info("... pipeline loading ...")
        pipe = joblib.load(pipe_path)
        logging.info("... pipeline loaded ...")

        return pipe

    def load_model(
        self, model_directory: str, onnx_file_name: str, providers: List[str]
    ):
        onnx_directory = os.path.join(model_directory, onnx_file_name)
        logging.info("... onnx model loading ...")
        sess = rt.InferenceSession(onnx_directory, providers=providers)
        logging.info("... onnx model loaded ...")

        return sess

    def transform(self, x: List[List[Union[str, int, float]]]):

        logging.info("... input data is trasnforming ...")
        x_frame = pd.DataFrame(x, columns=self.columns)
        x_transformed = (
            self.preprocess_pipeline.transform(x_frame).to_numpy().astype(np.float32)
        )
        logging.info("... input data has been trasnformed ...")
        return x_transformed

    def predict(self, x: pd.DataFrame):

        logging.info("... model is predicting ...")
        pred = self.onnx_session.run([self.onnx_label_name], {self.onnx_input_name: x})
        logging.info("... model has been predicted ...")

        return [p for p in pred[0].tolist()]

    def predict_label(self, x: pd.DataFrame):
        logging.info("... model is predicting for label ...")
        pred = self.onnx_session.run([self.onnx_label_name], {self.onnx_input_name: x})
        logging.info("... model has been predicted for label ...")
        return [str(self.labels[p]) for p in pred[0].tolist()]


print("current PATH ::-----", os.getcwd())
classifier = Classifier(
    pipeline_directory=ModelConfigurations().PIPELINE_DIRECTORY,
    pipe_file_name=ModelConfigurations().PIPE_FILE_NAME,
    model_directory=ModelConfigurations().MODEL_DIRECTORY,
    onnx_file_name=ModelConfigurations().ONNX_FILE_NAME,
)
logging.info("Classifier sucessfully loaded")
