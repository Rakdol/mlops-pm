import os
import json
from typing import List, Optional

import mlflow
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt

from src.transformers import get_input_pipeline
from machine_detector.src.configurations import (
    FeatureConfigurations,
    ModelConfigurations,
)

from typing import Dict, List, Sequence


from pydantic import BaseModel


class Data(BaseModel):
    data: List[List[float]] = [[5.1, 3.5, 1.4, 0.2]]


class Classifier(object):
    def __init__(
        self,
        model_filepath: str,
        label_filepath: str,
    ):
        self.model_filepath: str = model_filepath
        self.label_filepath: str = label_filepath
        self.classifier = None
        self.label: Dict[str, str] = {}
        self.input_name: str = ""
        self.output_name: str = ""

        self.load_model()
        self.load_label()

    def load_model(self):
        logging.info(f"load model in {self.model_filepath}")
        self.classifier = rt.InferenceSession(self.model_filepath)
        self.input_name = self.classifier.get_inputs()[0].name
        self.output_name = self.classifier.get_outputs()[0].name
        logging.info(f"initialized model")

    def load_label(self):
        logging.info(f"load label in {self.label_filepath}")
        with open(self.label_filepath, "r") as f:
            self.label = json.load(f)
        logging.info(f"label: {self.label}")

    def predict(self, data: List[List[int]]) -> np.ndarray:
        np_data = np.array(data).astype(np.float32)
        prediction = self.classifier.run(None, {self.input_name: np_data})
        output = np.array(list(prediction[1][0].values()))
        logging.info(f"predict proba {output}")
        return output

    def predict_label(self, data: List[List[int]]) -> str:
        prediction = self.predict(data=data)
        argmax = int(np.argmax(np.array(prediction)))
        return self.label[str(argmax)]


classifier = Classifier(
    model_filepath=ModelConfigurations().model_filepath,
    label_filepath=ModelConfigurations().label_filepath,
)


class RFClassifier(object):
    def __init__(
        self,
        pipeline_directory: str,
        model_directory: str,
        onnx_file_name: Optional[str],
        providers: List[str] = ["CPUExecutionProvider"],
    ):
        self.preprocess_pipeline = self.get_input_pipe(pipeline_directory)
        self.onnx_session = self.get_onnx_model(
            model_directory=model_directory,
            onnx_file_name=onnx_file_name,
            providers=providers,
        )
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_label_name = self.onnx_session.get_outputs()[0].name
        self.labels = FeatureConfigurations.LABELS

    def get_input_pipe(self, pipeline_directory: str):
        logging.info(f"loading pipeline in {pipeline_directory}")
        pipe_path = mlflow.artifacts.download_artifacts(
            os.path.join("s3://mlflow", pipeline_directory)
        )
        pipe = joblib.load(pipe_path)
        logging.info(f"loaded pipeline")
        return pipe

    def get_onnx_model(
        self, model_directory: str, onnx_file_name: str, providers: List[str]
    ):
        onnx_directory = os.path.join(model_directory, onnx_file_name)
        onnx = mlflow.artifacts.download_artifacts(
            os.path.join("s3://mlflow", onnx_directory)
        )

        sess = rt.InferenceSession(onnx, providers=providers)
        logging.info(f"initialize onnx model")
        return sess

    def transform(self, x: pd.DataFrame):
        x_transformed = (
            self.preprocess_pipeline.transform(x).to_numpy().astype(np.float32)
        )
        return x_transformed

    def predict(self, x: pd.DataFrame):

        pred = self.onnx_session.run([self.onnx_label_name], {self.onnx_input_name: x})

        return [p for p in pred[0].tolist()]

    def predict_label(self, x: pd.DataFrame):

        pred = self.onnx_session.run([self.onnx_label_name], {self.onnx_input_name: x})
        return [str(self.labels[p]) for p in pred[0].tolist()]
