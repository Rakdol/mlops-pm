import os
import json

import time
import joblib
import logging
from typing import Dict, List, Optinal
from argparse import ArgumentParser, RawTextHelpFormatter

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import onnxruntime as rt

from src.transformers import get_input_pipeline
from src.configurations import FeatureConfigurations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFClassifier(object):
    def __init__(
        self,
        pipeline_directory: str,
        model_directory: str,
        onnx_file_name: Optinal[str],
        providers: List[str] = ["CPUExecutionProvider"],
    ):
        self.preprocess_pipeline = joblib.load(pipeline_directory)
        self.onnx_session = rt.InferenceSession(
            os.path.join(model_directory, onnx_file_name), providers=providers
        )
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_label_name = self.onnx_session.get_outputs()[0].name
        self.labels = FeatureConfigurations.LABELS

    def transform(self, x: pd.DataFrame):
        x_transformed = (
            self.preprocess_pipeline.transform(x).to_numpy().astype(np.float32)
        )
        return x_transformed

    def predict(self, x: pd.DataFrame):

        pred = self.onnx_session.run(
            [self.onnx_label_name], {self.onnx_input_name: x}[0]
        )
        return pred

    def predict_label(self, x: pd.DataFrame):
        pred = self.onnx_session.run(
            [self.onnx_label_name], {self.onnx_input_name: x}[0]
        )
        return str(self.labels[pred[0]])


def batch_evaluate(
    test_data_directory: str,
    pipeline_directory: str,
    model_directory: str,
    model_file_name: str,
    batch_size: int = 32,
) -> Dict:

    classifier = RFClassifier(
        pipeline_directory=pipeline_directory,
        model_directory=model_directory,
        model_file_name=model_file_name,
    )

    test_data = pd.read_csv(test_data_directory)
    X_test = test_data.drop(labels=[FeatureConfigurations.TARGET], axis=1)
    y_test = test_data[FeatureConfigurations.TARGET]

    batch_labels = []
    batch_predicted = []
    batch_transform_durations = []
    batch_infer_durations = []

    batch_step = 0
    for i in range(0, X_test.shape[0], batch_size):
        x = X_test[i : i + batch_size]

        batch_transform_start = time.time()
        x_transform = classifier.transform(x)
        batch_transform_end = time.time()
        pred_label = classifier.predict_label(x_transform)
        batch_infer_end = time.time()
        pred_number = classifier.predict(x_transform)

        batch_labels.extend([p for p in pred_label.tolist()])
        batch_predicted.extend([p for p in pred_number.tolist()])
        batch_transform_durations.append(batch_transform_end - batch_transform_start)
        batch_infer_durations.append(batch_infer_end - batch_transform_end)
        batch_step += 1

    batch_total_time = sum(batch_transform_durations) + sum(batch_infer_durations)
    batch_transform_total_time = sum(batch_transform_durations)
    batch_infer_total_time = sum(batch_infer_durations)
    average_batch_duration_second = batch_total_time / batch_step
    average_batch_transform_second = batch_transform_total_time / batch_step
    average_batch_infer_second = batch_infer_total_time / batch_step

    accuracy = accuracy_score(y_test, batch_predicted)
    precision = precision_score(y_test, batch_predicted, average="macro")
    recall = recall_score(y_test, batch_predicted, average="macro")
    f1_macro = f1_score(y_test, batch_predicted, average="macro")

    evaluation = {
        "batch_size": batch_size,
        "batch_step": batch_step,
        "batch_total_time": batch_total_time,
        "batch_transform_total_time": batch_transform_total_time,
        "batch_infer_total_time": batch_infer_total_time,
        "average_batch_duration_second": average_batch_duration_second,
        "average_batch_transform_duration_second": average_batch_transform_second,
        "average_batch_infer_duration_second": average_batch_infer_second,
        "accuracy_score": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1_macro,
    }

    return {
        "evaluation": evaluation,
        "predicted_labels": batch_labels,
        "predictions": batch_predicted,
    }


def evaluate(
    test_data_directory: str,
    pipeline_directory: str,
    model_directory: str,
    model_file_name: str,
) -> Dict:

    classifier = RFClassifier(
        pipeline_directory=pipeline_directory,
        model_directory=model_directory,
        model_file_name=model_file_name,
    )

    test_data = pd.read_csv(test_data_directory)
    X_test = test_data.drop(labels=[FeatureConfigurations.TARGET], axis=1)
    y_test = test_data[FeatureConfigurations.TARGET]

    labels = []
    predicted = []
    trans_durations = []
    infer_durations = []
    for i, row in X_test.iterrows():
        x = pd.DataFrame([row])

        transform_start = time.time()
        x_transform = classifier.transform(x)
        transform_end = time.time()
        pred_label = classifier.predict_label(x_transform)
        infer_end = time.time()
        pred_number = classifier.predict(x_transform)

        labels.append(pred_label)
        predicted.append(pred_number)
        trans_durations.append(transform_end - transform_start)
        infer_durations.append(infer_end - transform_end)

    total_time = sum(trans_durations) + sum(infer_durations)
    total_transform_time = sum(trans_durations)
    total_infer_time = sum(infer_durations)
    total_tested = len(predicted)
    average_duration_second = total_time / total_tested
    average_transform_duration_second = total_transform_time / total_tested
    average_infer_duration_second = total_infer_time / total_tested

    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, average="macro")
    recall = recall_score(y_test, predicted, average="macro")
    f1_macro = f1_score(y_test, predicted, average="macro")

    evaluation = {
        "total_step": total_tested,
        "total_time": total_time,
        "total_transform_time": total_transform_time,
        "total_infer_time": total_infer_time,
        "average_duration_second": average_duration_second,
        "average_transform_duration_second": average_transform_duration_second,
        "average_infer_duration_second": average_infer_duration_second,
        "accuracy_score": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1_macro,
    }
    return {
        "evaluation": evaluation,
        "predicted_labels": labels,
        "predictions": predicted,
    }
