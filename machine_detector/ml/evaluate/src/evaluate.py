import os
import sys
import json

import time
import joblib
import logging
from io import BytesIO
from typing import Dict, List, Optional
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
from src.configurations import (
    FeatureConfigurations,
    TrainConfigurations,
    ModelConfigurations,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFClassifier(object):
    def __init__(
        self,
        model_directory: str,
        onnx_file_name: Optional[str],
        providers: List[str] = ["CPUExecutionProvider"],
    ):
        self.onnx_session = self.get_onnx_model(
            model_directory=model_directory,
            onnx_file_name=onnx_file_name,
            providers=providers,
        )

        self.labels = FeatureConfigurations.LABELS

    def get_onnx_model(
        self, model_directory: str, onnx_file_name: str, providers: List[str]
    ):
        onnx_directory = os.path.join(model_directory, onnx_file_name)
        onnx = mlflow.artifacts.download_artifacts(
            os.path.join("s3://mlflow", onnx_directory)
        )

        sess = rt.InferenceSession(onnx, providers=providers)

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

    def predict(self, x: pd.DataFrame):
        inputs = self.convert_onnx_input(
            x[FeatureConfigurations.CAT_FEATURES + FeatureConfigurations.NUM_FEATURES]
        )

        pred = self.onnx_session.run(None, inputs)

        return [p for p in pred[0].tolist()]

    def predict_label(self, x: pd.DataFrame):
        inputs = self.convert_onnx_input(
            x[FeatureConfigurations.CAT_FEATURES + FeatureConfigurations.NUM_FEATURES]
        )

        pred = self.onnx_session.run(None, inputs)
        return [str(self.labels[p]) for p in pred[0].tolist()]


def batch_evaluate(
    test_data_directory: str,
    model_directory: str,
    model_file_name: str,
    batch_size: int = 32,
) -> Dict:

    classifier = RFClassifier(
        model_directory=model_directory,
        onnx_file_name=model_file_name,
    )

    batch_labels = []
    batch_predicted = []
    batch_infer_durations = []
    batch_step = 0

    test_data = mlflow.artifacts.download_artifacts(
        os.path.join("s3://mlflow", test_data_directory)
    )
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop(labels=[FeatureConfigurations.TARGET], axis=1)
    y_test = test_data[FeatureConfigurations.TARGET]

    for i in range(0, X_test.shape[0], batch_size):
        x = X_test[i : i + batch_size]
        batch_infer_start = time.time()
        pred_label = classifier.predict_label(x)
        batch_infer_end = time.time()
        pred_number = classifier.predict(x)

        batch_labels.extend([p for p in pred_label])
        batch_predicted.extend([p for p in pred_number])
        batch_infer_durations.append(batch_infer_end - batch_infer_start)
        batch_step += 1

    batch_total_time = sum(batch_infer_durations)
    batch_infer_total_time = sum(batch_infer_durations)
    average_batch_duration_second = batch_total_time / batch_step
    average_batch_infer_second = batch_infer_total_time / batch_step

    accuracy = accuracy_score(y_test, batch_predicted)
    precision = precision_score(y_test, batch_predicted, average="macro")
    recall = recall_score(y_test, batch_predicted, average="macro")
    f1_macro = f1_score(y_test, batch_predicted, average="macro")

    evaluation = {
        "batch_size": batch_size,
        "batch_step": batch_step,
        "batch_total_time": batch_total_time,
        "batch_infer_total_time": batch_infer_total_time,
        "average_batch_duration_second": average_batch_duration_second,
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
    model_directory: str,
    model_file_name: str,
) -> Dict:

    classifier = RFClassifier(
        model_directory=model_directory,
        onnx_file_name=model_file_name,
    )

    labels = []
    predicted = []
    infer_durations = []

    test_data = mlflow.artifacts.download_artifacts(
        os.path.join("s3://mlflow", test_data_directory)
    )
    test_data = pd.read_csv(test_data)
    X_test = test_data.drop(labels=[FeatureConfigurations.TARGET], axis=1)
    y_test = test_data[FeatureConfigurations.TARGET]

    for i, row in X_test.iterrows():
        x = pd.DataFrame([row])

        infer_start = time.time()
        pred_label = classifier.predict_label(x)
        infer_end = time.time()
        pred_number = classifier.predict(x)

        labels.append(pred_label)
        predicted.append(pred_number)
        infer_durations.append(infer_end - infer_start)

    total_time = sum(infer_durations)
    total_infer_time = sum(infer_durations)
    total_tested = len(predicted)
    average_duration_second = total_time / total_tested
    average_infer_duration_second = total_infer_time / total_tested

    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, average="macro")
    recall = recall_score(y_test, predicted, average="macro")
    f1_macro = f1_score(y_test, predicted, average="macro")

    evaluation = {
        "total_step": total_tested,
        "total_time": total_time,
        "total_infer_time": total_infer_time,
        "average_duration_second": average_duration_second,
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


def main():
    parser = ArgumentParser(
        description="evaluate machine detection model",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--upstream",
        type=str,
        default="/opt/data/model/",
        help="upstream directory",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/data/evaluate/",
        help="downstream diretory",
    )
    parser.add_argument(
        "--test_parent_directory",
        type=str,
        default="/opt/data/processed/",
        help="test data directory",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    # upstream_directory = "/opt/data/model"
    upstream_directory = args.upstream
    downstream_directory = args.downstream
    # os.makedirs(upstream_directory, exist_ok=True)
    os.makedirs(downstream_directory, exist_ok=True)

    # test_parent_directory = "/opt/data/processed/"
    test_parent_directory = args.test_parent_directory

    log_file = os.path.join(downstream_directory, f"{mlflow_experiment_id}.json")
    # with open(log_file, "w") as f:
    #     json.dump(log_file, f)

    mlflow.log_artifact(log_file)

    test_data_directory = os.path.join(
        os.path.join(test_parent_directory, TrainConfigurations.TEST_PREFIX),
        TrainConfigurations.TEST_NAME,
    )

    logger.info(".... start one sample evaluate ....")
    one_sample_result = evaluate(
        test_data_directory=test_data_directory,
        model_directory=upstream_directory,
        model_file_name=ModelConfigurations.ONNX_FILE_NAME,
    )
    logger.info(".... end one sample evaluate ....")

    mlflow.log_metric(
        "total_step",
        one_sample_result["evaluation"]["total_step"],
    )
    mlflow.log_metric(
        "total_time",
        one_sample_result["evaluation"]["total_time"],
    )
    mlflow.log_metric(
        "accuracy_score",
        one_sample_result["evaluation"]["accuracy_score"],
    )
    mlflow.log_metric(
        "precision_score",
        one_sample_result["evaluation"]["precision_score"],
    )
    mlflow.log_metric(
        "recall_score",
        one_sample_result["evaluation"]["recall_score"],
    )
    mlflow.log_metric(
        "f1_score",
        one_sample_result["evaluation"]["f1_score"],
    )

    mlflow.log_metric(
        "average_duration_second",
        one_sample_result["evaluation"]["average_duration_second"],
    )

    mlflow.log_metric(
        "average_infer_duration_second",
        one_sample_result["evaluation"]["average_infer_duration_second"],
    )

    logger.info(".... start batch sample evaluate ....")
    batch_sample_result = batch_evaluate(
        test_data_directory=test_data_directory,
        model_directory=upstream_directory,
        model_file_name=ModelConfigurations.ONNX_FILE_NAME,
    )
    logger.info(".... end batch sample evaluate ....")

    mlflow.log_metric(
        "batch_size",
        batch_sample_result["evaluation"]["batch_size"],
    )
    mlflow.log_metric(
        "batch_step",
        batch_sample_result["evaluation"]["batch_step"],
    )
    mlflow.log_metric(
        "batch_total_time",
        batch_sample_result["evaluation"]["batch_total_time"],
    )
    mlflow.log_metric(
        "batch_infer_total_time",
        batch_sample_result["evaluation"]["batch_infer_total_time"],
    )
    mlflow.log_metric(
        "accuracy_score",
        batch_sample_result["evaluation"]["accuracy_score"],
    )
    mlflow.log_metric(
        "precision_score",
        batch_sample_result["evaluation"]["precision_score"],
    )
    mlflow.log_metric(
        "recall_score",
        batch_sample_result["evaluation"]["recall_score"],
    )
    mlflow.log_metric(
        "f1_score",
        batch_sample_result["evaluation"]["f1_score"],
    )

    mlflow.log_metric(
        "average_batch_duration_second",
        batch_sample_result["evaluation"]["average_batch_duration_second"],
    )

    mlflow.log_metric(
        "average_batch_infer_duration_second",
        batch_sample_result["evaluation"]["average_batch_infer_duration_second"],
    )


if __name__ == "__main__":
    main()
