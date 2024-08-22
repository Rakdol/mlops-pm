import os
import sys
import time
import joblib
import datetime

from pathlib import Path
from io import StringIO, BytesIO
from typing import Optional, List, Dict
from logging import getLogger

import boto3
import mlflow
import optuna
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_score,
    BaseCrossValidator,
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)

from src.transformers import get_input_pipeline
from src.bucket import bucket_client
from src.configurations import FeatureConfigurations
from src.utils import get_or_create_experiment, champion_callback

logger = getLogger(__name__)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class MachineDataset(object):
    def __init__(
        self,
        upstream_directory: str,
        file_prefix: str,
        file_name: str,
        pipe_prefix: str,
        pipe_name: Optional[str],
        minio_client: Optional[boto3.client],
    ):
        self.upstream_directory = upstream_directory
        self.file_prefix = file_prefix
        self.file_name = file_name
        self.pipe_prefix = pipe_prefix
        self.pipe_name = pipe_name
        self.client = minio_client

    def get_input_pipe(self):
        pipe_path = str(
            Path() / self.upstream_directory / self.pipe_prefix / self.pipe_name
        )
        try:
            response = self.client.get_object(Bucket="mlflow", Key=pipe_path)
            pkl_data = response["Body"].read()
            pipe = joblib.load(BytesIO(pkl_data))

        except Exception as e:
            logger.debug(f"Pipeline cannot be loaded Exceptions {e}")
            raise Exception(e, sys)

        return pipe

    def pandas_reader_dataset(self, target_col: str, seed=42):
        filepaths = str(
            Path() / self.upstream_directory / self.file_prefix / self.file_name
        )
        print(filepaths)
        try:
            response = minio_client.get_object(Bucket="mlflow", Key=filepaths)
            csv_data = response["Body"].read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_data))
        except Exception as e:
            logger.debug(f"dataset cannot be loaded Exceptions {e}")
            raise Exception(e, sys)

        features = df.drop(labels=[target_col], axis=1)
        target = df[target_col]

        return features, target


def evaluate(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    average: str = "macro",
):
    if model_type == "classification":
        metrics = [accuracy_score, precision_score, recall_score, f1_score]
    if model_type == "regression":
        metrics = [
            root_mean_squared_error,
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        ]

    y_pred = model.predict(X_test)
    results = {}
    if model_type == "classification":
        results[model.__class__.__name__] = {
            metric.__name__: (
                metric(y_test, y_pred)
                if metric.__name__ == "accuracy_score"
                else metric(y_test, y_pred, average=average)
            )
            for metric in metrics
        }

    elif model_type == "regression":
        results[model.__name__] = {
            metric.__name__: metric(y_test, y_pred) for metric in metrics
        }

    return pd.DataFrame(results)


def train(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_type: str,
    params: Optional[dict],
):
    if params is not None:
        model.set_params(**params)

    model.fit(X_train, y_train)
    eval_result = evaluate(model, X_valid, y_valid, model_type)
    
    mlflow.log_metrics({key: eval_result.loc[key].to_numpy()[0] for key in eval_result.index})

    return model


def tune_hyperparameters(model:BaseEstimator, cv:BaseCrossValidator, pipe:Pipeline, objective, n_split=5):



if __name__ == "__main__":

    train_set = MachineDataset(
        upstream_directory="0/bcba77b442a743c6a7bf9debe8b855f5/artifacts/downstream_directory",
        file_prefix="train",
        file_name="train_dataset.csv",
        pipe_prefix="pipe",
        pipe_name="pipe.joblib",
        minio_client=minio_client,
    )

    train_x, train_y = train_set.pandas_reader_dataset(FeatureConfigurations.TARGET)
    print(train_x.head(), train_y.head())
