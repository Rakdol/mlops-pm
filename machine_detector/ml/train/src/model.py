import os
import sys
import time
import joblib
import datetime

from pathlib import Path
from io import StringIO, BytesIO
from typing import Optional
from logging import getLogger

import boto3
import optuna
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from sklearn.pipeline import make_pipeline, Pipeline

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
from src.utils import champion_callback

logger = getLogger(__name__)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class MachineDataset(object):
    def __init__(
        self,
        upstream_directory: str,
        file_prefix: str,
        file_name: str,
        bucket_client: Optional[boto3.client],
    ):
        self.upstream_directory = upstream_directory
        self.file_prefix = file_prefix
        self.file_name = file_name
        self.client = bucket_client

    def pandas_reader_dataset(self, target_col: str, seed=42):
        filepaths = str(
            Path() / self.upstream_directory / self.file_prefix / self.file_name
        )
        try:
            response = bucket_client.get_object(Bucket="mlflow", Key=filepaths)
            csv_data = response["Body"].read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_data))
        except Exception as e:
            logger.debug(f"dataset cannot be loaded Exceptions {e}")
            raise Exception(e, sys)

        features = df.drop(labels=[target_col], axis=1)
        target = df[target_col]

        return features, target


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
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
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_type: str,
    params: Optional[dict] = None,
):
    if params is not None:
        model.set_params(**params)

    model = Pipeline([("preprocessor", pipe), ("classifier", model)])
    model.fit(X_train, y_train)
    eval_result = evaluate(model, X_valid, y_valid, model_type)

    logger.info(f"model trained")
    return model, eval_result


def tune_hyperparameters(
    objective,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    pipe: Pipeline,
    n_trials: int = 5,
    direction: str = "maximize",
    scoring: str = "roc_auc",
):
    study = optuna.create_study(direction=direction)
    func = lambda trial: objective(trial, X=X, y=y, pipe=pipe, cv=cv, scoring=scoring)
    study.optimize(func, n_trials=n_trials, callbacks=[champion_callback])

    return study
