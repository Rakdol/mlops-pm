import os
import sys
from pathlib import Path
from typing import List, Dict, Union

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class SkModelTrainer(object):
    def __init__(self, models: Dict[str, BaseEstimator]) -> None:

        self.models = models
        self.trained_models: Dict[str, BaseEstimator] = {}
        self.best_params: Dict[str, Union[int, float, str]] = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        name: str,
        params: dict,
    ):

        model = self.models[name]
        model.set_params(**params)

        model.fit(X_train, y_train)

        return model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: BaseEstimator,
        model_type: str = "classification",
        average: str = "macro",
    ) -> pd.DataFrame:

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
                    else metric(y_valid, y_pred, average=average)
                )
                for metric in metrics
            }
        elif model_type == "regression":
            results[model.__name__] = {
                metric.__name__: metric(y_test, y_pred) for metric in metrics
            }

        return pd.DataFrame(results)

    def cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
        stratified: bool = False,
        shuffle: bool = True,
    ) -> pd.DataFrame:

        results = {}

        if stratified:
            kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=42)

        else:
            kfold = KFold(n_splits=cv, shuffle=shuffle, random_state=42)

        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
            results[name] = scores

        return pd.DataFrame(results)

    def evaluate_models(
        self,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        model_type: str = "classification",
        average: str = "macro",
    ) -> pd.DataFrame:

        if model_type == "classification":
            metrics = [accuracy_score, precision_score, recall_score, f1_score]
        if model_type == "regression":
            metrics = [
                root_mean_squared_error,
                mean_squared_error,
                mean_absolute_error,
                r2_score,
            ]

        results = {}

        for name, model in self.trained_models.items():
            y_pred = model.predict(X_valid)
            if model_type == "classification":
                results[name] = {
                    metric.__name__: (
                        metric(y_valid, y_pred)
                        if metric.__name__ == "accuracy_score"
                        else metric(y_valid, y_pred, average=average)
                    )
                    for metric in metrics
                }
            elif model_type == "regression":
                results[name] = {
                    metric.__name__: metric(y_valid, y_pred) for metric in metrics
                }

        return pd.DataFrame(results)

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grids: Dict[str, Dict[str, List[Union[int, float, str]]]],
        cv: int = 5,
        scoring: str = "accuracy",
        stratified: bool = False,
        shuffle: bool = True,
    ) -> None:

        if stratified:
            kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=42)

        else:
            kfold = KFold(n_splits=cv, shuffle=shuffle, random_state=42)

        for name, model in self.models.items():
            if name not in param_grids:
                continue

            param_grid = param_grids[name]
            search = GridSearchCV(model, param_grid, cv=kfold, scoring=scoring)

            search.fit(X, y)

            self.trained_models[name] = search.best_estimator_
            self.best_params[name] = search.best_params_


if __name__ == "__main__":

    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    digits = load_digits()

    X = digits.data
    y = digits.target

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
    }

    trainer = SkModelTrainer(models=models)
    cross_result = trainer.cross_validation(X_train, y_train, cv=5)
    param_grids = {
        "LogisticRegression": {"max_iter": [100, 500, 1000]},
        "DecisionTreeClassifier": {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [3, 5, 7, 9],
        },
        "RandomForestClassifier": {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [3, 5, 7, 9],
            "n_estimators": [8, 16, 32, 64, 128, 256],
        },
    }

    trainer.tune_hyperparameters(X_train, y_train, param_grids=param_grids, cv=3)

    eval_report = trainer.evaluate_models(
        X_valid=X_valid,
        y_valid=y_valid,
        model_type="classification",
        average="macro",
    )

    model = trainer.train(
        X_train,
        y_train,
        name="RandomForestClassifier",
        params=trainer.best_params["RandomForestClassifier"],
    )

    evaluations = trainer.evaluate(X_valid, y_valid, model)
    k = 1
