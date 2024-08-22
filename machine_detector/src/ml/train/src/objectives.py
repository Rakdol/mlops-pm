import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline


def rf_objective(
    trial: optuna.trial.Trial,
    X: np.ndarray,
    y: np.ndarray,
    pipe: Pipeline,
    cv: BaseCrossValidator,
    scoring: str,
) -> float:
    # Define the hyperparameters to be tuned
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30, step=5),
        "min_samples_split": trial.suggest_int("min_samples_split", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 10),
        "n_jobs": -1,
    }

    # Create a pipeline with the input preprocessing and the classifier
    model = make_pipeline(pipe, RandomForestClassifier(**params, random_state=42))

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    mean_score = scores.mean()

    return mean_score


def logit_objective(
    trial: optuna.trial.Trial,
    X: np.ndarray,
    y: np.ndarray,
    pipe: Pipeline,
    cv: BaseCrossValidator,
    scoring: str,
) -> float:

    params = {
        "tol": trial.suggest_uniform("tol", 1e-6, 1e-3),
        "C": trial.suggest_loguniform("C", 1e-2, 1),
        "n_jobs": -1,
    }

    # model = LogisticRegression(**params, random_state=42)
    model = make_pipeline(
        pipe.input_pipeline,
        LogisticRegression(**params, random_state=42, max_iter=3000),
    )

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    mean_score = scores.mean()

    return mean_score
