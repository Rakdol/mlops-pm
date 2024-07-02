import os
import sys

from pathlib import Path
from argparse import ArgumentParser

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

print(PACKAGE_ROOT)
import mlflow
import pandas as pd
import optuna
import psycopg2

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.config import config
import src.pipeline.pipeline as pipe
from src.logger import logging
from src.exception import CustomException
from utils import get_or_create_experiment, champion_callback

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"


parser = ArgumentParser()
parser.add_argument("--exp-name", dest="exp_name", type=str)
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
args = parser.parse_args()


def get_train_data(db_connect):
    df = pd.read_sql(
        "SELECT * FROM machine_data ORDER BY timestamp DESC LIMIT 500", db_connect
    )
    return df


try:
    db_connect = psycopg2.connect(
        user="admin",
        password="1234",
        host="localhost",
        port=5432,
        database="machinedb",
    )
    train = get_train_data(db_connect)
    X_train, y_train = train.drop(config.TARGET, axis=1), train[config.TARGET]

    logging.info(f"Get Training Data from Postgresql DB: {print(db_connect)}")

except Exception as e:
    raise CustomException(e, sys)


skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# experiment_id = get_or_create_experiment("Logit Experiment 1")

# Set the current active MLflow experiment
mlflow.set_experiment(args.exp_name)
logging.info(f"Set Mlflow Experiment: {args.exp_name}")


def objective(trial, X, y, cv, scoring):

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
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    roc = scores["test_score"].mean()
    return roc


with mlflow.start_run():
    study = optuna.create_study(direction="maximize")

    func = lambda trial: objective(trial, X_train, y_train, cv=skf, scoring="roc_auc")
    study.optimize(func, n_trials=10, callbacks=[champion_callback])

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_roc", study.best_value)
    logging.info(f"Complete Oputna Optimization and Best Metric: {study.best_value}")

    # # Log tags
    # mlflow.set_tags(
    #     tags={
    #         "project": "PM Project",
    #         "optimizer_engine": "optuna",
    #         "model_family": "Logistic",
    #         "feature_set_version": 1,
    #     }
    # )

    # Log a fit model instance
    model = make_pipeline(
        pipe.input_pipeline, LogisticRegression(**study.best_params, random_state=42)
    )
    model.fit(X_train, y_train)

    logging.info("Make Pipeline and Fit the model with best parmas")

    train_pred = model.predict(X_train)

    train_accuracy = accuracy_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred)
    train_recall = recall_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred)

    # valid_pred = model.predict(X_valid)
    # valid_accuracy = accuracy_score(y_valid, valid_pred)
    # valid_precision = precision_score(y_valid, valid_pred)
    # valid_recall = recall_score(y_valid, valid_pred)
    # valid_f1 = f1_score(y_valid, valid_pred)

    signature = mlflow.models.signature.infer_signature(
        model_input=X_train, model_output=train_pred
    )
    input_sample = X_train.iloc[:10]

    mlflow.log_metrics(
        {
            "train_accuracy": train_accuracy,
            # "valid_accuracy": valid_accuracy,
            "train_precision": train_precision,
            # "valid_precision": valid_precision,
            "train_recall": train_recall,
            # "valid_recall": valid_recall,
            "train_f1": train_f1,
            # "valid_f1": valid_f1,
        }
    )

    logging.info("Save model metrics in mlflow")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=args.model_name,
        signature=signature,
        input_example=input_sample,
    )

    logging.info("Save model artifacts in mlflow")
