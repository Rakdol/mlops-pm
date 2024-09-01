import os
import sys
import datetime
from argparse import ArgumentParser, RawTextHelpFormatter

import mlflow
from mlflow import MlflowClient

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.logger import logging

import pandas as pd


def main():
    parser = ArgumentParser(
        description="Main Pipeline", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--preprocess_data",
        type=str,
        default="machine",
        help="machine failure detection data",
    )

    parser.add_argument(
        "--preprocess_fetch",
        type=str,
        default="db",
        help="local raw data or database are used to process data",
    )

    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="./data/preprocess",
        help="preprocess downstream directory",
    )

    parser.add_argument(
        "--preprocess_cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )

    parser.add_argument(
        "--train_upstream",
        type=str,
        default="../data/preprocess",
        help="upstream directory",
    )

    parser.add_argument(
        "--train_downstream",
        type=str,
        default="../data/model",
        help="downstream directory",
    )

    parser.add_argument(
        "--train_model_type",
        type=str,
        default="rf",
        help="random forest",
    )
    parser.add_argument(
        "--train_cv_type", type=str, default="strat_cv", help="stratified CV"
    )

    parser.add_argument("--train_n_split", type=int, default=5, help="Train CV n_split")
    parser.add_argument(
        "--train_n_trials", type=int, default=2, help="Optuna hyperparameters n_trials"
    )

    parser.add_argument(
        "--evaluate_downstream",
        type=str,
        default="../data/evaluate/",
        help="evaluate downstram directory",
    )

    args = parser.parse_args()

    logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    with mlflow.start_run() as run:
        logging.info(f"run_id: {run.info.run_id}")
        logging.info("description: {}".format(run.data.tags.get("mlflow.note.content")))
        logging.info("version tag value: {}".format(run.data.tags.get("version")))
        logging.info("priority tag value: {}".format(run.data.tags.get("priority")))

        logging.info(f".... Preprocess MLproject start ....")
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            parameters={
                "data": args.preprocess_data,
                "fetch": args.preprocess_fetch,
                "downstream": args.preprocess_downstream,
                "cached_data_id": args.preprocess_cached_data_id,
            },
        )
        logging.info(
            f"""
                     Preprocess ML project has been completed, 
                     data: {args.preprocess_data},
                     fetch: {args.preprocess_fetch},
                     downstream: {args.preprocess_downstream},
                     cached_data_id: {args.preprocess_cached_data_id},
                     """
        )

        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)

        train_upstream = os.path.join(
            "",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )

        logging.info(f".... Train MLproject start ....")
        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            parameters={
                "upstream": train_upstream,
                "downstream": args.train_downstream,
                "model_type": args.train_model_type,
                "cv_type": args.train_cv_type,
                "n_split": args.train_n_split,
                "n_trials": args.train_n_trials,
            },
        )
        logging.info(
            f"""
                     Train ML project has been completed, 
                     upstream: {train_upstream},
                     downstream: {args.train_downstream},
                     model_type: {args.train_model_type},
                     cv_type: {args.train_cv_type},
                     n_split: {args.train_n_split},
                     n_trials: {args.train_n_trials},
                     """
        )
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

        evaluate_upstream = os.path.join(
            "",
            str(mlflow_experiment_id),
            train_run.info.run_id,
            "artifacts",
        )

        logging.info(f".... Evaluate MLproject start ....")
        evaluate_run = mlflow.run(
            uri="./evaluate",
            entry_point="evaluate",
            backend="local",
            parameters={
                "upstream": evaluate_upstream,
                "downstream": args.evaluate_downstream,
                "test_parent_directory": train_upstream,
            },
        )

        logging.info(
            f"""
                     Evaluate ML project has been completed, 
                     upstream: {evaluate_upstream},
                     downstream: {args.evaluate_downstream},
                     test_parent_directory: {train_upstream}
                     """
        )
        evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run.run_id)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        mlflow.set_tag("Release Date", value=current_date)
        mlflow.set_tag("Release Model", value="RandomForest")


if __name__ == "__main__":
    main()
