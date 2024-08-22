import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

import mlflow

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.logger import logging

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
        default='db',
        help="local raw data or database are used to process data",
    )

    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="./data/processed",
        help="preprocess downstream directory",
    )

    parser.add_argument(
        "--preprocess_cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )

    args = parser.parse_args()
    logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    with mlflow.start_run():
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
        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)


if __name__ == "__main__":
    main()
