import os
from logging import getLogger
from argparse import ArgumentParser, RawTextHelpFormatter

from distutils.dir_util import copy_tree

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from db.database import SessionLocal
from src.configurations import (
    DataConfigurations,
    FeatureConfigurations,
)
from src.transformers import get_input_pipeline
from src.extract_data import (
    fetch_data_from_local,
    fetch_from_database_wtih_limit,
    save_object,
    save_to_csv,
)

logger = getLogger(__name__)


def main():
    parser = ArgumentParser(
        description="Make Dataset", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=str,
        default="machine data",
        help="machine failure detection dataset",
    )

    parser.add_argument(
        "--fetch",
        type=str,
        default="db",
        help="Collect Data from local disk or database",
    )

    parser.add_argument(
        "--downstream",
        type=str,
        default="../data/processed/",
        help="downstream directory",
    )
    parser.add_argument(
        "--cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )

    args = parser.parse_args()

    downstream_directory = args.downstream

    if args.cached_data_id:
        cached_artifact_directory = os.path.join(
            "/tmp/mlruns/0",
            args.cached_data_id,
            "artifacts/downstream_directory",
        )
        copy_tree(
            cached_artifact_directory,
            downstream_directory,
        )
    else:
        train_output_destination = os.path.join(downstream_directory, "train")
        valid_output_destination = os.path.join(downstream_directory, "validation")
        test_output_destination = os.path.join(downstream_directory, "test")
        pipe_output_destination = os.path.join(downstream_directory, "pipe")

        os.makedirs(downstream_directory, exist_ok=True)
        os.makedirs(train_output_destination, exist_ok=True)
        os.makedirs(valid_output_destination, exist_ok=True)
        os.makedirs(test_output_destination, exist_ok=True)
        os.makedirs(pipe_output_destination, exist_ok=True)

        if args.fetch == "local":
            features, target = fetch_data_from_local(DataConfigurations.LOCAL_FILE_PATH)
            print("Dataset loaded from the local disk")
            logger.info("Dataset loaded from the local disk")
        else:
            features, target = fetch_from_database_wtih_limit(db=SessionLocal)
            datetime = features["timestamp"]
            features = features.drop(labels=["timestamp"], axis=1)
            print(features.head())
            print("Dataset loaded from the database")
            logger.info("Dataset loaded from the database")

        pipe = get_input_pipeline()
        pipe.fit(features)

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            features, target, random_state=42
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42
        )

        train_data = pd.concat([X_train, y_train], axis=1)
        valid_data = pd.concat([X_valid, y_valid], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        header_cols = features.columns.to_list() + [FeatureConfigurations.TARGET]
        header = ",".join(header_cols)

        save_to_csv(train_data, train_output_destination, "train", header)
        save_to_csv(valid_data, valid_output_destination, "validation", header)
        save_to_csv(test_data, test_output_destination, "test", header)
        save_object(pipe, pipe_output_destination, "pipe.joblib", header)

        mlflow.log_artifacts(downstream_directory, artifact_path="downstream_directory")

        # dataset = mlflow.data.from_pandas(
        #     train_data,
        #     name="machine_failure",
        #     targets="machine_failure",
        # )
        # mlflow.log_input(dataset, context="preprocess")


if __name__ == "__main__":
    main()
