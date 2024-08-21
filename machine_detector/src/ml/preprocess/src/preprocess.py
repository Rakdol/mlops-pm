import os
from argparse import ArgumentParser, RawTextHelpFormatter

from distutils.dir_util import copy_tree

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from src.configurations import (
    DataConfigurations,
    FeatureConfigurations,
)

from src.transformers import get_input_pipeline
from src.extract_data import (
    fetch_data_from_local,
    save_object,
    save_to_csv,
)


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
        "--local",
        type=bool,
        default=True,
        help="Collect Data from local disk",
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

        if args.local:
            features, target = fetch_data_from_local(DataConfigurations.LOCAL_FILE_PATH)

        pipe = get_input_pipeline()
        transformed_features = pipe.fit_transform(features)

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            transformed_features, target, random_state=42
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42
        )

        train_data = np.c_[X_train.values, y_train.values].astype("float32")
        valid_data = np.c_[X_valid.values, y_valid.values].astype("float32")
        test_data = np.c_[X_test.values, y_test.values].astype("float32")
        header_cols = pipe.get_feature_names_out() + [FeatureConfigurations.TARGET]
        header = ",".join(header_cols)

        save_to_csv(train_data, train_output_destination, "train", header)
        save_to_csv(valid_data, valid_output_destination, "validation", header)
        save_to_csv(test_data, test_output_destination, "test", header)
        save_object(pipe, pipe_output_destination, "pipe.pkl", header)

        mlflow.log_artifacts(downstream_directory, artifact_path="downstream_directory")


if __name__ == "__main__":
    main()
