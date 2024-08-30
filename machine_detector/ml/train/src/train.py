import os
from argparse import ArgumentParser, RawTextHelpFormatter
import logging
import joblib

import mlflow
import numpy as np
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from src.configurations import TrainConfigurations, FeatureConfigurations
from src.constants import MODEL_ENUM, CV_ENUM
from src.model import MachineDataset, tune_hyperparameters, train, evaluate
from src.objectives import rf_objective, logit_objective
from src.transformers import get_input_pipeline
from src.bucket import bucket_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    model_type: str,
    cv_type: str,
    n_split: int,
    n_trials: int,
):

    train_set = MachineDataset(
        upstream_directory=upstream_directory,
        file_prefix=TrainConfigurations.TRAIN_PREFIX,
        file_name=TrainConfigurations.TRAIN_NAME,
        pipe_prefix=TrainConfigurations.PIPE_PREFIX,
        pipe_name=TrainConfigurations.PIPE_NAME,
        bucket_client=bucket_client,
    )
    input_pipe = train_set.get_input_pipe()

    X_train, y_train = train_set.pandas_reader_dataset(
        target_col=FeatureConfigurations.TARGET
    )

    valid_set = MachineDataset(
        upstream_directory=upstream_directory,
        file_prefix=TrainConfigurations.VALID_PREFIX,
        file_name=TrainConfigurations.VALID_NAME,
        pipe_prefix=TrainConfigurations.PIPE_PREFIX,
        pipe_name=TrainConfigurations.PIPE_NAME,
        bucket_client=bucket_client,
    )

    X_valid, y_valid = valid_set.pandas_reader_dataset(
        target_col=FeatureConfigurations.TARGET
    )

    test_set = MachineDataset(
        upstream_directory=upstream_directory,
        file_prefix=TrainConfigurations.TEST_PREFIX,
        file_name=TrainConfigurations.TEST_NAME,
        pipe_prefix=TrainConfigurations.PIPE_PREFIX,
        pipe_name=TrainConfigurations.PIPE_NAME,
        bucket_client=bucket_client,
    )

    X_test, y_test = test_set.pandas_reader_dataset(
        target_col=FeatureConfigurations.TARGET
    )

    if model_type == MODEL_ENUM.LOGIT_MODEL.value:
        objective_func = logit_objective
        model = LogisticRegression()
    elif model_type == MODEL_ENUM.RF_MODEL.value:
        objective_func = rf_objective
        model = RandomForestClassifier()
    else:
        raise ValueError("Unknown Model")

    if cv_type == CV_ENUM.simple_CV.value:
        cv = KFold(n_splits=n_split, shuffle=True, random_state=42)
    elif cv_type == CV_ENUM.strat_cv.value:
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    else:
        raise ValueError("Unknown CV")

    study = tune_hyperparameters(
        objective=objective_func,
        X=X_train,
        y=y_train,
        cv=cv,
        pipe=input_pipe,
        n_trials=n_trials,
        direction="maximize",
        scoring="roc_auc",
    )

    logger.info("get hyperparameters using optuna")

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_roc", study.best_value)

    trained_model, valid_metric = train(
        model=model,
        pipe=input_pipe,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        model_type="classification",
        params=study.best_params,
    )

    logger.info("model training complete")

    test_metric = evaluate(
        trained_model,
        input_pipe,
        X_test=X_test,
        y_test=y_test,
        model_type="classification",
    )

    signature = mlflow.models.signature.infer_signature(
        input_pipe.transform(X_train),
        trained_model.predict(input_pipe.transform(X_train)),
    )
    input_sample = input_pipe.transform(X_train)[:5]

    mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="model",
        signature=signature,
        input_example=input_sample,
    )

    model_file_name = os.path.join(
        downstream_directory,
        f"machine_{model_type}_{mlflow_experiment_id}.joblib",
    )
    onnx_file_name = os.path.join(
        downstream_directory,
        f"machine_{model_type}_{mlflow_experiment_id}.onnx",
    )
    pipe_file_name = os.path.join(
        downstream_directory, f"machine_input_pipeline_{mlflow_experiment_id}.joblib"
    )

    initial_type = [
        (
            "float_input",
            FloatTensorType([None, input_pipe.transform(X_train[:1]).shape[1]]),
        )
    ]
    onx = skl2onnx.to_onnx(
        trained_model,
        initial_type,
        target_opset=12,
    )

    joblib.dump(trained_model, model_file_name)
    joblib.dump(input_pipe, pipe_file_name)

    with open(onnx_file_name, "wb") as f:
        f.write(onx.SerializeToString())

    mlflow.log_metrics(
        {key: test_metric.loc[key].to_numpy()[0] for key in test_metric.index}
    )
    mlflow.log_artifact(model_file_name)
    mlflow.log_artifact(onnx_file_name)
    mlflow.log_artifact(pipe_file_name)
    logger.info("Save model metrics in mlflow")


def main():
    parser = ArgumentParser(
        description="Train machine detector",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--upstream",
        type=str,
        default="/opt/data/processed",
        help="upstream directory",
    )

    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/machine/model/",
        help="downstream directory",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_ENUM.RF_MODEL.value,
        help="random forest model",
    )

    parser.add_argument(
        "--cv_type",
        type=str,
        default=CV_ENUM.strat_cv.value,
        help="Stratified CV",
    )

    parser.add_argument("--n_split", type=int, default=5, help="CV's n_split")

    parser.add_argument(
        "--n_trials",
        type=int,
        default=2,
        help="Optuna hyperparameters tuning with n_trials",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = args.upstream
    downstream_directory = args.downstream
    os.makedirs(downstream_directory, exist_ok=True)

    start_run(
        mlflow_experiment_id=mlflow_experiment_id,
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        model_type=args.model_type,
        cv_type=args.cv_type,
        n_split=args.n_split,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
