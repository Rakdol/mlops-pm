import mlflow
import os
from pathlib import Path
from dotenv import load_dotenv

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_DIR = os.path.join(PACKAGE_ROOT, "src/config")
load_dotenv(os.path.join(BASE_DIR, ".env"))


MODEL_DIR = "/home/moon/project/mlops-pm/artifacts"


def download_model_artifacts(model_name, stage, model_dir):
    artifact_uri = f"models:/{model_name}/{stage}"
    mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri,
        dst_path=model_dir,
    )


def load_model_from_directory(model_dir):
    model = mlflow.pyfunc.load_model(model_uri=model_dir)
    return model


if __name__ == "__main__":
    model_name = "sk-learn-random-forest-clf-model"
    stage = "Production"

    # Download the model artifacts
    download_model_artifacts(model_name, stage, MODEL_DIR)

    # Check the contents of the downloaded artifacts
    for root, dirs, files in os.walk(MODEL_DIR):
        print(f"Root: {root}")
        for file in files:
            print(f"File: {file}")

    # Load the model
    try:
        model = load_model_from_directory(MODEL_DIR)
        print("Model loaded successfully:", model)
    except Exception as e:
        print(f"Error loading model: {e}")
