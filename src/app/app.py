import os
import sys
from datetime import datetime
from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd
import mlflow.pyfunc

from pathlib import Path
from fastapi import FastAPI
import cloudpickle
from dotenv import load_dotenv

from schemas import MachineData, MachineFailure

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_DIR = os.path.join(PACKAGE_ROOT.parent, "config")
load_dotenv(os.path.join(BASE_DIR, ".env"))

# MODEL_DIR = os.path.join(PACKAGE_ROOT.parent.parent, "artifacts")
# print(MODEL_DIR)


MODEL_DIR = "/home/moon/project/mlops-pm/artifacts"


def download_model_artifacts(model_name, stage, model_dir):
    artifact_uri = f"models:/{model_name}/{stage}"
    mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri,
        dst_path=model_dir,
    )


def load_model_from_directory(model_dir):
    model = mlflow.pyfunc.load_model(model_uri=model_dir, pickle_module=cloudpickle)
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


# MODEL = load_model()
# app = FastAPI()


# @app.get("/ping")
# async def pong():
#     return {"message": "ping"}


# @app.get("/")
# async def index():
#     return {"message": "Manchine Failure Prediction App"}


# @app.post("/predict", response_model=MachineFailure)
# async def predict(data: MachineData) -> MachineFailure:
#     df = pd.DataFrame([data.model_dump()])
#     pred = MODEL.predict(df).item()
#     return MachineFailure(failure=pred)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
