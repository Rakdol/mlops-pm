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
from dotenv import load_dotenv


from schemas import MachineData, MachineFailure

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_DIR = os.path.join(PACKAGE_ROOT.parent, "config")
load_dotenv(os.path.join(BASE_DIR, ".env"))

model_path = os.path.join(PACKAGE_ROOT.parent.parent, "artifacts")


def load_model():
    model = mlflow.pyfunc.load_model(model_uri=model_path)
    return model


MODEL = load_model()
app = FastAPI()


@app.get("/ping")
async def pong():
    return {"message": "ping"}


@app.get("/")
async def index():
    return {"message": "Manchine Failure Prediction App"}


@app.post("/predict", response_model=MachineFailure)
async def predict(data: MachineData) -> MachineFailure:
    df = pd.DataFrame([data.model_dump()])
    pred = MODEL.predict(df).item()
    return MachineFailure(failure=pred)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
