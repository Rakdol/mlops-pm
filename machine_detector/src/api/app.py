import os
import sys
from pathlib import Path
import mlflow
from dotenv import load_dotenv
from argparse import ArgumentParser


# 환경 변수 설정
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_DIR = os.path.join(PACKAGE_ROOT.parent, "config")
load_dotenv(os.path.join(BASE_DIR, ".env"))


MODEL_DIR = os.path.join(PACKAGE_ROOT.parent.parent, "artifacts/")
model_name = "sk-logits"
stage = "Production"

print(MODEL_DIR)


def download_model(model_name, stage):
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/{stage}",
        dst_path=MODEL_DIR,
    )


# download_model(model_name, stage)

model = mlflow.pyfunc.load_model(model_uri=MODEL_DIR)
print(model)
