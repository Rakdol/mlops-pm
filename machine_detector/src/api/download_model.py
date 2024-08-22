import os
import mlflow

from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_DIR = os.path.join(PACKAGE_ROOT.parent, "config")
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL_DIR = os.path.join(PACKAGE_ROOT, "artifacts/")


def download_model(args):
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{args.model_name}/{args.stage}",
        dst_path=MODEL_DIR,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model-name", dest="model_name", type=str)
    parser.add_argument("--stage", dest="stage", type=str, default="Production")
    args = parser.parse_args()
    download_model(args)
