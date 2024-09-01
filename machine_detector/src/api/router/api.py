import uuid
import time
from logging import getLogger
from typing import Any, Dict, List

from fastapi import APIRouter
from src.infer.predict import Data, classifier

logger = getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}


@router.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "data_type": "pd.DataFrame",
        "data_structure": "(1, 12)",
        "data_sample": Data().data,
        "prediction_type": "int64",
        "prediction_structure": "(1, 1)",
        "prediction_sample": [1],
    }


@router.get("/label")
def label() -> Dict[int, str]:
    return classifier.labels


@router.get("/predict/test")
def predict_test() -> Dict[str, List[float]]:
    job_id = str(uuid.uuid4())
    prediction = classifier.predict(Data().data)
    prediction_list = list(prediction)
    logger.info(f"test {job_id}: {prediction_list}")
    return {"prediction": prediction_list}


@router.get("/predict/test/label")
def predict_test_label() -> Dict[str, List[str]]:
    job_id = str(uuid.uuid4())
    prediction = classifier.predict(Data().data)
    logger.info(f"test {job_id}: {prediction}")
    return {"prediction": prediction}


@router.post("/predict")
def predict(data: Data) -> Dict[str, List[int]]:
    job_id = str(uuid.uuid4())
    prediction = classifier.predict(data.data)
    prediction_list = list(prediction)
    logger.info(f"{job_id}: {prediction_list}")
    return {"prediction": prediction_list}


@router.post("/predict/label")
def predict_label(data: Data) -> Dict[str, List[str]]:
    job_id = str(uuid.uuid4())
    prediction = classifier.predict_label(data.data)
    logger.info(f"test {job_id}: {prediction}")
    return {"prediction": prediction}
