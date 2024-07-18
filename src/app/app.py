import uuid
from typing import Dict, List, Any


from fastapi import FastAPI

from ..logger import logging
from src.pipeline.prediction import classifier, Data


app = FastAPI()

@app.get("/ping")
def ping()->Dict[str,str]:
    return {"pong": "ok"}


@app.get(".metadata")
def metadata() -> Dict[str, Any]:
    return {
        "data_type": "float32",
        "data_structure": "(1,4)",
        "data_sample": "",
        "prediction_type": "float32",
        "prediction_structure": "(1,3)",
        "prediction_sample": [0.97093159, 0.01558308, 0.01348537],
    }
    

@app.get("/label")
def label() -> Dict[int, str]:
    return classifier.label


@app.get("/predict/test")
def predict_test() -> Dict[str, List[float]]:
    job_id = str(uuid.uuid4)
    prediction = classifier.predict(Data().data)
    prediction_list = list(prediction)
    logging.info(f"test {job_id}: {prediction_list}")
    return {"prediction": prediction_list}


@app.post("/predict")
def predict(data):
    job_id = str(uuid.uuid4())
    prediction = classifier.predict(data.data)
    prediction_list = list(prediction)
    logging.info(f"{job_id}: {prediction_list}")
    return {"prediction": prediction_list}

@app.post("/predict/label")
def predict_label(data: Data) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    prediction = classifier.predict_label(data.data)
    logging.info(f"test {job_id}: {prediction}")
    return {"prediction": prediction}
