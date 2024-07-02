from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
import joblib
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime

app = FastAPI()


class MachineData(BaseModel):
    timestamp: datetime
    product_id: str
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: float
    machine_failure: float
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int


@app.get("/ping")
def pong():
    return {"message": "ping"}


@app.get("/")
def index():
    return {"message": "Manchine Failure Prediction App"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
