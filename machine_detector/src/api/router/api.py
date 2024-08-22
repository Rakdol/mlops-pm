import onnxruntime as ort
import numpy as np

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.db import cruds, schemas
from src.db.database import get_db


router = APIRouter()


# ONNX 모델 로드
ort_session = ort.InferenceSession("model.onnx")


@router.get("/projects/all")
async def predict(input_data: dict):
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    data = np.array(input_data["input_data"]).astype(np.float32)
    data = data.reshape((data.shape[0], 28, 28))

    # 추론 실행
    outputs = ort_session.run(None, {input_name: data})

    return {"predictions": outputs[0].tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
