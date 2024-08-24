import datetime
from typing import Dict, Optional

from pydantic import BaseModel


class MachineBase(BaseModel):
    id: int
    timestamp: datetime.datetime


class MachineCreate(MachineBase):
    pass


class MachineData(MachineBase):
    product_id: str
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: float
    machine_fauilre: int
    twf: int
    hdf: int
    pwf: int
    osf: int
    rnf: int

    class Config:
        from_attributes = True


class PredictBase(BaseModel):
    id: int
    timestamp: datetime.datetime


class PredictCreate(PredictBase):
    pass


class PredictData(PredictBase):
    predict_label: str
    predict_value: str
