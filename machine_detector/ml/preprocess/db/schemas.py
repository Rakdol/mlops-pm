import datetime
from typing import Dict, Optional

from pydantic import BaseModel


class MachineBase(BaseModel):
    pass


class MachineCreate(MachineBase):
    pass


class MachineData(BaseModel):
    id: int
    timestamp: datetime.datetime
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
