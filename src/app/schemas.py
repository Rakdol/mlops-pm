from datetime import datetime
from pydantic import BaseModel


class MachineData(BaseModel):
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


class MachineFailure(BaseModel):
    failure: int
