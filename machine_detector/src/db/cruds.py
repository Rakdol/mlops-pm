import uuid
import datetime
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from db import models, schemas


def select_machine_all(db: Session) -> List[schemas.MachineData]:
    return db.query(models.MachineData).all()


def select_machine_by_product_id(
    db: Session, product_id: str
) -> List[schemas.MachineData]:
    return (
        db.query(models.MachineData)
        .filter(models.MachineData.product_id == product_id)
        .all()
    )


def select_machine_by_type(db: Session, machine_type: str) -> List[schemas.MachineData]:
    return (
        db.query(models.MachineData)
        .filter(models.MachineData.machine_type == machine_type)
        .all()
    )

def select_machine_from_btw_time(
    db: Session, start_time: datetime.datetime, end_time: datetime.datetime
) -> List[schemas.MachineData]:
    return (
        db.query(models.MachineData)
        .filter(models.MachineData.timestamp.between(start_time, end_time))
        .all()
    )
