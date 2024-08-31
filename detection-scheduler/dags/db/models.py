from sqlalchemy import (
    Column,
    DateTime,
    String,
    Integer,
    Float,
    CHAR,
)
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from db.database import BaseSource, BaseEvent


class MachineData(BaseSource):
    __tablename__ = "machine_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(
        DateTime(timezone=True), server_default=current_timestamp(), nullable=False
    )

    product_id = Column(
        String(20),
        nullable=True,
    )

    machine_type = Column(
        String(2),
        nullable=True,
    )

    air_temperature = Column(
        Float,
        nullable=True,
    )

    process_temperature = Column(
        Float,
        nullable=True,
    )

    rotational_speed = Column(
        Integer,
        nullable=True,
    )

    torque = Column(
        Float,
        nullable=True,
    )

    tool_wear = Column(
        Float,
        nullable=True,
    )

    machine_failure = Column(
        Integer,
        nullable=True,
    )

    twf = Column(
        Integer,
        nullable=True,
    )

    hdf = Column(
        Integer,
        nullable=True,
    )

    pwf = Column(
        Integer,
        nullable=True,
    )

    osf = Column(
        Integer,
        nullable=True,
    )

    rnf = Column(
        Integer,
        nullable=True,
    )


class MachinePredict(BaseEvent):
    __tablename__ = "machine_predict"

    id = Column(Integer, primary_key=True)
    source_time = Column(
        DateTime(timezone=True), server_default=current_timestamp(), nullable=False
    )
    predict_time = Column(
        DateTime(timezone=True), server_default=current_timestamp(), nullable=False
    )

    predict_label = Column(
        String(20),
        nullable=True,
    )
    predict_value = Column(
        String(5),
        nullable=True,
    )