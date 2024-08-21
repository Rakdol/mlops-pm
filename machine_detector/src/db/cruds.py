import uuid
import datetime
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from db import models, schemas
from db.database import get_db


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


# def fetch_all_database_wtih_chunk(db: Session, chunk_size=10000):
#     with db() as session:

#         query = session.query(models.MachineData)
#         chunks = pd.read_sql(
#             query.statement, session.connection(), chunksize=chunk_size
#         )
#         # Process chunks
#         df_list = []
#         for chunk in chunks:
#             # Optionally process each chunk here
#             df_list.append(chunk)

#         # Combine chunks into a single DataFrame
#         df = pd.concat(df_list, ignore_index=True)
#         features = df.drop(labels=[FeatureConfigurations.TARGET], axis=1)
#         target = df[FeatureConfigurations.TARGET]

#     return features, target


# def fetch_from_database_wtih_limit(db: Session, limit=10000):
#     with db() as session:

#         query = (
#             session.query(models.MachineData)
#             .order_by(desc(models.MachineData.timestamp))
#             .limit(limit)
#         )
#         chunk = pd.read_sql(query.statement, session.connection())

#     features = chunk.drop(labels=[FeatureConfigurations.TARGET], axis=1)
#     target = chunk[FeatureConfigurations.TARGET]

#     return features, target