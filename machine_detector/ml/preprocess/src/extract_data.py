import json
import joblib
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.configurations import FeatureConfigurations
from db import models


def fetch_data_from_local(filepath: str):
    df = pd.read_csv(filepath)

    features = df.drop(labels=[FeatureConfigurations.TARGET], axis=1)
    target = df[FeatureConfigurations.TARGET]

    return features, target


def fetch_all_database_wtih_chunk(db: Session, chunk_size=10000):
    with db() as session:

        query = session.query(models.MachineData)
        chunks = pd.read_sql(
            query.statement, session.connection(), chunksize=chunk_size
        )
        # Process chunks
        df_list = []
        for chunk in chunks:
            # Optionally process each chunk here
            df_list.append(chunk)

        # Combine chunks into a single DataFrame
        df = pd.concat(df_list, ignore_index=True)
        features = df.drop(labels=[FeatureConfigurations.TARGET], axis=1)
        target = df[FeatureConfigurations.TARGET]

    return features, target


def fetch_from_database_wtih_limit(db: Session, limit=10000):
    with db() as session:

        query = (
            session.query(models.MachineData)
            .order_by(desc(models.MachineData.timestamp))
            .limit(limit)
        )
        chunk = pd.read_sql(query.statement, session.connection())

    features = chunk.drop(labels=[FeatureConfigurations.TARGET], axis=1)
    target = chunk[FeatureConfigurations.TARGET]

    return features, target


def save_to_csv(
    data: Union[np.array, pd.DataFrame],
    destination: str,
    name_prefix: str,
    header: Optional[str],
) -> None:
    save_dest = Path(destination)
    filename_format = f"{name_prefix}_dataset.csv"
    csv_path = save_dest / filename_format
    df = pd.DataFrame(data, columns=header.split(","))
    df.to_csv(csv_path, index=False)


def save_object(
    object_file, destination: str, object_name: str, header: Optional[str]
) -> None:

    object_dir = Path(destination)
    save_path = str(object_dir / object_name)

    joblib.dump(object_file, save_path)

    if header is not None:
        headers = {i: col for i, col in enumerate(header.split(","))}
        header_filename = "object_labels.json"
        header_path = object_dir / header_filename
        with open(header_path, "w") as f:
            json.dump(headers, f)
