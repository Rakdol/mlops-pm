import json
import cloudpickle as pickle
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from sqlalchemy import desc
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split

from src.configurations import FeatureConfigurations


def fetch_data_from_local(filepath: str):
    df = pd.read_csv(filepath)

    features = df.drop(labels=[FeatureConfigurations.TARGET], axis=1)
    target = df[FeatureConfigurations.TARGET]

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

    with open(save_path, "wb") as f:
        pickle.dump(object_file, f)

    if header is not None:
        headers = {i: col for i, col in enumerate(header.split(","))}
        header_filename = "object_labels.json"
        header_path = object_dir / header_filename
        with open(header_path, "w") as f:
            json.dump(headers, f)
