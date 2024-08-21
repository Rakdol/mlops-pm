import os
import sys
import time
import pickle
import datetime

from pathlib import Path
from io import StringIO, BytesIO
from typing import Optional, List, Dict

import boto3

from src.bucket import minio_client


class MachineDataset(object):
    def __init__(
        self,
        upstream_directory: str,
        file_prefix: str,
        file_name: str,
        pipe_prefix: str,
        pipe_name: Optional[str],
        minio_client: Optional[boto3.client],
    ):
        self.upstream_directory = upstream_directory
        self.file_prefix = file_prefix
        self.file_name = file_name
        self.pipe_prefix = pipe_prefix
        self.pipe_name = pipe_name
        self.client = minio_client

    def get_input_pipe(self):
        pipe_path = str(
            Path() / self.upstream_directory / self.pipe_prefix / self.pipe_name
        )
        print(pipe_path)
        if self.client is not None:
            response = self.client.get_object(Bucket="mlflow", Key=pipe_path)
            pkl_data = response["Body"].read()
            pipe = pickle.load(BytesIO(pkl_data))

        else:
            pipe = None

        return pipe


if __name__ == "__main__":

    train_set = MachineDataset(
        upstream_directory="0/abf8ddaa5a874294b517a961a960ada3/artifacts/downstream_directory",
        file_prefix="train",
        file_name="train_dataset",
        pipe_prefix="pipe",
        pipe_name="pipe.pkl",
        minio_client=minio_client,
    )

    pipe = train_set.get_input_pipe()
    print(pipe)
