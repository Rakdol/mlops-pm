import os
import time
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import psycopg2


def create_table(db_connect):
    drop_table_query = """
    DROP TABLE IF EXISTS machine_data;
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS machine_data (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        product_id VARCHAR(20),
        machine_type CHAR(2),
        air_temperature FLOAT8,
        process_temperature FLOAT8,
        rotational_speed INTEGER,
        torque FLOAT8,
        tool_wear FLOAT8,
        machine_failure INT,
        TWF INT,
        HDF INT,
        PWF INT,
        OSF INT,
        RNF INT
    );"""

    print(create_table_query)
    with db_connect.cursor() as cur:
        # cur.execute(drop_table_query)
        cur.execute(create_table_query)
        db_connect.commit()


def insert_data(db_connect, data):
    machine_failure = (
        data["Machine failure"] if not np.isnan(data["Machine failure"]) else "NULL"
    )

    insert_data_query = f"""
    INSERT INTO machine_data (
        timestamp, product_id, machine_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear, machine_failure, TWF, HDF, PWF, OSF, RNF
    ) VALUES (
        NOW(),
        '{data["Product ID"]}',
        '{data["Type"]}',
        {data["Air temperature [K]"]},
        {data["Process temperature [K]"]},
        {data["Rotational speed [rpm]"]},
        {data["Torque [Nm]"]},
        {data["Tool wear [min]"]},
        {machine_failure},
        {data["TWF"]},
        {data["HDF"]},
        {data["PWF"]},
        {data["OSF"]},
        {data["RNF"]}
    );
    """
    print(insert_data_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_data_query)
        db_connect.commit()


def generate_data(db_connect, df):
    while True:
        insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)


def get_parent(path, levels=1):
    common = path

    # Using for loop for getting
    # starting point required for
    # os.path.relpath()
    for i in range(levels + 1):

        # Starting point
        common = os.path.dirname(common)

    # Parent directory upto specified
    # level
    return common


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    parser.add_argument(
        "--csv-path", dest="csv_path", type=str, default="/usr/app/total_data.csv"
    )

    args = parser.parse_args()

    print("DataBase Host", args.db_host)
    print("Data File Path", args.csv_path)

    db_connect = psycopg2.connect(
        user="admin",
        password="1234",
        host=args.db_host,
        port=5432,
        database="machinedb",
    )

    create_table(db_connect)
    df = pd.read_csv(args.csv_path)
    generate_data(db_connect, df)
