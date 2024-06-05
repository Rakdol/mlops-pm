import os
from argparse import ArgumentParser

from sqlalchemy import create_engine, Integer, Column, String
from sqlalchemy.orm import declarative_base


FILE = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    parser.add_argument(
        "--path", dest="path", type=str, default="./data/raw/total_data.csv"
    )
    args = parser.parse_args()

    db_url = f"postgresql+psycopg2://admin:1234@{args.db_host}:5432/machinedb"
    engine = create_engine(db_url)
    Base = declarative_base()

    class Machine(Base):
        __tablename__ = "machine_data"
