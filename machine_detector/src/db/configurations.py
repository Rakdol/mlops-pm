from pathlib import Path
import os


class DBConfigurations:
    postgres_username = os.getenv("POSTGRES_USER", "admin")
    postgres_password = os.getenv("POSTGRES_PASSWORD", 1234)
    postgres_port = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db = os.getenv("POSTGRES_DB", "machinedb")
    postgres_server = os.getenv("POSTGRES_SERVER", "localhost")
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
