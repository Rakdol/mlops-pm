import os


class SourceDBConfigurations:
    postgres_username = os.getenv("POSTGRES_USER", "admin")
    postgres_password = os.getenv("POSTGRES_PASSWORD", 1234)
    postgres_port = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db = os.getenv("POSTGRES_DB", "machinedb")
    postgres_server = os.getenv(
        "POSTGRES_SERVER", "postgres-server"
    )  # default docker bridge
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"


class TargetDBConfigurations:
    postgres_username = os.getenv("POSTGRES_USER", "eventuser")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "eventpassword")
    postgres_port = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db = os.getenv("POSTGRES_DB", "eventdb")
    postgres_server = os.getenv(
        "POSTGRES_SERVER", "event-postgres-server"
    )  # default docker bridge
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
