import os


class APIConfigurations:
    title = os.getenv("API_TITLE", "Machine Detector Service")
    description = os.getenv("API_DESCRIPTION", "Machine Detector Service API")
    version = os.getenv("API_VERSION", "0.1")


class DBConfigurations:
    postgres_username = os.getenv("POSTGRES_USER", "admin")
    postgres_password = os.getenv("POSTGRES_PASSWORD", 1234)
    postgres_port = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_db = os.getenv("POSTGRES_DB", "machinedb")
    postgres_server = os.getenv(
        "POSTGRES_SERVER", "172.17.0.1"
    )  # default docker bridge
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
