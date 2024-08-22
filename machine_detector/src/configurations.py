import os


class APIConfigurations:
    title = os.getenv("API_TITLE", "Machine Detector Service")
    description = os.getenv("API_DESCRIPTION", "Machine Detector Service API")
    version = os.getenv("API_VERSION", "0.1")
