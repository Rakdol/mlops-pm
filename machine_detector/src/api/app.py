
import time

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.router import api
from src.configurations import APIConfigurations
from src.db import initialize
from src.db.database import engine
from src.utils.logger import logging

initialize.initialize_table(engine=engine, checkfirst=True)


app = FastAPI(
    title=APIConfigurations.title,
    description=APIConfigurations.description,
    version=APIConfigurations.version,
)

app.include_router(
    api.router, prefix=f"/v{APIConfigurations.version}/api", tags=["api"]
)

Instrumentator().instrument(app).expose(app)

# if __name__ == "__main__":
#     start_http_server(8001)  # 이 코드를 메인 프로세스에서만 실행되도록 함
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)