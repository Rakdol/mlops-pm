from logging import getLogger

from fastapi import FastAPI


from src.api.router import api
from configurations import APIConfigurations
from db import initialize
from db.database import engine
from utils.logger import logging


initialize.initialize_table(engine=engine, checkfirst=True)


app = FastAPI(
    title=APIConfigurations.title,
    description=APIConfigurations.description,
    version=APIConfigurations.version,
)

app.include_router(
    api.router, prefix=f"/v{APIConfigurations.version}/api", tags=["api"]
)
