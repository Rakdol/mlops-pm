import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from logging import getLogger

from db.configurations import SourceDBConfigurations, TargetDBConfigurations

logger = getLogger(__name__)
logger.info(f"Source DB Configuration URL: {SourceDBConfigurations.sql_alchemy_database_url}")
source_engine = create_engine(
    SourceDBConfigurations.sql_alchemy_database_url,
    pool_recycle=3600,
    echo=False,
)

logger.info(f"Event DB Configuration URL: {TargetDBConfigurations.sql_alchemy_database_url}")
event_engine = create_engine(
    TargetDBConfigurations.sql_alchemy_database_url,
    pool_recycle=3600,
    echo=False,
)

Session_Source = sessionmaker(autocommit=False, autoflush=False, bind=source_engine)
Session_Event = sessionmaker(autocommit=False, autoflush=False, bind=event_engine)

BaseSource = declarative_base()
BaseEvent = declarative_base()