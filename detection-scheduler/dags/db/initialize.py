from logging import getLogger

from db.database import BaseEvent, BaseSource

logger = getLogger(__name__)

def create_source_tables(engine, checkfirst: bool = True):
    logger.info("Initialize tables if not exist.")
    BaseSource.metadata.create_all(engine, checkfirst=checkfirst)

def create_event_tables(engine, checkfirst: bool = True):
    logger.info("Initialize tables if not exist.")
    BaseEvent.metadata.create_all(engine, checkfirst=checkfirst)


def initialize_source_table(engine, checkfirst: bool = True):
    logger.info("Initialize tables")
    create_source_tables(engine=engine, checkfirst=checkfirst)

def initialize_event_table(engine, checkfirst: bool = True):
    logger.info("Initialize tables")
    create_event_tables(engine=engine, checkfirst=checkfirst)


