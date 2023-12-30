from sqlalchemy import create_engine
from sqlalchemy_utils import create_database, database_exists
from dotenv import load_dotenv
import os
from database.models import Base
from sqlalchemy.exc import SQLAlchemyError

from logs.Logger import Logger

# Load .env variables
load_dotenv()
HOSTNAME = os.environ.get("HOST_NAME")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = os.environ.get("DB_PORT")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def connect_to_database(logger: Logger):
    try:
        # Create an engine
        engine = create_engine(
            f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{HOSTNAME}:{DB_PORT}/{DB_NAME}")

        # Check if the database exists
        if not database_exists(engine.url):
            logger.get_logger().info(f"Creating database {DB_NAME}")
            create_database(engine.url)

        return engine
    except SQLAlchemyError as e:
        raise e
