from sqlalchemy import create_engine
from sqlalchemy_utils import create_database, database_exists
from dotenv import load_dotenv
import os
from database.models.models import Base

# Load .env variables
load_dotenv()
HOSTNAME = os.environ.get("HOST_NAME")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = os.environ.get("DB_PORT")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def connect_to_database():
    # Create an engine
    engine = create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{HOSTNAME}:{DB_PORT}/{DB_NAME}")

    # Check if the database exists
    if not database_exists(engine.url):
        print(f"Creating database {DB_NAME}")
        create_database(engine.url)

    return engine


def drop_tables(engine):
    Base.metadata.drop_all(engine)
    print("Tables dropped")


def create_tables(engine):
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("Tables created")
