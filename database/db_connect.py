from sqlalchemy import create_engine, table
from sqlalchemy_utils import create_database, database_exists
import os
from dotenv import load_dotenv
from models import Base


# load .env variables
load_dotenv()
HOSTNAME = os.environ.get("HOST_NAME")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = os.environ.get("DB_PORT")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


engine = create_engine(
    "postgresql+psycopg://{}:{}@{}:{}/{}".format(DB_USER, DB_PASSWORD, HOSTNAME, DB_PORT, DB_NAME))

if not database_exists(engine.url):
    print("Creating database {}".format(DB_NAME))
    create_database(engine.url)

print("Database connected")


print("Creating database tables")
Base.metadata.create_all(bind=engine)
