from sqlalchemy import create_engine
from database.connect import DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, HOSTNAME
from database.models import Base


def drop_tables(engine):
    # Drop tables
    Base.metadata.drop_all(engine)
    print("Tables dropped")


def create_tables(engine):
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("Tables created")


if __name__ == "__main__":
    # Create an engine
    engine = create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{HOSTNAME}:{DB_PORT}/{DB_NAME}")

    drop_tables(engine)
    create_tables(engine)
