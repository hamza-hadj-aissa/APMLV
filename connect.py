from database.connect import DB_NAME, connect_to_database
from logs.Logger import Logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker


def connect(db_logger: Logger):
    try:
        db_logger.get_logger().info(f"Connecting to Database ({DB_NAME})...")
        # Connect to the database
        database_engine = connect_to_database(db_logger)
        DBSession = sessionmaker(bind=database_engine)
        session = DBSession()
        db_logger.get_logger().info(f"Connected to Database ({DB_NAME})")
        return session
    except SQLAlchemyError as e:
        raise e
