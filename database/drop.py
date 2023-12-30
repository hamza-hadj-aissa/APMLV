

from database.connect import connect_to_database, create_tables, drop_tables
from logs.Logger import Logger

db_logger = Logger("Postgres")
engine = connect_to_database(logger=db_logger)
drop_tables(engine, db_logger)
create_tables(engine, db_logger)
