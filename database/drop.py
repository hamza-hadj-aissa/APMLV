

from database.connect import connect_to_database, create_tables, drop_tables
from logs.Logger import Logger
from root_directory import root_directory
log_file_path = f"{root_directory}/logs/lvm_balancer.log"
db_logger = Logger("Postgres", path=log_file_path)
engine = connect_to_database(logger=db_logger)
drop_tables(engine, db_logger)
create_tables(engine, db_logger)
