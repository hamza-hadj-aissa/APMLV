

from database.connect import connect_to_database, create_tables, drop_tables


engine = connect_to_database()
drop_tables(engine)
create_tables(engine)
