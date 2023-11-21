from models import Group_volume, Physical_volume, Base
from db_connect import engine


print("Creating database tables")
Base.metadata.create_all(bind=engine)
