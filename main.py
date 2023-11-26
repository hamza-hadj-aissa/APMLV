import logging
from commands.lvs_lsblk import get_lv_lsblk
from commands.pvs import get_physical_volumes
from commands.vgs import get_group_volumes
from database.connect import connect_to_database
from database.utils import insert_to_logical_volume_stats, insert_to_physical_volume_stats, insert_to_segment_stats, insert_to_volume_group_stats
from sqlalchemy.orm import sessionmaker, Session
import schedule

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


def scrape_lvm_stats(session: Session):
    pvs = get_physical_volumes()
    vgs = get_group_volumes()
    lvs_fs = get_lv_lsblk()
    insert_to_volume_group_stats(
        session, vgs[["vg_uuid", "vg_name", "vg_size", "vg_free"]].drop_duplicates())
    insert_to_physical_volume_stats(
        session, pvs[["pv_uuid", "pv_name", "pv_size", "vg_uuid"]].drop_duplicates())
    insert_to_logical_volume_stats(
        session, lvs_fs[["lv_uuid", "lv_name", "fstype", "fssize", "fsused", "fsavail"]])
    insert_to_segment_stats(
        session, lvs_fs[["seg_size", "segment_range_start", "segment_range_end", "pv_name", "lv_uuid"]])


if __name__ == "__main__":
    # Connect to the database
    database_engine = connect_to_database()
    DBSession = sessionmaker(bind=database_engine)
    session = DBSession()
    schedule.every(5).seconds.do(scrape_lvm_stats, session=session)
    try:
        while True:
            schedule.run_pending()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
