import logging
import subprocess
from commands.lvs_lsblk import get_lv_lsblk
from commands.pvs import get_physical_volumes
from commands.utils import run_command
from commands.vgs import get_group_volumes
from database.connect import DB_NAME, connect_to_database
from database.utils import insert_to_logical_volume_stats, insert_to_physical_volume_stats, insert_to_segment_stats, insert_to_volume_group_stats
from sqlalchemy.orm import sessionmaker, Session
import schedule
from sqlalchemy.exc import SQLAlchemyError
from exceptions.LvmCommandError import LvmCommandError
from logs.Logger import Logger


def scrape_lvm_stats(session: Session, lvm_logger: Logger):
    pvs = get_physical_volumes(lvm_logger)
    vgs = get_group_volumes(lvm_logger)
    lvs_fs = get_lv_lsblk(lvm_logger)
    insert_to_volume_group_stats(
        session, vgs[["vg_uuid", "vg_name", "vg_size", "vg_free"]].drop_duplicates())
    insert_to_physical_volume_stats(
        session, pvs[["pv_uuid", "pv_name", "pv_size", "vg_uuid"]].drop_duplicates())
    insert_to_logical_volume_stats(
        session, lvs_fs[["lv_uuid", "lv_name", "fstype", "fssize", "fsused", "fsavail"]])
    insert_to_segment_stats(
        session, lvs_fs[["seg_size", "segment_range_start", "segment_range_end", "pv_name", "lv_uuid"]])
    session.close()


def connect(db_logger: Logger):
    try:
        db_logger.get_logger().info(f"Connecting to Database ({DB_NAME})...")
        # Connect to the database
        database_engine = connect_to_database(db_logger)
        DBSession = sessionmaker(bind=database_engine)
        session = DBSession()
        db_logger.get_logger().info(f"Connected to database ({DB_NAME})")
        return session
    except SQLAlchemyError as e:
        raise e


if __name__ == "__main__":
    # expressed in seconds
    # 60 * 5 = 5 minutes
    time_interval = 60 * 5
    db_logger = Logger(
        "Postgres", path="/home/hamza/Desktop/studio/python/lvm_balancer/logs/lvm_balancer.log")
    lvm_logger = Logger(
        "LVM", path="/home/hamza/Desktop/studio/python/lvm_balancer/logs/lvm_balancer.log")
    main_logger = Logger(
        "Main", path="/home/hamza/Desktop/studio/python/lvm_balancer/logs/lvm_balancer.log")
    main_logger.get_logger().info("Starting Lvm Balancer...")
    try:
        session = connect(db_logger)
        schedule.every(time_interval).seconds.do(scrape_lvm_stats,
                                                 session=session, lvm_logger=lvm_logger)
        while True:
            schedule.run_pending()
    except KeyboardInterrupt:
        pass
    except LvmCommandError as e:
        lvm_logger.get_logger().error(e)
    except SQLAlchemyError as e:
        db_logger.get_logger().error(e)
