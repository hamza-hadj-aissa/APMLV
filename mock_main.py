import random
import pandas as pd
import schedule
from database.utils import insert_to_logical_volume_adjustment, insert_to_logical_volume_stats
from sqlalchemy.orm import Session
from exceptions.LvmCommandError import LvmCommandError
from logs.Logger import Logger
from main import calculate_allocations_volumes, connect, execute_logical_volumes_sizes_adjustments
from prediction.dataset.generate import generate_usage_history
from database.models import (
    LogicalVolumeStats,
    LogicalVolume
)
from root_directory import root_directory
from sqlalchemy.exc import SQLAlchemyError


def lv_stats_mock(session: Session):
    logical_volumes = session.query(LogicalVolume)\
        .join(LogicalVolumeStats)\
        .all()
    for lv in logical_volumes:
        lv_stats = session.query(LogicalVolumeStats)\
            .filter_by(logical_volume_id_fk=lv.id)\
            .order_by(LogicalVolumeStats.created_at.desc())\
            .first()
        lv_uuid = lv.lv_uuid
        lv_name = lv.info.lv_name
        priority = lv.priority.value
        file_system_type = lv_stats.file_system.file_system_type
        file_system_size = lv_stats.file_system_size
        file_system_used_size = lv_stats.file_system_used_size
        generated_lv_stats = generate_usage_history(lv_uuid, lv_name, file_system_type, priority, file_system_size, file_system_used_size, 10,
                                                    random.randint(1, 24))
        print(generated_lv_stats)
        df = pd.DataFrame.from_dict(generated_lv_stats)
        df.rename(columns={
            "uuid": "lv_uuid",
            "name": "lv_name",
            "total_capacity": "fssize",
            "free_space": "fsavail",
            "used_space": "fsused"
        }, inplace=True)
        insert_to_logical_volume_stats(session, df)


def mock_program(session: Session, lvm_logger: Logger):
    lv_stats_mock(session)
    allocations = calculate_allocations_volumes(
        session,
        lvm_logger,
        allocation_volume=1000
    )
    insert_to_logical_volume_adjustment(session, allocations)
    execute_logical_volumes_sizes_adjustments(session)


if __name__ == "__main__":
    # expressed in seconds
    # 60 * 5 = 5 minutes
    time_interval = 5
    log_file_path = f"{root_directory}/logs/lvm_balancer.log"
    # define loggers
    db_logger = Logger("Postgres", path=log_file_path)
    lvm_logger = Logger("LVM", path=log_file_path)
    main_logger = Logger("Main", path=log_file_path)
    main_logger.get_logger().info("Starting Lvm Balancer...")

    # process starts here --
    try:
        session = connect(db_logger)
        schedule.every(time_interval).seconds.do(mock_program,
                                                 session=session, lvm_logger=lvm_logger)
        while True:
            schedule.run_pending()

    except KeyboardInterrupt:
        pass
    except LvmCommandError as e:
        lvm_logger.get_logger().error(e)
    except SQLAlchemyError as e:
        db_logger.get_logger().error(e)
