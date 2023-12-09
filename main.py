import torch
from commands.lvs_lsblk import get_lv_lsblk
from commands.pvs import get_physical_volumes
from commands.vgs import get_group_volumes
from database.connect import DB_NAME, connect_to_database
from database.utils import insert_to_logical_volume_adjustment, insert_to_logical_volume_stats, insert_to_physical_volume_stats, insert_to_segment_stats, insert_to_volume_group_stats
from sqlalchemy.orm import sessionmaker, Session
import schedule
from sqlalchemy.exc import SQLAlchemyError
from exceptions.LvmCommandError import LvmCommandError
from helpers import load_model, load_scaler, reverse_transform
from logs.Logger import Logger
from prediction.lstm import LOOKBACK
from root_directory import root_directory
from database.models import (
    Adjustment,
    LogicalVolumeStats,
    VolumeGroup,
    LogicalVolume,
    LogicalVolumeInfo
)
import numpy as np


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


def scrape_lvm_stats(session: Session, lvm_logger: Logger):
    pvs = get_physical_volumes(lvm_logger)
    vgs = get_group_volumes(lvm_logger)
    lvs_fs = get_lv_lsblk(lvm_logger)
    insert_to_volume_group_stats(
        session, vgs[["vg_uuid", "vg_name", "vg_size", "vg_free"]].drop_duplicates())
    insert_to_physical_volume_stats(
        session, pvs[["pv_uuid", "pv_name", "pv_size", "vg_uuid"]].drop_duplicates())
    insert_to_logical_volume_stats(
        session, lvs_fs[["lv_uuid", "lv_name", "fstype", "fssize", "fsused", "fsavail", "priority"]])
    insert_to_segment_stats(
        session, lvs_fs[["seg_size", "segment_range_start", "segment_range_end", "pv_name", "lv_uuid"]])
    allocations = calculate_allocations_volumes(session, lvm_logger, 1000)
    insert_to_logical_volume_adjustment(session, allocations)
    execute_logical_volumes_sizes_adjustments(session)
    session.close()


def execute_logical_volumes_sizes_adjustments(session: Session):
    adjustments = session.query(Adjustment).filter_by(status="pending").all()
    for adjustment in adjustments:
        print(
            f"Making adjustment for {adjustment.logical_volume.info.lv_name}: {adjustment.size}")
        adjustment.status = "done"
        session.add(adjustment)
        session.commit()


def get_logical_volumes_info_list(session: Session, limit: int = 6) -> list[dict[str, any]]:
    # get logical volumes
    logical_volumes = session.query(LogicalVolume, LogicalVolumeInfo).\
        join(LogicalVolumeInfo).all()

    logical_volumes_data_list = []
    for logical_volume in logical_volumes:
        # get latest (depends on the number of the lookbacks) stats of one logical volumes
        if limit == 0:
            lv_stat = session.query(LogicalVolumeStats).filter_by(
                logical_volume_id_fk=logical_volume.LogicalVolume.id)\
                .order_by(LogicalVolumeStats.created_at.desc())\
                .limit(1)\
                .all()
        else:
            lv_stat = session.query(LogicalVolumeStats)\
                .filter_by(logical_volume_id_fk=logical_volume.LogicalVolume.id)\
                .order_by(LogicalVolumeStats.created_at.desc())\
                .limit(limit)\
                .all()

        if len(lv_stat) > 0:
            # array of latest logical_volume fs usage
            lv_fs_usage = [stat.file_system_used_size for stat in lv_stat]
            # # append the mean value | array has to be of 7 entries for the scaling
            # # it is gonna be ignored during prediction
            # lv_fs_usage.append(sum(lv_fs_usage) / len(lv_fs_usage))

            # to reshaped numpy array
            lv_fs_usage_np = np.array(
                # flip the order of the usage serie
                # newest usage volume is at the end of the list (array)
                np.flip(lv_fs_usage)
            ).flatten()

            # create a dictionary for each logical volume
            lv_data = {
                'lv_uuid': logical_volume.LogicalVolume.lv_uuid,
                'priority': logical_volume.LogicalVolume.priority.value,
                'file_system_size': lv_stat[0].file_system_size,
                'usage_serie': lv_fs_usage_np.tolist(),  # Convert numpy array to list
            }

            # add the dictionary to the list
            logical_volumes_data_list.append(lv_data)

    # return the list of dictionaries
    return logical_volumes_data_list


def predict(lv_uuid: str, lvs_lastest_usage_series: list[int]):
    # load model
    model = load_model(lv_uuid=lv_uuid)
    # load scaler
    scaler = load_scaler(lv_uuid=lv_uuid)
    lvs_lastest_usage_series.append(
        sum(lvs_lastest_usage_series)/len(lvs_lastest_usage_series))
    # transform (scale) lv fs usage
    lvs_lastest_usage_series = np.array(
        np.array(lvs_lastest_usage_series)
    ).reshape(1, LOOKBACK+1)
    lvs_lastest_usage_series_scaled = scaler.transform(
        lvs_lastest_usage_series)
    lv_fs_usage_tensor = torch.tensor(
        lvs_lastest_usage_series_scaled[:, :-1].reshape(-1, LOOKBACK, 1)).float()
    # make prediction
    prediction = model(lv_fs_usage_tensor).to(
        "cpu").detach().numpy().flatten()
    return prediction


def get_vg_size(session: Session, vg_uuid):
    return session.query(VolumeGroup).filter_by(vg_uuid=vg_uuid).first().stats.vg_size


def calculate_mean_proportions(logical_volumes_info_list: list[dict[str, int]]) -> list[float]:
    """
    Calculate mean proportions based on the provided usage series and lvs_priorities.

    Args:
    - lvs_latest_usage_series (list): The latest usage series for different LV types.
    - lvs_priorities (list): Priorities corresponding to each LV type.

    Returns:
    - list: Normalized mean proportions.
    """
    usage_series_np = np.array(
        [usage_serie["usage_serie"] for usage_serie in logical_volumes_info_list])
    column_sums = usage_series_np.sum(axis=0)
    proportions = usage_series_np / column_sums
    mean_proportions: list[float] = np.mean(proportions, axis=1).tolist()
    return [
        {
            "lv_uuid": usage_serie["lv_uuid"],
            "priority": usage_serie["priority"],
            "mean_proportion": mean_proportions[index]
        }
        for index, usage_serie
        in enumerate(logical_volumes_info_list)
    ]


def calculate_proportions_by_priority(logical_volumes_info_list: list[dict[str, int | str | list[float]]]) -> list[float]:
    """
    Calculate and normalize mean proportions based on the provided mean proportions
    and priorities for each logical volume.

    Args:
    - mean_proportions (list[float]): List of mean proportions for each logical volume.
    - lvs_priorities (list[int]): Priorities corresponding to each logical volume.

    Returns:
    - list[float]: Normalized mean proportions.
    """
    logical_volumes_info_list: list[
        dict[str, int | str | list[float]]
    ] = calculate_mean_proportions(logical_volumes_info_list)
    # Step 1: Scale mean proportions by lvs_priorities
    mean_proportions_scaled = np.zeros_like(logical_volumes_info_list)
    for index in range(len(mean_proportions_scaled)):
        priority = logical_volumes_info_list[index]["priority"]
        mean_proportion = logical_volumes_info_list[index]["mean_proportion"]

        priority_count = [element["priority"]
                          for element in logical_volumes_info_list].count(priority)

        mean_proportions_scaled[index] = (
            mean_proportion / priority * priority_count)
        # print(f"Mean proportion LV{index+1} | {logical_volumes_info_list[index]['lv_uuid']}:",
        #       mean_proportions_scaled[index])
    # Step 2: Normalize scaled proportions to ensure they sum up to 1
    sum_mean_proportions_scaled = np.sum(mean_proportions_scaled)
    # print("Sum of scaled mean proportions:", sum_mean_proportions_scaled)
    # print("------------------------------------------------")
    normalized_mean_proportions = mean_proportions_scaled / sum_mean_proportions_scaled
    # for index, normalized_mean_proportion in enumerate(normalized_mean_proportions):
    #     print(
    #         f"Normalized mean proportions LV{index +1} | {logical_volumes_info_list[index]['lv_uuid']}:", normalized_mean_proportion)
    return normalized_mean_proportions


def process_logical_volumes(session: Session):
    """
    Make decisions based on predictions, mean proportions, and allocation volume.

    Args:
    - session (Session): The session object for database operations.
    - allocation_volume (int): Max allocated volume.

    Returns:
    - None
    """
    # Get LV data from the database
    logical_volume_info_list = get_logical_volumes_info_list(
        session, LOOKBACK
    )
    # list of predictions of the LSTM model
    predictions: list[dict[str, int]] = []
    # list of the differences between the actual logical volume size and the predicted usage
    needed_allocations: list[dict[str, int]] = []
    # Step 1: Make predictions for each LV
    for index, logical_volume_info in enumerate(logical_volume_info_list):
        lv_uuid = logical_volume_info["lv_uuid"]
        usage_serie = logical_volume_info["usage_serie"]
        file_system_size = logical_volume_info["file_system_size"]
        current_used_volume_pourcentage = usage_serie[-1] * \
            100 / file_system_size
        if current_used_volume_pourcentage > 80:
            # give the logical volume a superieur priority
            if logical_volume_info["priority"] > 1:
                logical_volume_info_list[index]["priority"] = logical_volume_info["priority"] - 1
                print(
                    f"Giving more priority to {lv_uuid}: {logical_volume_info_list[index]['priority']}")

        # get logical volume future usage prediction
        prediction = predict(lv_uuid, usage_serie.copy())

        # load the logical volume scaler
        scaler = load_scaler(lv_uuid=lv_uuid)
        # reverse transform logical volume prediction
        reverse_transformed_prediction = np.round(
            reverse_transform(scaler, prediction).flatten()[0]
        )
        allocation = reverse_transformed_prediction - file_system_size
        # # calculate the difference between the latest usage volume size and the prediction
        # # to observe if the prediction is about extending or reducing
        # diff = latest_usage_volume - reverse_transformed_prediction

        predictions.append(
            {
                "lv_uuid": logical_volume_info["lv_uuid"],
                "prediction": reverse_transformed_prediction
            }
        )
        needed_allocations.append(
            {
                "lv_uuid": logical_volume_info["lv_uuid"],
                # if the logical volume is asking for more than it has --> positive --> possibility of volume extention
                # if the logical volume is asking for less than it has --> negative --> possiblity of volume reduction
                "allocation": allocation
            }
        )
    return logical_volume_info_list, predictions, needed_allocations


def calculate_allocations_volumes(session: Session, lvm_logger: Logger, allocation_volume: int):
    logical_volume_info_list, predictions, needed_allocations = process_logical_volumes(
        session)

    # Step 2: Calculate mean proportions by each logical volume priority
    mean_proportions: list[float] = calculate_proportions_by_priority(
        [
            {
                "lv_uuid": logical_volume_data["lv_uuid"],
                "priority": logical_volume_data["priority"],
                "usage_serie": logical_volume_data["usage_serie"]
            }
            for logical_volume_data
            in logical_volume_info_list
        ]
    )

    # Step 3: Scale predictions by the mean proportions/priority of each LV
    scaled_predictions: list[float] = [pred["prediction"] * mean_proportions[index]
                                       for index, pred in enumerate(predictions)]

    # Step 4: Calculate the allocation factor
    allocation_factor: float = allocation_volume / sum(scaled_predictions)
    # Step 5 : Calculate the allocation volume size for each logical volume
    allocations: list[dict[str, int]] = []
    for index, needed_allocation in enumerate(needed_allocations):
        # scale possible allocation size by allocation factor
        allocation_volume_size = int(
            np.round(needed_allocation["allocation"] * allocation_factor)
        )
        if allocation_volume_size >= 0:
            allocation_volume_size = min(
                allocation_volume_size,
                needed_allocation["allocation"]
            )
        else:
            allocation_volume_size = max(
                allocation_volume_size,
                needed_allocation["allocation"]
            )
        allocations.append(
            {
                "lv_uuid": needed_allocation["lv_uuid"],
                "size": allocation_volume_size
            }
        )
        lvm_logger.get_logger().debug(
            f"Allocation volume size with max limited allocation of {allocation_volume} MiB: {allocation_volume_size} MiB")
    return allocations


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
