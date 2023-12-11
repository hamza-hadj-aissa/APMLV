import json
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
    PhysicalVolume,
    Segment,
    VolumeGroup,
    LogicalVolume,
    LogicalVolumeInfo,
    VolumeGroupInfo,
    VolumeGroupStats
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
    allocations = process_volume_groups(session, lvm_logger)
    insert_to_logical_volume_adjustment(session, allocations)
    execute_logical_volumes_sizes_adjustments(session, lvm_logger)
    session.close()


def execute_logical_volumes_sizes_adjustments(session: Session, lvm_logger: Logger):
    adjustments = session.query(Adjustment).filter_by(status="pending").all()
    for adjustment in adjustments:
        lvm_logger.get_logger().debug(
            f"Making adjustment for {adjustment.logical_volume.info.lv_name}: {adjustment.size}")
        adjustment.status = "done"
        session.add(adjustment)
        session.commit()


def get_logical_volumes_info_list(session: Session, vg_uuid: str, logical_volumes: list[dict[str, str]], lv_stats_limit):
    logical_volumes_list_by_volume_group = session.query(LogicalVolumeInfo, LogicalVolume, VolumeGroupInfo, VolumeGroup)\
        .join(LogicalVolume)\
        .join(Segment)\
        .join(PhysicalVolume)\
        .join(VolumeGroup)\
        .join(VolumeGroupStats)\
        .join(VolumeGroupInfo)\
        .filter(VolumeGroup.vg_uuid == vg_uuid)\
        .filter(LogicalVolumeInfo.lv_name.in_(logical_volumes))\
        .all()
    logical_volumes_informations_list = []
    for logical_volume in logical_volumes_list_by_volume_group:
        # get latest (depends on the number of the lookbacks) stats of one logical volumes
        if lv_stats_limit == 0:
            lv_stat = session.query(LogicalVolumeStats).filter_by(
                logical_volume_id_fk=logical_volume.LogicalVolume.id)\
                .order_by(LogicalVolumeStats.created_at.desc())\
                .limit(1)\
                .all()
        else:
            lv_stat = session.query(LogicalVolumeStats)\
                .filter_by(logical_volume_id_fk=logical_volume.LogicalVolume.id)\
                .order_by(LogicalVolumeStats.created_at.desc())\
                .limit(lv_stats_limit)\
                .all()

        if len(lv_stat) > 0:
            # array of latest logical_volume fs usage
            lv_fs_usage = [stat.file_system_used_size for stat in lv_stat]
            # # append the mean value | array has to be of 7 entries for the scaling
            # # it is gonna be ignored during prediction
            # lv_fs_usage.append(sum(lv_fs_usage) / len(lv_fs_usage))

            # to reshaped numpy array
            lv_fs_usage_allocation_reclaim_size = np.array(
                # flip the order of the usage serie
                # newest usage volume is at the end of the list (array)
                np.flip(lv_fs_usage)
            ).flatten()

            # create a dictionary for each logical volume
            lv_data = {
                'lv_uuid': logical_volume.LogicalVolume.lv_uuid,
                'priority': logical_volume.LogicalVolume.priority.value,
                'file_system_size': lv_stat[0].file_system_size,
                # Convert numpy array to list
                'usage_serie': lv_fs_usage_allocation_reclaim_size.tolist(),
            }
            # add the dictionary to the list
            logical_volumes_informations_list.append(lv_data)
    return logical_volumes_informations_list


def get_volume_groups_info_list(session: Session) -> list[dict[str, str | int | float]]:
    with open(f"{root_directory}/logical_volumes.json", 'r') as file:
        logical_volumes_json = json.load(file)

    volume_groups = sorted([vg
                            for vg in logical_volumes_json["volume_groups"]], key=lambda x: x['vg_name'])

    volume_groups_instances = session.query(
        VolumeGroup, VolumeGroupInfo)\
        .join(VolumeGroupInfo)\
        .filter(VolumeGroupInfo.vg_name.in_([volume_group["vg_name"] for volume_group in volume_groups]))\
        .order_by(VolumeGroupInfo.vg_name.asc())\
        .all()
    volume_groups_informations_list = []

    for index, volume_group in enumerate(volume_groups_instances):
        logical_volumes_names = [logical_volume["lv_name"]
                                 for logical_volume in volume_groups[index]["logical_volumes"]]
        print(logical_volumes_names, volume_group.VolumeGroupInfo.vg_name)
        volume_group_stat = session.query(VolumeGroupStats).filter_by(
            volume_group_id_fk=volume_group.VolumeGroup.id)\
            .order_by(VolumeGroupStats.created_at.desc())\
            .limit(1)\
            .all()
        vg_uuid = volume_group.VolumeGroup.vg_uuid
        vg_name = volume_group.VolumeGroupInfo.vg_name
        vg_free = volume_group_stat[0].vg_free
        vg_data = {
            "vg_name": vg_name,
            "vg_uuid": vg_uuid,
            "vg_free": vg_free,
            "logical_volumes": get_logical_volumes_info_list(session, vg_uuid, logical_volumes_names, lv_stats_limit=6)
        }
        volume_groups_informations_list.append(vg_data)
    # return the list of dictionaries
    return volume_groups_informations_list


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


def calculate_mean_proportions(logical_volumes_info_list: list[dict[str, int]]) -> list[dict[str, str | int | float]]:
    """
    Calculate mean proportions based on the provided usage series and lvs_priorities.

    Args:
    - lvs_latest_usage_series (list): The latest usage series for different LV types.
    - lvs_priorities (list): Priorities corresponding to each LV type.

    Returns:
    - list: Normalized mean proportions.
    """
    usage_series = np.array(
        [usage_serie["usage_serie"] for usage_serie in logical_volumes_info_list])
    column_sums = usage_series.sum(axis=0)
    proportions = usage_series / column_sums
    mean_proportions: list[float] = np.mean(
        proportions, axis=1).tolist()

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
    mean_proportions: list[
        dict[str, int | str | list[float]]
    ] = calculate_mean_proportions(logical_volumes_info_list.copy())
    # Step 1: Scale mean proportions by lvs_priorities
    mean_proportions_scaled = np.zeros_like(
        logical_volumes_info_list)
    for index in range(len(mean_proportions_scaled)):
        priority: int = mean_proportions[index]["priority"]
        mean_proportion: float = mean_proportions[index]["mean_proportion"]

        priority_count: int = [element["priority"]
                               for element in logical_volumes_info_list].count(priority)

        mean_proportions_scaled[index] = mean_proportion / \
            priority * priority_count
        # lvm_logger.get_logger().debug(f"Mean proportion LV{index+1} | {logical_volumes_info_list[index]['lv_uuid']}:",
        #       mean_proportions_scaled[index])
    # Step 2: Normalize scaled proportions to ensure they sum up to 1
    sum_mean_proportions_scaled = np.sum(mean_proportions_scaled)
    # lvm_logger.get_logger().debug("Sum of scaled mean proportions:", sum_mean_proportions_scaled)
    # lvm_logger.get_logger().debug("------------------------------------------------")
    normalized_mean_proportions = mean_proportions_scaled / sum_mean_proportions_scaled
    # for index, normalized_mean_proportion in enumerate(normalized_mean_proportions):
    #     lvm_logger.get_logger().debug(
    #         f"Normalized mean proportions LV{index +1} | {logical_volumes_info_list[index]['lv_uuid']}:", normalized_mean_proportion)
    return normalized_mean_proportions


def process_logical_volumes(lvm_logger: Logger, logical_volumes_informations_list: list):
    for index, logical_volume_informations in enumerate(logical_volumes_informations_list):
        lv_uuid = logical_volume_informations["lv_uuid"]
        usage_serie = logical_volume_informations["usage_serie"]
        file_system_size = logical_volume_informations["file_system_size"]

        # logical volume current usage percentage
        current_used_volume_percentage = usage_serie[-1] * \
            100 / file_system_size
        if current_used_volume_percentage > 80:
            # give the logical volume a superieur priority
            if logical_volume_informations["priority"] > 1:
                logical_volumes_informations_list[index][
                    "priority"] = logical_volume_informations["priority"] - 1
                lvm_logger.get_logger().debug(
                    f"Giving more priority to {lv_uuid}: {logical_volume_informations['priority']}")

        # get logical volume future usage prediction
        prediction = predict(lv_uuid, usage_serie.copy())

        # load the logical volume scaler
        scaler = load_scaler(lv_uuid=lv_uuid)
        # reverse transform logical volume prediction
        reverse_transformed_prediction = np.round(
            reverse_transform(scaler, prediction).flatten()[0]
        )
        allocation_reclaim_size = reverse_transformed_prediction - file_system_size
        # # calculate the difference between the latest usage volume size and the prediction
        # # to observe if the prediction is about extending or reducing
        logical_volumes_informations_list[index]["twenty_percent_of_prediction"] = int(
            np.round(0.2 * reverse_transformed_prediction))
        logical_volumes_informations_list[index]["prediction_size"] = reverse_transformed_prediction
        logical_volumes_informations_list[index]["allocation_reclaim_size"] = allocation_reclaim_size
    return logical_volumes_informations_list


def process_volume_groups(session: Session, lvm_logger: Logger):
    """
    Make decisions based on predictions, mean proportions, and allocation volume.

    Args:
    - session (Session): The session object for database operations.
    - allocation_volume (int): Max allocated volume.

    Returns:
    - None
    """
    # Get VG data from the database
    volume_groups_informations_list = get_volume_groups_info_list(session)
    adjustments = []
    # Step 1: Process each VG
    for index, volume_group_informations in enumerate(volume_groups_informations_list):
        volume_groups_informations_list[index]["logical_volumes"] = process_logical_volumes(
            lvm_logger, volume_group_informations["logical_volumes"])
        vg_free = volume_group_informations["vg_free"]
        logical_volumes_adjustments_sizes = calculate_allocations_volumes(
            lvm_logger,
            volume_groups_informations_list[index],
            vg_free
        )
        adjustments.append({
            "vg_name": volume_group_informations["vg_name"],
            "logical_volumes_adjustments_sizes": logical_volumes_adjustments_sizes
        })

    return adjustments


def calculate_total_allocations_reclaims_needed(twenty_percent_of_predictions, allocation_reclaim_sizes, mean_proportions):
    total_allocation_needed = []
    for twenty_percent_of_prediction, allocation_reclaim_size, mean_proportion in zip(twenty_percent_of_predictions, allocation_reclaim_sizes, mean_proportions):

        allocation_reclaim_size = allocation_reclaim_size
        if allocation_reclaim_size >= 0:
            total_allocation_needed.append(
                (mean_proportion * allocation_reclaim_size) + twenty_percent_of_prediction)

        else:
            total_allocation_needed.append(
                (mean_proportion * twenty_percent_of_prediction) +
                twenty_percent_of_prediction
            )

    return sum(total_allocation_needed)


def calculate_allocations_volumes(lvm_logger: Logger, volume_group_informations: list, allocation_volume: int):
    # Step 1: process logical volumes for (predictions, possible allocations/reclaims)
    lvm_logger.get_logger().debug(
        f"**************************************   {volume_group_informations['vg_name']}   *****************************************")
    logical_volumes_informations_list = [
        logical_volume_informations for logical_volume_informations in volume_group_informations["logical_volumes"]]
    allocation_reclaim_sizes = [
        logical_volume_informations["allocation_reclaim_size"] for index, logical_volume_informations in enumerate(logical_volumes_informations_list)]
    predictions = [
        logical_volume_informations["prediction_size"] for index, logical_volume_informations in enumerate(logical_volumes_informations_list)]
    twenty_percent_of_predictions = [
        logical_volume_informations["twenty_percent_of_prediction"] for index, logical_volume_informations in enumerate(logical_volumes_informations_list)]

    # Step 2: Calculate mean proportions scaled by each logical volume priority
    # The mean proportions scaled by logical volume priority refer to adjusting
    # the mean proportions based on the priority of each logical volume.
    # This adjustment is done to take into account the priority levels
    # when calculating how resources or allocations are distributed among different logical volumes
    mean_proportions: list[float] = calculate_proportions_by_priority(
        [
            {
                "lv_uuid": logical_volume_informations["lv_uuid"],
                "priority": logical_volume_informations["priority"],
                "usage_serie": logical_volume_informations["usage_serie"]
            }
            for logical_volume_informations
            in logical_volumes_informations_list
        ]
    )

    # Calculate the total allocation/reclaims needed
    total_allocations_reclaims_needed = calculate_total_allocations_reclaims_needed(
        twenty_percent_of_predictions.copy(),
        allocation_reclaim_sizes.copy(),
        mean_proportions.copy()
    )

    # the allocation factor helps avoid a situation where one logical volume
    # uses an excessive portion of the available allocation volume.
    # It prevents any single logical volume from monopolizing resources,
    # ensuring a fair and balanced distribution of the allocation among all logical volumes.
    if total_allocations_reclaims_needed == 0:
        allocation_factor = 1
    else:
        allocation_factor = allocation_volume / total_allocations_reclaims_needed
    lvm_logger.get_logger().debug(f"Allocation factor: {allocation_factor}")

    # Step 5 : Calculate the allocation volume size for each logical volume
    logical_volumes_adjustments_sizes: list[dict[str, int]] = []
    for logical_volume_informations, prediction, twenty_percent_of_prediction, allocation_reclaim_size, mean_proportion in zip(logical_volumes_informations_list, predictions, twenty_percent_of_predictions, allocation_reclaim_sizes, mean_proportions):
        lvm_logger.get_logger().debug(
            f"--------------------------------------------------------------------------------")
        file_system_size = logical_volume_informations["file_system_size"]
        prediction_size = int(prediction)
        if allocation_reclaim_size < 0:
            # logical volume prediction < file system size
            # giving up volume

            # logical volume future size
            size = int(np.round(
                (twenty_percent_of_prediction * mean_proportion) + twenty_percent_of_prediction))
            + allocation_reclaim_size
            # logical volume future size, scaled by the allocation factor
            size_scaled = int(np.round(
                (twenty_percent_of_prediction * mean_proportion * allocation_factor) + twenty_percent_of_prediction))
            + allocation_reclaim_size
        else:
            # logical volume prediction > file system size
            # requesting for more volume
            size = int(np.round(
                (mean_proportion * allocation_reclaim_size) + twenty_percent_of_prediction))
            + allocation_reclaim_size
            size_scaled = int(np.round(
                (allocation_factor * mean_proportion * allocation_reclaim_size) + twenty_percent_of_prediction))
            + allocation_reclaim_size

        if size >= 0:
            # When the future_size is positive,
            # we opt for the smaller value between it and the scaled future size
            # considering that the allocation factor could be greater than 1
            # leading to a scenario of over-sizing
            free_space_addition = int(min(
                size, size_scaled))
        else:
            # When the future_size is negative,
            # we opt for the larger value between it and the scaled future size
            # considering that the allocation factor could be greater than 1
            # leading to a scenario of over-decreasing
            free_space_addition = int(max(
                size, size_scaled))

        if free_space_addition >= 0:
            logical_volumes_adjustment_size = int(prediction_size +
                                                  free_space_addition - file_system_size)
            logical_volume_future_size = prediction_size + \
                free_space_addition
        else:
            logical_volumes_adjustment_size = int(free_space_addition)
            logical_volume_future_size = int(abs(
                allocation_reclaim_size) - abs(free_space_addition) + prediction_size)

        # Update allocation_volume by subtracting the allocated/reclaimed volume
        allocation_volume -= logical_volumes_adjustment_size

        lvm_logger.get_logger().debug(
            f"File system size : {file_system_size} | "
            f"Prediction: {prediction_size} | "
            f"Future LV size: {logical_volume_future_size} | "
            f"Adjustment size: {logical_volumes_adjustment_size} | "
            f"Priority: {logical_volume_informations['priority']} | "
            f"Mean proportion: {mean_proportion}"
        )
        logical_volumes_adjustments_sizes.append(
            {"lv_uuid": logical_volume_informations["lv_uuid"],
                "size": logical_volumes_adjustment_size}
        )
        lvm_logger.get_logger().debug(
            f"Allocation volume size with max limited allocation of {allocation_volume} MiB: {logical_volumes_adjustment_size} MiB"
        )
        lvm_logger.get_logger().debug(
            "----------------------------------------------------------------------------------")

    lvm_logger.get_logger().debug(f"Allocation volume: {allocation_volume}")
    lvm_logger.get_logger().debug(
        "----------------------------------------------------------------------------------")
    return logical_volumes_adjustments_sizes


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
