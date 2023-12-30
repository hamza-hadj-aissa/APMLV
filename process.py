import json
from adjust import adjust
from calculations import calculate_allocations_volumes, check_logical_volume_usage_thresholds, predict_logical_volume_future_usage
from Counter import Counter
from database.utils import get_logical_volumes_list_per_volume_group_per_host, insert_to_logical_volume_adjustment
from logs.Logger import Logger
from prediction.lstm import LOOKBACK
from root_directory import root_directory
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


# Process the logical volumes data of volume group of each host
def process(session: Session, lvm_logger: Logger, db_logger: Logger, ansible_logger: Logger, hostname: str, volume_group: dict):
    lvm_logger.get_logger().info(
        f"Processing logical volumes of {hostname}...")
    lvm_logger.get_logger().info(
        f"Processing logical volumes of {hostname}/{volume_group['vg_name']}..."
    )
    vg_name = volume_group['vg_name']
    logical_volumes_names = [lv["lv_name"]
                             for lv in volume_group["logical_volumes"]]
    try:
        volume_group_informations = get_logical_volumes_list_per_volume_group_per_host(
            session=session, hostname=hostname, vg_name=vg_name, logical_volumes_names=logical_volumes_names, lv_stats_limit=LOOKBACK
        )
    except SQLAlchemyError as error:
        lvm_logger.get_logger().error(
            f"Error while processing volume group {hostname}/{vg_name}: {error}"
        )
        return

    run_volume_group_calculations = False
    # Step 1: check the usage threshold for each logical volume
    for index in range(len(volume_group_informations['logical_volumes'])):
        volume_group_informations['logical_volumes'][index], run_calculations = check_logical_volume_usage_thresholds(
            lvm_logger, volume_group_informations['logical_volumes'][index]
        )
        # If at least one logical volume needs to be adjusted,
        # then all the logical volumes in the volume group needs to be adjusted
        if not run_volume_group_calculations and run_calculations:
            # Assign the run_calculations value to run_volume_group_calculations
            run_volume_group_calculations = run_calculations

    # Step 2: Run the calculations if needed
    counter = Counter(instance_id=f"{hostname}/{vg_name}")
    if counter.getCounter() >= LOOKBACK-1 or run_volume_group_calculations:
        for index in range(len(volume_group_informations['logical_volumes'])):
            volume_group_informations['logical_volumes'][index] = predict_logical_volume_future_usage(
                lvm_logger, volume_group_informations['hostname'], volume_group_informations[
                    'vg_name'], volume_group_informations['logical_volumes'][index]
            )

        # Step 3: Calculate mean proportions scaled by each logical volume priority
        if len(volume_group_informations) > 0:
            volume_group_adjustments = calculate_allocations_volumes(
                lvm_logger,
                volume_group_informations,
            )
            # Step 4: Process the adjustments
            adjustments_ids: list[int] = insert_to_logical_volume_adjustment(
                session=session, db_logger=db_logger, hostname=hostname, volume_group_adjustments=volume_group_adjustments.copy()
            )
            if len(adjustments_ids) > 0:
                # Step 5: Run the adjustments
                adjust(ansible_logger, db_logger, adjustments_ids)

        # Reset the counter to collect new data for another LOOKBACK intervals (10 Minutes * LOOKBACK)
        lvm_logger.get_logger().info(
            f"Resetting the counter to collect new data for another {LOOKBACK} intervals (10 Minutes x {LOOKBACK})"
        )
        counter.resetCounter()
    lvm_logger.get_logger().info(
        f"Processed logical volumes of {hostname}/{volume_group['vg_name']}"
    )
