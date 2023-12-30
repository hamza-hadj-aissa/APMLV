from math import ceil
import numpy as np
import torch

from helpers import load_model, load_scaler, reverse_transform
from logs.Logger import Logger
from prediction.lstm import LOOKBACK


# Check if the usage of a logical volume is above 80% of its file system size
# If so, give it a higher priority
# and return the updated logical volume
# Otherwise, return the logical volume as is
# Also, return a boolean indicating if the calculations should be run
def check_logical_volume_usage_thresholds(lvm_logger: Logger, logical_volume: list):
    # This variable is used to determine if the calculations/adjustments
    # should be run after processing each logical volume
    run_calculations = False
    # Step 1: Check the usage threshold
    lv_uuid = logical_volume["lv_uuid"]
    usage_serie = logical_volume["usage_serie"]
    file_system_size = logical_volume["file_system_size"]
    lvm_logger.get_logger().info(
        f"Checking usage threshold for {logical_volume['lv_name']}..."
    )
    # logical volume current usage percentage
    current_used_volume_percentage = int(ceil(usage_serie[-1] *
                                              100 / file_system_size))
    lvm_logger.get_logger().info(
        f"Current usage percentage for {logical_volume['lv_name']}: {current_used_volume_percentage}%"
    )
    if current_used_volume_percentage > 80:
        run_calculations = True
        # give the logical volume a superieur priority
        if logical_volume["priority"] > 1:
            logical_volume["priority"] = logical_volume["priority"] - 1
            lvm_logger.get_logger().debug(
                f"Giving more priority to {lv_uuid}: {logical_volume['priority']}"
            )

    return logical_volume, run_calculations


# Calculate the mean proportions for each logical volume
# By mean proportions, we mean the mean of the proportions of each logical volume
# The proportions of a logical volume refer to the proportion of its usage
# compared to the total usage of all logical volumes
def calculate_mean_proportions(logical_volumes_info_list: list[dict[str, int]]) -> list[dict[str, str | int | float]]:
    usage_series = np.array(
        [usage_serie["usage_serie"] for usage_serie in logical_volumes_info_list])
    column_sums = usage_series.sum(axis=0)
    proportions = np.where(column_sums == 0, 0, usage_series / column_sums)
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


# Calculate the mean proportions scaled by each logical volume priority
# The mean proportions scaled by logical volume priority refer to adjusting
# the mean proportions based on the priority of each logical volume.
def calculate_proportions_by_priority(logical_volumes_info_list: list[dict[str, int | str | list[float]]]) -> list[float]:
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
    # Step 2: Normalize scaled proportions to ensure they sum up to 1
    sum_mean_proportions_scaled = np.sum(mean_proportions_scaled)
    normalized_mean_proportions = mean_proportions_scaled / sum_mean_proportions_scaled
    return normalized_mean_proportions


# Calculate the total allocation/reclaims needed
# The total allocation/reclaims needed refer to the total allocation/reclaims needed for all logical volumes
# This is calculated by summing the 20% of predictions and (the allocation/reclaim sizes * mean proportions) for each logical volume
# This is done to make a precalculation of the total consumtion/reclaim of the free allocation volume
def sum_of_allocations_reclaims_per_volume_group(twenty_percent_of_predictions, allocation_reclaim_sizes, mean_proportions):
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


# Calculate the allocation volume size for each logical volume
# This is done by calculating the mean proportions of the logical volume scaled by it's priority
# Then, it's scaled by the allocation factor, which helps avoid a situation where one logical volume
# uses an excessive portion of the available allocation volume.
def calculate_allocations_volumes(lvm_logger: Logger, volume_group_informations: dict):
    # Step 0: Get the free allocation volume size for the VG
    allocation_volume = volume_group_informations["vg_free"]
    # Step 1: process logical volumes for (predictions, possible allocations/reclaims)
    lvm_logger.get_logger().debug(
        f"**************************************   {volume_group_informations['vg_name']}   *****************************************")
    # Extract the logical volumes informations list
    logical_volumes_informations_list = [
        logical_volume_informations for logical_volume_informations in volume_group_informations["logical_volumes"]]
    # Extract the allocation/reclaim sizes
    allocation_reclaim_sizes = [
        logical_volume_informations["allocation_reclaim_size"] for index, logical_volume_informations in enumerate(logical_volumes_informations_list)]
    # Extract the predictions
    predictions = [
        logical_volume_informations["prediction_size"] for index, logical_volume_informations in enumerate(logical_volumes_informations_list)]
    # Extract the 20% of predictions
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
    total_allocations_reclaims_needed = sum_of_allocations_reclaims_per_volume_group(
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


# Make predictions for a logical volume based on its usage serie
# and return the logical volume with the prediction
def predict_logical_volume_future_usage(lvm_logger: Logger, hostname: str, vg_name: str, logical_volume: dict):
    lv_uuid = logical_volume["lv_uuid"]
    usage_serie = logical_volume["usage_serie"]
    # load model
    try:
        lvm_logger.get_logger().info(
            f"Loading model for {hostname}/{vg_name}/{logical_volume['lv_name']}..."
        )
        model = load_model(lv_uuid=lv_uuid)
        lvm_logger.get_logger().info(
            f"Model for {hostname}/{vg_name}/{logical_volume['lv_name']} loaded"
        )
    except FileNotFoundError:
        lvm_logger.get_logger().error(
            f"Model for {hostname}/{vg_name}/{logical_volume['lv_name']} not found"
        )
        # return None to indicate that the model was not found
        return None
    # load scaler
    try:
        lvm_logger.get_logger().info(
            f"Loading scaler for {hostname}/{vg_name}/{logical_volume['lv_name']}..."
        )
        scaler = load_scaler(lv_uuid=lv_uuid)
        lvm_logger.get_logger().info(
            f"Scaler for {hostname}/{vg_name}/{logical_volume['lv_name']} loaded"
        )
    except FileNotFoundError:
        lvm_logger.get_logger().error(
            f"Scaler for {hostname}/{vg_name}/{logical_volume['lv_name']} not found"
        )
        # return None to indicate that the scaler was not found
        return logical_volume

    # append a dummy value to the usage serie, since the model expects a sequence of length LOOKBACK+1
    # that +1 is supposed to be the label (real value)
    usage_serie.append(
        sum(usage_serie)/len(usage_serie)
    )

    # transform (scale) lv fs usage
    lvm_logger.get_logger().info(
        f"Transforming usage serie for {hostname}/{vg_name}/{logical_volume['lv_name']}..."
    )
    lvs_lastest_usage_series = np.array(
        np.array(usage_serie)
    ).reshape(1, LOOKBACK+1)
    lvs_lastest_usage_series_scaled = scaler.transform(
        lvs_lastest_usage_series)
    lv_fs_usage_tensor = torch.tensor(
        lvs_lastest_usage_series_scaled[:, :-1].reshape(-1, LOOKBACK, 1)).float()
    lvm_logger.get_logger().info(
        f"Usage serie for {hostname}/{vg_name}/{logical_volume['lv_name']} transformed"
    )
    lvm_logger.get_logger().info(
        f"Predicting usage serie for {hostname}/{vg_name}/{logical_volume['lv_name']}..."
    )
    # make prediction
    prediction = model(lv_fs_usage_tensor).to(
        "cpu").detach().numpy().flatten()
    # reverse transform logical volume prediction
    reverse_transformed_prediction = np.round(
        reverse_transform(scaler, prediction).flatten()[0]
    )
    lvm_logger.get_logger().info(
        f"Future usage for {hostname}/{vg_name}/{logical_volume['lv_name']} predicted: {reverse_transformed_prediction} MiB"
    )
    allocation_reclaim_size = reverse_transformed_prediction - \
        logical_volume["file_system_size"]
    # # calculate the difference between the latest usage volume size and the prediction
    # # to observe if the prediction is about extending or reducing
    logical_volume["twenty_percent_of_prediction"] = int(
        np.round(0.2 * reverse_transformed_prediction))
    logical_volume["prediction_size"] = reverse_transformed_prediction
    logical_volume["allocation_reclaim_size"] = allocation_reclaim_size
    return logical_volume
