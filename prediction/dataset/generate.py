import uuid
import random
import csv
from datetime import datetime, timedelta
from root_directory import root_directory

# Function to generate random usage data for a logical volume


def generate_volume_data(total_capacity, previous_used_space, is_spike):
    pourcentage = 0.05
    # Change can be up to 5% of total capacity
    change_range = max(1, int(total_capacity * pourcentage))
    if is_spike:
        # Introduce a sudden spike (e.g., 25% increase)
        change = random.randint(change_range, change_range * 5)
    else:
        change = random.randint(-change_range, change_range)
    used_space = max(0, min(total_capacity, previous_used_space + change))
    free_space = total_capacity - used_space
    return free_space, used_space


# Function to generate usage history for a logical volume
def generate_usage_history(uuid, lv_id, volume_name, fstype, mount_point, priority, disks_capacity, previous_used_space, number_of_minutes, spike_hour):
    history = []
    current_date = datetime.now()

    for hour in range(int(number_of_minutes/10)):
        is_spike = (hour+1) % spike_hour == 0
        free_space, used_space = generate_volume_data(
            disks_capacity, previous_used_space, is_spike)

        entry = {
            "uuid": uuid,
            "logical_volume_id": lv_id,
            "name": volume_name,
            "created_at": current_date.strftime("%Y-%m-%d %H:%M:%S"),
            "priority": priority,
            "fstype": fstype,
            "mount_point": mount_point,
            "file_system_size": disks_capacity,
            "file_system_available_size": free_space,
            "file_system_used_size": used_space
        }
        history.append(entry)
        current_date -= timedelta(minutes=10)

        # Update previous_used_space for the next iteration
        previous_used_space = used_space
    return history


# Function to write data to a CSV file
def write_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["uuid",
                      "logical_volume_id",
                      "name",
                      "created_at",
                      "fstype",
                      "priority",
                      "mount_point",
                      "file_system_size",
                      "file_system_available_size",
                      "file_system_used_size"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)


# Main script
if __name__ == "__main__":
    dataset = []
    # 10 logical volumes
    number_of_lv = 3
    # disks_capacities = [random.randint(
    #     1000, 90000) for _ in range(number_of_lv)]
    uuids = [
        "kBHN6m-Rjk6-o0YS-j5rw-O5a8-zx47-NdfUOV",
        "6kC4RB-YmIQ-XT2Y-zKuK-vDhB-ACTf-1vTcDM",
        "WQq3J9-GZNm-sLaa-PrV1-JO2H-UsIm-1lVZ6p",
        "IMUZO6-dxCS-01Sy-k4sg-8yLm-Q35Q-IArvVk",
        "ZTPRAX-O9Ii-xAU1-A2bY-bhHc-qO2f-efnUVK"
    ]

    priorities = [4, 4, 5, 6, 6]
    ids = [1, 2, 3, 4, 5]
    mountpoints = [51, 52, 53, 54, 55]
    for index, disk_capacity in enumerate([452, 452, 500, 452, 281]):
        uuid = uuids[index]
        volume_name = f"lv{index + 1}"
        volume_history = generate_usage_history(
            uuid, index+1, volume_name, index+1, index+1, priorities[index], disk_capacity, previous_used_space=random.randint(0, disk_capacity), number_of_minutes=60*24*30*12, spike_hour=random.randint(1, 10*6*24))
        dataset.extend(volume_history)

    write_to_csv(
        dataset, f"{root_directory}/prediction/dataset/logical_volume_usage_history.csv")
