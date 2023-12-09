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
def generate_usage_history(volume_id, volume_name, fstype, priority, disks_capacity, previous_used_space, number_of_minutes, spike_hour):
    history = []
    current_date = datetime.now()

    for hour in range(int(number_of_minutes/10)):
        is_spike = (hour+1) % spike_hour == 0
        free_space, used_space = generate_volume_data(
            disks_capacity, previous_used_space, is_spike)

        entry = {
            "uuid": volume_id,
            "name": volume_name,
            "date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
            "fstype": fstype,
            "priority": priority,
            "total_capacity": disks_capacity,
            "free_space": free_space,
            "used_space": used_space
        }
        history.append(entry)
        current_date -= timedelta(minutes=10)

        # Update previous_used_space for the next iteration
        previous_used_space = used_space
    return history


# Function to write data to a CSV file
def write_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        fieldnames = ["uuid", "name", "date",
                      "total_capacity", "free_space", "used_space"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)


# Main script
if __name__ == "__main__":
    dataset = []
    # 10 logical volumes
    number_of_lv = 3
    disks_capacities = [random.randint(
        1000, 90000) for _ in range(number_of_lv)]
    print(disks_capacities)
    counter = 0
    uuids = ["G05dru-uGpm-BOuB-b53j-ZWQb-sJhv-NdvNjg",
             "Gwys5G-B3Zc-qlg3-tUHF-O2J2-Rvvm-OpeuZs", "fVepbE-JBEw-IKaa-7svK-P56K-Pi9X-9WyIGZ"]
    for disk_capacity in [974, 974, 974]:
        volume_id = uuids[counter]
        volume_name = f"Volume_{counter + 1}"
        counter = counter + 1
        volume_history = generate_usage_history(
            volume_id, volume_name, disk_capacity, number_of_minutes=60*24*30*12, spike_hour=random.randint(1, 1440))
        dataset.extend(volume_history)

    write_to_csv(
        dataset, f"{root_directory}/prediction/dataset/logical_volume_usage_history.csv")