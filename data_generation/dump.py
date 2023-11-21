import uuid
import random
import csv
from datetime import datetime, timedelta

# Function to generate random usage data for a logical volume


def generate_volume_data(total_capacity, previous_used_space):
    # Change can be up to 10% of total capacity
    change_range = max(1, int(total_capacity * 0.1))
    change = random.randint(-change_range, change_range)
    used_space = max(0, min(total_capacity, previous_used_space + change))
    free_space = total_capacity - used_space
    return free_space, used_space

# Function to generate usage history for a logical volume


def generate_usage_history(volume_id, volume_name, disks_capacity, number_of_hours):
    history = []
    current_date = datetime.now()

    previous_used_space = random.randint(
        0, disks_capacity)  # Initial used space

    for _ in range(number_of_hours):
        free_space, used_space = generate_volume_data(
            disks_capacity, previous_used_space)
        entry = {
            "uuid": volume_id,
            "name": volume_name,
            "date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
            "total_capacity": disks_capacity,
            "free_space": free_space,
            "used_space": used_space
        }
        history.append(entry)
        current_date -= timedelta(hours=1)

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
    number_of_lv = 10
    disks_capacities = [random.randint(
        10000, 999999) for _ in range(number_of_lv)]
    counter = 0
    for disk_capacity in disks_capacities:
        volume_id = str(uuid.uuid4())
        volume_name = f"Volume_{counter + 1}"
        counter = counter + 1
        volume_history = generate_usage_history(
            volume_id, volume_name, disk_capacity, 7*24)
        dataset.extend(volume_history)

    write_to_csv(dataset, "dataset/logical_volume_usage_history.csv")
