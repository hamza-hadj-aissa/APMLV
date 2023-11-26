import datetime
import json
import logging
import math
import subprocess
from numpy import double
import pandas as pd

from exceptions.LvmCommandError import LvmCommandError
from logs.Logger import Logger

# output volumes size unit for pvs, vgs and lvs commands
SIZE_UNIT = "m"


def parse_segment_ranges(df: pd.DataFrame):
    for index, lv in df.iterrows():
        le_ranges_array = lv["seg_le_ranges"][0].split(":")
        pv_name = le_ranges_array[0]
        ranges_array = le_ranges_array[1].split("-")
        df.loc[index, "pv_name"] = pv_name
        df.loc[index, "segment_range_start"] = int(ranges_array[0])
        df.loc[index, "segment_range_end"] = int(ranges_array[1])
    df.drop("seg_le_ranges", axis=1, inplace=True)
    return df


# remove any trailing charachters "m", "k"...
def parse_size_number(size):
    # 1024.00m --> 1024
    size = str(size).replace(SIZE_UNIT, "")
    return math.ceil(double(size))


def convert_bytes_to_mib(bytes_size: int):
    return bytes_size / (1024 ** 2)


def run_command(command_array, lvm_logger: Logger):
    lvm_logger.get_logger().info("Scraping LVM statistics...")
    command_result = subprocess.run(
        command_array, capture_output=True, text=True)
    try:
        if command_result.returncode == 0:
            return json.loads(command_result.stdout.strip())
        else:
            # mostly, it caused by permission access
            raise LvmCommandError(
                f"{command_array[1]} command not executed. Permission denied")
    except subprocess.CalledProcessError as e:
        # Handle Permission Denied error
        print(f"Permission Denied: {e}")
    except json.JSONDecodeError as e:
        # Handle JSON decoding error
        print(f"JSON Decoding Error: {e}")

    # Return an empty dictionary if the command fails
    return {}
