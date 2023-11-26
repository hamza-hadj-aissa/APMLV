import json
import math
import subprocess
from numpy import double
import pandas as pd

# output volumes size unit for pvs, vgs and lvs commands
SIZE_UNIT = "m"


def parse_segment_ranges(df: pd.DataFrame):
    for index, lv in df.iterrows():
        le_ranges_array = lv["seg_le_ranges"][0].split(":")
        print(le_ranges_array)
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


def run_command(command_array):
    command_result = subprocess.run(
        command_array, capture_output=True, text=True)
    if command_result.returncode == 0:
        return json.loads(command_result.stdout.strip())
    return None
