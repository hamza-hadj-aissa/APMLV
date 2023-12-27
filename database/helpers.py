import math
from numpy import double
import pandas as pd


# output volumes size unit for pvs, vgs and lvs commands
SIZE_UNIT = "m"


# Transform the seg_le_ranges into pv_name, segment_range_start and segment_range_end and drop the seg_le_ranges column
def parse_segment_ranges(df: pd.DataFrame):
    for index, lv in df.iterrows():
        # split the seg_le_ranges column into pv_name, segment_range_start and segment_range_end
        le_ranges_array = lv["seg_le_ranges"].split(":")
        pv_name = le_ranges_array[0]
        # split the segment range into start and end
        ranges_array = le_ranges_array[1].split("-")
        # update the dataframe
        df.loc[index, "pv_name"] = pv_name
        # convert the segment start range to int and update the dataframe
        df.loc[index, "segment_range_start"] = int(ranges_array[0])
        # convert the segment end range to int and update the dataframe
        df.loc[index, "segment_range_end"] = int(ranges_array[1])
    # drop the seg_le_ranges column
    df.drop("seg_le_ranges", axis=1, inplace=True)
    return df


# Remove any trailing charachters "m", "k"...
# and convert the size to int
def parse_size_number(size):
    # 1024.00m --> 1024
    if isinstance(size, float) or isinstance(size, double):
        return math.c(size)

    if size.endswith(SIZE_UNIT):
        size = size[:-1]
    # Convert to integer and round up
    try:
        size_int = math.ceil(float(size))
    except ValueError:
        raise ValueError(f"Invalid size string: {size}")

    return size_int
# convert bytes to mib


def convert_bytes_to_mib(bytes_size: int):
    # 1 MiB = 1024 KiB = 1048576 B
    return bytes_size / (1024 ** 2)
