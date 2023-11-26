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


def get_physical_volumes():
    # Command to retrieve information about physical volumes
    physical_volume_command_array = ["sudo", "pvs", "-o", "pv_uuid,pv_name,pv_size,pv_free,pv_used,pv_allocatable,dev_size,pv_missing,vg_uuid",
                                     "--units", SIZE_UNIT, "--reportformat", "json_std"]
    # Extract relevant information from the JSON output
    json_format_output = run_command(physical_volume_command_array)[
        'report'][0]['pv']
    # Create a DataFrame from the extracted information
    df = pd.DataFrame.from_dict(json_format_output)
    # remove any trailing charachters "m", "k"...
    df["pv_size"] = df["pv_size"].apply(
        lambda x: parse_size_number(x))
    df["pv_free"] = df["pv_free"].apply(
        lambda x: parse_size_number(x))
    df["pv_used"] = df["pv_used"].apply(
        lambda x: parse_size_number(x))
    return df


def get_logical_volumes():
    # Command to retrieve information about logical volumes
    logical_volumes_command_array = ["sudo", "lvs", "-o", "lv_uuid,lv_name,vg_name,lv_size,seg_size,seg_le_ranges,lv_path",
                                     "--units", SIZE_UNIT, "--reportformat", "json_std"]
    # Extract relevant information from the JSON output
    json_format_output = run_command(logical_volumes_command_array)[
        'report'][0]['lv']
    # Create a DataFrame from the extracted information
    df = pd.DataFrame.from_dict(json_format_output)

    # remove any trailing charachters "m", "k"...
    df["lv_size"] = df["lv_size"].apply(
        lambda x: parse_size_number(x))
    df["seg_size"] = df["seg_size"].apply(
        lambda x: parse_size_number(x))
    df = parse_segment_ranges(df)
    return df


def get_group_volumes():
    # Command to retrieve information about volume groups
    volume_group_command_array = ["sudo", "vgs", "-o", "vg_uuid,vg_name,vg_size,vg_free,lv_count,pv_count,vg_permissions,pv_name",
                                  "--units", SIZE_UNIT, "--reportformat", "json_std"]
    # Extract relevant information from the JSON output
    json_format_output = run_command(volume_group_command_array)[
        'report'][0]['vg']
    # Create a DataFrame from the extracted information
    df = pd.DataFrame.from_dict(json_format_output)
    # remove any trailing charachters "m", "k"...
    df["vg_size"] = df["vg_size"].apply(
        lambda x: parse_size_number(x))
    df["vg_free"] = df["vg_free"].apply(
        lambda x: parse_size_number(x))
    return df


def get_lv_lsblk():
    # Retrieve logical volume information
    lvs_df = get_logical_volumes()
    lv_fs_array = []
    # Iterate through unique logical volumes and retrieve filesystem information
    for index, row in lvs_df[['lv_uuid', 'lv_path']].drop_duplicates().iterrows():
        # Command to retrieve filesystem information for a logical volume
        logical_volumes_fs_command_array = [
            'lsblk', row['lv_path'], '-o', 'NAME,FSTYPE,FSSIZE,FSUSED,FSAVAIL,MOUNTPOINT', '--bytes', '--json']
        json_format_output = run_command(logical_volumes_fs_command_array)[
            'blockdevices'][0]
        json_format_output['lv_uuid'] = row['lv_uuid']
        lv_fs_array.append(json_format_output)

    # Create a DataFrame from the extracted filesystem information
    lv_fs_dataframe = pd.DataFrame.from_dict(lv_fs_array)
    # Filter out rows where 'mountpoint' is not null, drop NaN values
    filtered_lv_fs_dataframe = lv_fs_dataframe.dropna(subset=['mountpoint'])

    # Merge logical volume and filesystem DataFrames on 'lv_uuid'
    merged_df = pd.merge(lvs_df, filtered_lv_fs_dataframe,
                         "right", on='lv_uuid')
    # remove any trailing charachters "m", "k"...
    # convert bytes to MiB
    merged_df['fssize'] = merged_df['fssize'].apply(
        lambda x: convert_bytes_to_mib(parse_size_number(x)))
    merged_df['fsused'] = merged_df['fsused'].apply(
        lambda x: convert_bytes_to_mib(parse_size_number(x)))
    merged_df['fsavail'] = merged_df['fsavail'].apply(
        lambda x: convert_bytes_to_mib(parse_size_number(x)))
    return merged_df
