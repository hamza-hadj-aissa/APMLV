import pandas as pd
from commands.lvs import get_logical_volumes
from commands.utils import convert_bytes_to_mib, parse_size_number, run_command
from logs.Logger import Logger


def get_lv_lsblk(lvm_logger: Logger):
    # Retrieve logical volume information
    lvs_df = get_logical_volumes(lvm_logger)
    lv_fs_array = []
    # Iterate through unique logical volumes and retrieve filesystem information
    for index, row in lvs_df[['lv_uuid', 'lv_path']].drop_duplicates().iterrows():
        # Command to retrieve filesystem information for a logical volume
        logical_volumes_fs_command_array = [
            'lsblk', row['lv_path'], '-o', 'NAME,FSTYPE,FSSIZE,FSUSED,FSAVAIL,MOUNTPOINT', '--bytes', '--json']
        lvm_logger.get_logger().info(" ".join(logical_volumes_fs_command_array))
        json_format_output = run_command(logical_volumes_fs_command_array, lvm_logger)[
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
