import subprocess
import pandas as pd
import json


def get_lv_dataframe():
    # lvs -o lv_uuid,lv_name,vg_name,lv_size,seg_size,seg_le_ranges --units k --reportformat json_std
    logical_volumes_command_array = ["sudo", "lvs", "-o", "lv_uuid,lv_name,vg_name,lv_size,seg_size,seg_le_ranges,lv_path",
                                     "--units", "k", "--reportformat", "json_std"]
    logical_volumes_command = subprocess.run(
        logical_volumes_command_array, capture_output=True, text=True)

    if logical_volumes_command.returncode == 0:
        json_format_output = json.loads(
            logical_volumes_command.stdout.__str__().strip())['report'][0]['lv']
        df = pd.DataFrame.from_dict(
            json_format_output)
        return df


def get_vg_dataframe():
    # vgs -o vg_uuid,vg_name,vg_size,vg_free,lv_count,pv_count,vg_permissions,pv_name --units k --reportformat json_std
    volume_group_command_array = ["sudo", "vgs", "-o", "vg_uuid,vg_name,vg_size,vg_free,lv_count,pv_count,vg_permissions,pv_name",
                                  "--units", "k", "--reportformat", "json_std"]
    volume_group_command = subprocess.run(
        volume_group_command_array, capture_output=True, text=True)

    if volume_group_command.returncode == 0:
        json_format_output = json.loads(
            volume_group_command.stdout.__str__().strip())['report'][0]['vg']
        df = pd.DataFrame.from_dict(
            json_format_output)
        return df


def get_pv_dataframe():
    # pvs -o pv_uuid,pv_name,pv_size,pv_free,pv_used,pv_allocatable,dev_size,pv_missing --units k --reportformat json_std
    physical_volume_command_array = ["sudo", "pvs", "-o", "pv_uuid,pv_name,pv_size,pv_free,pv_used,pv_allocatable,dev_size,pv_missing",
                                     "--units", "k", "--reportformat", "json_std"]
    physical_volume_command = subprocess.run(
        physical_volume_command_array, capture_output=True, text=True)

    if physical_volume_command.returncode == 0:
        json_format_output = json.loads(
            physical_volume_command.stdout.__str__().strip())['report'][0]['pv']
        df = pd.DataFrame.from_dict(
            json_format_output)
        return df


def get_lv_fs_dataframe():
    # lsblk -o NAME,FSTYPE,FSSIZE,FSUSED,FSAVAIL,MOUNTPOINT /dev/vg/lg --bytes --json
    lv_df = get_lv_dataframe()
    lv_fs_array = []
    for index, row in lv_df[['lv_uuid', 'lv_path']].drop_duplicates().iterrows():
        logical_volumes_fs_command_array = [
            'lsblk', row['lv_path'], '-o', 'NAME,FSTYPE,FSSIZE,FSUSED,FSAVAIL,MOUNTPOINT', '--bytes', '--json']
        logical_volumes_fs_command = subprocess.run(
            logical_volumes_fs_command_array, capture_output=True, text=True)
        if logical_volumes_fs_command.returncode == 0:
            json_format_output = json.loads(
                logical_volumes_fs_command.stdout.__str__().strip())['blockdevices'][0]
            json_format_output['lv_uuid'] = row['lv_uuid']
            lv_fs_array.append(json_format_output)
    lv_fs_dataframe = pd.DataFrame.from_dict(
        lv_fs_array)
    filtered_lv_fs_dataframe = lv_fs_dataframe.where(
        lv_fs_dataframe['mountpoint'].notnull()).dropna()

    print(pd.merge(lv_df, filtered_lv_fs_dataframe, "right", on='lv_uuid').stack())


# print(get_lv_dataframe(), "\n")
# print(get_vg_dataframe(), "\n")
# print(get_pv_dataframe(), "\n")
print(get_lv_fs_dataframe())
