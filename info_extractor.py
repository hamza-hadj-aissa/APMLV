import subprocess
import pandas as pd
import json

logical_volumes_command_array = ["sudo", "lvs", "-o", "lv_uuid,lv_name,lv_size,lv_full_name,lv_tags,lv_role,seg_count,vg_name",
                                 "--units", "k", "--reportformat", "json"]
logical_volumes_command = subprocess.run(
    logical_volumes_command_array, capture_output=True, text=True)

if logical_volumes_command.returncode == 0:
    json_format_output = json.loads(
        logical_volumes_command.stdout.__str__().strip())['report'][0]['lv']
    df = pd.DataFrame.from_dict(
        json_format_output)
    print(df)


volume_group_command_array = ["sudo", "vgs", "-o", "vg_uuid,vg_name,vg_size,vg_free,vg_tags,lv_count,pv_count,vg_permissions,pv_name",
                              "--units", "k", "--reportformat", "json"]
volume_group_command = subprocess.run(
    volume_group_command_array, capture_output=True, text=True)

if volume_group_command.returncode == 0:
    json_format_output = json.loads(
        volume_group_command.stdout.__str__().strip())['report'][0]['vg']
    df = pd.DataFrame.from_dict(
        json_format_output)
    print(df)


physical_volume_command_array = ["sudo", "pvs", "-o", "pv_uuid,pv_name,pv_size,pv_free,pv_used,pv_allocatable,dev_size,pv_missing,pv_tags",
                                 "--units", "k", "--reportformat", "json"]
physical_volume_command = subprocess.run(
    physical_volume_command_array, capture_output=True, text=True)

if physical_volume_command.returncode == 0:
    json_format_output = json.loads(
        physical_volume_command.stdout.__str__().strip())['report'][0]['pv']
    df = pd.DataFrame.from_dict(
        json_format_output)
    print(df)


logical_volumes_fs_command_array = [
    'lsblk', '/dev/vg1/lg1', '-o', 'NAME,FSTYPE', '--json']
logical_volumes_fs_command = subprocess.run(
    logical_volumes_fs_command_array, capture_output=True, text=True)

if logical_volumes_fs_command.returncode == 0:
    json_format_output = json.loads(
        logical_volumes_fs_command.stdout.__str__().strip())['blockdevices']
    df = pd.DataFrame.from_dict(
        json_format_output)
    print(df)
else:
    print(logical_volumes_fs_command.stderr)
