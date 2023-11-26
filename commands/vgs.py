import pandas as pd
from commands.utils import SIZE_UNIT, parse_segment_ranges, parse_size_number, run_command
from exceptions.LvmCommandError import LvmCommandError


def get_group_volumes():
    # Command to retrieve information about volume groups
    volume_group_command_array = ["sudo", "vgs", "-o", "vg_uuid,vg_name,vg_size,vg_free,lv_count,pv_count,vg_permissions,pv_name",
                                  "--units", SIZE_UNIT, "--reportformat", "json_std"]
    try:
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
    except LvmCommandError as e:
        raise e
