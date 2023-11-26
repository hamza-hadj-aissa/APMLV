import pandas as pd
from commands import SIZE_UNIT, parse_segment_ranges, parse_size_number, run_command


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
