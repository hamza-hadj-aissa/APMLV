import pandas as pd
from commands.utils import SIZE_UNIT, parse_segment_ranges, parse_size_number, run_command


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
