import json
from ansible_runner import run
from logs.Logger import Logger


# Extract informations from a host using ansible and return a dict with the host and the lvm informations
def extract_lvm_informations_from_host(ansible_logger: Logger, playbook: str, host: str, extravars: dict[str, str]):
    host_informations = {}
    lvm_informations = {}
    hostname = host.split(" ")[0]
    ansible_logger.get_logger().info(
        f"Extracting lvm informations from ({hostname})...")
    try:
        # Run the playbook and get the output
        r = run(
            # Path to the playbook
            playbook=playbook,
            # Target hosts
            inventory=host,
            # Prevents the output from being displayed in the console
            quiet=True,
            # Output the result in json
            json_mode=True,
            # Extra variables to pass to the playbook
            extravars=extravars
        )
    except KeyboardInterrupt:
        ansible_logger.get_logger().error(
            f"Extraction of lvm informations from ({hostname}) interrupted.")
        return host_informations

    ansible_logger.get_logger().info(
        f"Extraction of lvm informations from ({hostname}) completed.")
    # Parse the output and extract the informations
    for each_host_event in r.events:
        # Check if the event is a runner_on_ok or runner_on_failed
        if each_host_event['event'] in ['runner_on_ok', 'runner_on_failed']:
            # Check if the task is "Display gathered informations"
            if (each_host_event['event_data']['task'] == "Display gathered informations" and each_host_event['event'] == "runner_on_ok"):
                # Parse the output
                output_json = json.loads(
                    each_host_event['event_data']['res']['msg']
                )
                # Get the host name
                host_informations['host'] = output_json['host']
                # Get the lvm informations
                for entry in ["LVs", "VGs", "PVs", "lsblk"]:
                    if entry == "lsblk":
                        lvm_informations['lsblk'] = output_json[entry]['blockdevices']
                    else:
                        lvm_informations[entry] = output_json[entry]["report"][0][entry.lower()[
                            :-1]]
                host_informations['lvm_informations'] = lvm_informations
        # else:
        #     # If the task failed, return an empty dict
        #     ansible_logger.get_logger().error(
        #         f"Failed to extract lvm informations from ({hostname}).")
    return host_informations
