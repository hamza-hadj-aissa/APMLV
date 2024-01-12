import json
from ansible_runner import run
from logs.Logger import Logger


class AnsibleException(Exception):
    def __init__(self, message="Ansible error occurred"):
        self.message = message
        super().__init__(self.message)

    def __eq__(self, __value: object) -> bool:
        return super().__eq__(__value)


# Extract informations from a host using ansible and return a dict with the host and the lvm informations
def extract_lvm_informations_from_host(ansible_logger: Logger, playbook: str, host: str, extravars: dict[str, str]):
    host_informations = {}
    lvm_informations = {}
    hostname = host.split(" ")[0]
    ansible_logger.get_logger().info(
        f"Extracting lvm informations from ({hostname})...")
    # try:
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
        extravars=extravars,
    )

    # Check if the playbook failed to execute for the host
    if any(event['event'] == 'runner_on_unreachable' for event in r.events):
        ansible_logger.get_logger().error(
            f"Failed to extract lvm informations from ({hostname})")
        raise AnsibleException(
            message=f"Failed to connect to host ({hostname}). Check your SSH connection or host settings.")

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
    return host_informations
    # except KeyboardInterrupt as e:
    #     raise e
