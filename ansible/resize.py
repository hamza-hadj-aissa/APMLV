import json
from ansible.extract import AnsibleException
from root_directory import root_directory
from ansible_runner import run


# TODO: Handle the output of the adjust_logical_volumes.yml playbook
# TODO: Save the result of each adjustment in the database


def adjust_logical_volumes(hostname: str, logical_volume: dict):
    with open(f"{root_directory}/ansible/inventory", 'r') as f:
        # remove empty lines
        hosts_passwords = [line.strip()
                           for line in f if line.strip().split(" ")[0] == hostname]
    if len(hosts_passwords) == 0:
        print("Host not found")
        return
    else:
        # Run the playbook
        r = run(
            # Path to the playbook
            playbook=f"{root_directory}/ansible/playbooks/resize_logical_volume.yml",
            # Target hosts
            inventory="".join(hosts_passwords),
            # list of logical volumes to adjust
            extravars={
                "host": hosts_passwords[0],
                "logical_volume": logical_volume,
            },
            # quiet=True,
            # json_mode=True
        )
        host_informations = {}
        lvm_informations = {}
        # Check if the playbook failed to execute for the host
        if any(event['event'] == 'runner_on_unreachable' for event in r.events):
            print(
                f"Failed to extract lvm informations from ({hostname})")
            raise AnsibleException(
                message=f"Failed to connect to host ({hostname}). Check your SSH connection or host settings.")

        print(
            f"Extraction of lvm informations from ({hostname}) completed.")
        # Parse the output and extract the informations
        for each_host_event in r.events:
            # Check if the event is a runner_on_ok or runner_on_failed
            if each_host_event['event'] in ['runner_on_ok', 'runner_on_failed']:
                # Check if the task is "Display gathered informations"
                # print(each_host_event['event_data'])
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


logical_volumes = {
    "logical_volumes": [
        {
            "vg_name": "vg1",
            "lv_name": "lv1",
            "mountpoint": "/home/bob/Desktop/lvm/vg1/lv1",
            "file_system_type": "ext2",
            "size": "+100M"
        },
        {
            "vg_name": "vg1",
            "lv_name": "lv2",
            "mountpoint": "/home/bob/Desktop/lvm/vg1/lv2",
            "file_system_type": "f2fs",
            "size": "+260M"
        },
        {
            "vg_name": "vg2",
            "lv_name": "lv3",
            "mountpoint": "/home/bob/Desktop/lvm/vg2/lv3",
            "file_system_type": "vfat",
            "size": "+30M"
        },
        {
            "vg_name": "vg2",
            "lv_name": "lv4",
            "mountpoint": "/home/bob/Desktop/lvm/vg2/lv4",
            "file_system_type": "btrfs",
            "size": "-210M"
        },
        {
            "vg_name": "vg2",
            "lv_name": "lv5",
            "mountpoint": "/home/bob/Desktop/lvm/vg2/lv5",
            "file_system_type": "ext4",
            "size": "+150M"
        },
    ]
}

if __name__ == "__main__":
    adjust_logical_volumes(
        "bob@192.168.1.76",
        logical_volume={
            "vg_name": "vg2",
            "lv_name": "lv5",
            "mountpoint": "/home/bob/Desktop/lvm/vg2/lv5",
            "file_system_type": "ext4",
            "size": "+150M"
        }
    )
