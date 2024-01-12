from ansible.extract import extract_lvm_informations_from_host
from database.utils import insert_lvm_data_to_database, transform_lvm_data_to_dataframe
from sqlalchemy.orm import Session
from logs.Logger import Logger
from process import process
from root_directory import root_directory


def collect(session: Session, lvm_logger: Logger, db_logger: Logger, ansible_logger: Logger, host_password, volume_group: dict):
    logical_volumes_names = [lv["lv_name"]
                             for lv in volume_group["logical_volumes"]]
    physical_volumes_names = [pv['pv_name']
                              for pv in volume_group["physical_volumes"]]
    vg_name = volume_group["vg_name"]
    try:
        # Extract the lvm informations from the host for the volume group
        host_informations = extract_lvm_informations_from_host(
            ansible_logger=ansible_logger,
            playbook=f"{root_directory}/ansible/playbooks/extract_lvm_data.yml",
            host=host_password,
            extravars={
                'logical_volumes': " ".join([f'/dev/mapper/{vg_name}-{lv_name}' for lv_name in logical_volumes_names]),
                'volume_groups': vg_name,
                'physical_volumes': " ".join(physical_volumes_names),
            }
        )
        # Transform the lvm informations to a dataframe
        hostname, lvm_dataframe = transform_lvm_data_to_dataframe(
            host_informations, {lv['lv_name']: lv['priority']
                                for lv in volume_group["logical_volumes"]}
        )
        # Insert the lvm dataframe to the database
        insert_lvm_data_to_database(
            session, hostname, lvm_dataframe)
        # Process the lvm data for the volume group
        process(
            session=session,
            lvm_logger=lvm_logger,
            db_logger=db_logger,
            ansible_logger=ansible_logger,
            hostname=hostname,
            volume_group=volume_group
        )
    except Exception as e:
        lvm_logger.get_logger().error(e)
