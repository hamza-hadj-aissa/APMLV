import json
import schedule
from Counter import Counter
from collect import collect
from connect import connect
from exceptions.LvmCommandError import LvmCommandError
from logs.Logger import Logger
from root_directory import root_directory
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


def start(session: Session, lvm_logger: Logger, db_logger: Logger, ansible_logger: Logger):
    # get the hosts from the logical_volumes.json file
    with open(f"{root_directory}/logical_volumes.json", 'r') as file:
        logical_volumes_json = json.load(file)
    hosts = [host for host in logical_volumes_json['hosts']]
    hostnames = [host['hostname'] for host in hosts]

    # get the hosts from the inventory file
    with open(f"{root_directory}/ansible/inventory", 'r') as f:
        # remove empty lines
        hosts_passwords = [line.strip()
                           for line in f if line.strip().split(" ")[0] in hostnames]
    if len(hosts_passwords) == 0:
        ansible_logger.get_logger().error(
            f"None of these hosts {(', '.join(hostnames))} found in the inventory file")
        exit(1)

    # Iterate over the hosts
    for host_password in hosts_passwords:
        volume_groups = [vg for host in hosts for vg in host["volume_groups"]
                         if host["hostname"] == host_password.split(" ")[0]]
        hostname = host_password.split(" ")[0]
        # Iterate over the volume groups
        for volume_group in volume_groups:
            counter = Counter(
                instance_id=f"{hostname}/{volume_group['vg_name']}"
            )
            # Collect the data for the volume group
            collect(
                session=session,
                lvm_logger=lvm_logger,
                db_logger=db_logger,
                ansible_logger=ansible_logger,
                host_password=host_password,
                volume_group=volume_group
            )
            # Increment the counter for the volume group
            counter.incrementCounter()


if __name__ == "__main__":
    # expressed in seconds
    # 60 * 5 = 5 minutes
    time_interval = 10 * 60
    log_file_path = f"{root_directory}/logs/lvm_balancer.log"
    # define loggers
    db_logger = Logger("Postgres", path=log_file_path)
    lvm_logger = Logger("LVM", path=log_file_path)
    ansible_logger = Logger("Ansible", path=log_file_path)
    main_logger = Logger("Main", path=log_file_path)
    main_logger.get_logger().info("Starting Lvm Balancer...")

    # connect to the database
    try:
        session = connect(db_logger)
    except SQLAlchemyError as e:
        db_logger.get_logger().error(e)
        exit(1)

    # process starts here --
    try:
        schedule\
            .every(time_interval)\
            .seconds\
            .do(
                job_func=start,
                session=session,
                lvm_logger=lvm_logger,
                ansible_logger=ansible_logger,
                db_logger=db_logger,
            )
        while True:
            schedule.run_pending()

    except KeyboardInterrupt:
        exit(2)
    except LvmCommandError as e:
        lvm_logger.get_logger().error(e)
    except SQLAlchemyError as e:
        db_logger.get_logger().error(e)
