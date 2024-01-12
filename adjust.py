from ansible.resize import adjust_logical_volume
from connect import connect
from logs.Logger import Logger
from database.models import (
    Adjustment,
    Host,
    Segment,
    LogicalVolume,
    VolumeGroup
)

# Run the adjustments for each logical volume
# This function is called by the adjustments executor thread
# Ensring that the adjustments are executed separately from the data collection and processing


def adjust(ansible_logger: Logger, db_logger: Logger, adjustments_ids: list[int]):
    # [
    #       {
    #           "vg_name": "vg1",
    #           "lv_name": "lv1",
    #           "mountpoint": "/home/bob/Desktop/lvm/vg1/lv1",
    #           "file_system_type": "ext2",
    #           "size": "+314M"
    #       },
    #       {
    #           "vg_name": "vg1",
    #           "lv_name": "lv2",
    #           "mountpoint": "/home/bob/Desktop/lvm/vg1/lv2",
    #           "file_system_type": "ext3",
    #           "size": "652M"
    #       },
    # ]

    # Get the adjustments from the database
    # and sort them by created_at in descending order (older adjustments first)
    session = connect(db_logger)
    rows = session.query(Adjustment)\
        .filter(Adjustment.id.in_(adjustments_ids), Adjustment.status == "pending")\
        .join(LogicalVolume)\
        .join(Segment)\
        .join(VolumeGroup)\
        .join(Host)\
        .order_by(Adjustment.created_at.desc())\
        .all()
    if len(rows) == 0:
        ansible_logger.get_logger().info(
            "No adjustments to be executed at this time")
        return
    else:
        adjustments = []
        vg_name = rows[0].logical_volume.segments[0].volume_group.info.vg_name
        hostname = rows[0].logical_volume.segments[0].host.hostname
        volume_group = {
            "hostname": hostname,
            "vg_name": vg_name,
        }
        for row in rows:
            lv_name = row.logical_volume.info.lv_name
            file_system_type = row.logical_volume.stats[-1].file_system.file_system_type
            mountpoint = row.logical_volume.stats[-1].mount_point.path
            size = row.size
            logical_volume_dict = {
                "lv_name": lv_name,
                "vg_name": vg_name,
                "file_system_type": file_system_type,
                "mountpoint": mountpoint,
                "size": f"+{size}M" if size > 0 else f"{size}M"
            }
            row.status = "done"
            # Add the adjustment to the adjustments list
            adjustments.append(logical_volume_dict)
            # Add the adjustment to the database
            session.add(row)
            ansible_logger.get_logger().info(
                f"Making adjustments on ({hostname}/{vg_name}/{lv_name}): {size} MiB")
        # Commit the changes to the database
        session.commit()
        # Add the adjustments to the volume group dictionary
        volume_group["logical_volumes"] = adjustments
        # Execute the adjustments
        adjust_logical_volume(ansible_logger, hostname, volume_group)
