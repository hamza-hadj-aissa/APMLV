from re import S
import numpy as np
import pandas as pd
from psycopg import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session
from database.helpers import convert_bytes_to_mib, parse_segment_ranges, parse_size_number
from database.models import (
    FileSystem,
    Host,
    LogicalVolumeStats,
    MountPoint,
    PhysicalVolumeStats,
    SegmentStats,
    VolumeGroup,
    VolumeGroupInfo,
    PhysicalVolume,
    PhysicalVolumeInfo,
    LogicalVolume,
    LogicalVolumeInfo,
    Segment,
    Priority,
    Adjustment,
    VolumeGroupStats
)
from logs.Logger import Logger
from connect import connect
from sqlalchemy.exc import SQLAlchemyError


class DuplicateRowError(Exception):
    pass


# generic function to insert host entity,
def insert_or_get_host_entity(session: Session, hostname: str):
    host: Host = session.query(Host).filter_by(hostname=hostname).first()
    if host is None:
        host = Host(hostname=hostname)
        session.add(host)
    return host


# generic function to insert volume entity,
def insert_volume_entity(session: Session, host_id: int, model: LogicalVolume | VolumeGroup | PhysicalVolume, info_model: LogicalVolumeInfo | VolumeGroupInfo | PhysicalVolumeInfo, name_column, uuid_column, name, uuid):
    # Check if name is already taken
    name_taken = session.query(model)\
        .join(getattr(model, 'info'))\
        .filter_by(**{name_column: name})\
        .join(Segment)\
        .filter_by(host_id_fk=host_id)\
        .first()

    if name_taken is not None:
        # Raise an error if name is already taken
        raise DuplicateRowError(f"Name already taken ({name})")
    else:
        # Check if uuid already exists in the database
        entity_exists = session.query(model)\
            .filter_by(**{uuid_column: uuid})\
            .join(Segment)\
            .filter_by(host_id_fk=host_id)\
            .first()

        # Check if info already exists in the database
        entity_info = session.query(model).join(info_model).filter_by(
            **{name_column: name}
        ).first()

        if entity_exists:
            # uuid exists
            if entity_info is None:
                # info doesn't exist, modify the existing entity's name
                entity_exists.info = info_model(**{name_column: name})
                return entity_exists
            else:
                # info exists, link it to the existing entity
                entity_exists.info = entity_info
                return entity_exists
        else:
            # uuid doesn't exist
            if entity_info is None:
                # info doesn't exist, create both entity and info
                return model(**{uuid_column: uuid, 'info': info_model(**{name_column: name})})
            else:
                # info exists, create entity and link it to info
                return model(**{uuid_column: uuid, f'{model.__tablename__}_info_id_fk': entity_info.id})


# Helper function to query a DB table
def get_volume_entity(session: Session, model: LogicalVolume | VolumeGroup | PhysicalVolume | None, hostname: str, **kwargs) -> LogicalVolume | VolumeGroup | PhysicalVolume | None:
    info_class = model.__mapper__.relationships.get('info').mapper.class_
    return session.query(model)\
        .join(info_class)\
        .join(Segment)\
        .join(Host)\
        .filter_by(hostname=hostname, **kwargs)\
        .first()


def get_or_create_volume(session: Session, host_id: int, model, info_model, name_column: str, uuid_column: str, name: str, uuid: str):
    if name_column == 'vg_name':
        volume_entity = session.query(VolumeGroup)\
            .filter_by(vg_uuid=uuid)\
            .join(VolumeGroupInfo)\
            .filter_by(vg_name=name)\
            .join(Segment)\
            .join(Host)\
            .filter_by(id=host_id)\
            .first()
    elif name_column == 'pv_name':
        volume_entity = session.query(PhysicalVolume)\
            .filter_by(pv_uuid=uuid)\
            .join(PhysicalVolumeInfo)\
            .filter_by(pv_name=name)\
            .join(Segment)\
            .join(Host)\
            .filter_by(id=host_id)\
            .first()
    elif name_column == 'lv_name':
        volume_entity = session.query(LogicalVolume)\
            .filter_by(lv_uuid=uuid)\
            .join(LogicalVolumeInfo)\
            .filter_by(lv_name=name)\
            .join(Segment)\
            .join(Host)\
            .filter_by(id=host_id)\
            .first()
    # If not found, create a new one
    if not volume_entity:
        volume_entity = insert_volume_entity(
            session, host_id, model, info_model, name_column, uuid_column, name, uuid)
    return volume_entity


# Helper function to get or insert a volume group
def insert_or_get_volume_group(session: Session, vg_uuid: str, vg_name: str, host_id: int):
    volume_group = get_or_create_volume(
        session, host_id, VolumeGroup, VolumeGroupInfo, "vg_name", "vg_uuid", vg_name, vg_uuid
    )
    return volume_group


# Helper function to get or insert a physical volume
def insert_or_get_physical_volume(session: Session, pv_uuid, pv_name, host_id: int):
    physical_volume = get_or_create_volume(session, host_id, PhysicalVolume,
                                           PhysicalVolumeInfo, "pv_name", "pv_uuid", pv_name, pv_uuid)
    return physical_volume


def insert_or_get_priority(session: Session, value: int) -> Priority:
    priority: Priority = session.query(Priority)\
        .filter_by(value=value)\
        .first()
    if priority is None:
        priority = Priority(
            value=value
        )
        session.add(priority)
    return priority


# Helper function to get or insert a logical volume
def insert_or_get_logical_volume(session: Session, lv_uuid, lv_name, host_id: int, priority: Priority):
    logical_volume = get_or_create_volume(session, host_id, LogicalVolume,
                                          LogicalVolumeInfo, "lv_name", "lv_uuid", lv_name, lv_uuid)
    logical_volume.priority = priority
    session.add(logical_volume)
    return logical_volume


# Helper function to get or insert filesystem type
def insert_or_get_file_system(session: Session, file_system_type: str) -> FileSystem:
    file_system: FileSystem = session.query(FileSystem).filter_by(
        file_system_type=file_system_type).first()
    if file_system is None:
        file_system = FileSystem(file_system_type=file_system_type)
        session.add(file_system)
    return file_system


# Helper function to get or insert filesystem type
def insert_or_get_mount_point(session: Session, path: str) -> MountPoint:
    mount_point: MountPoint = session.query(MountPoint).filter_by(
        path=path).first()
    if mount_point is None:
        mount_point = MountPoint(path=path)
        session.add(mount_point)
    return mount_point


# Function to insert logical volume adjustments into the database
def insert_to_logical_volume_adjustment(session: Session, db_logger: Logger, hostname: str, volume_group_adjustments: list[dict]) -> list[int]:
    with session.begin_nested():
        try:
            for index, adjustment in enumerate(volume_group_adjustments):
                lv_uuid = adjustment["lv_uuid"]
                # Query the logical volume based on UUID and hostname
                logical_volume: LogicalVolume = session.query(LogicalVolume)\
                    .filter_by(lv_uuid=lv_uuid)\
                    .join(Segment)\
                    .join(Host)\
                    .filter_by(hostname=hostname)\
                    .first()
                size = adjustment["size"]
                # Create a new adjustment object
                new_adjustment = Adjustment(
                    size=size,
                    extend=size >= 0,
                    logical_volume_id_fk=logical_volume.id
                )
                session.add(new_adjustment)
            session.commit()
        except IntegrityError as e:
            db_logger.get_logger().error(e)
            session.rollback()
            return []
        except SQLAlchemyError as e:
            db_logger.get_logger().error(e)
            session.rollback()
            return []
    return [adjustment.id for adjustment in session.query(Adjustment).order_by(Adjustment.created_at.desc()).limit(len(volume_group_adjustments)).all()]


# create a dataframe from the pvs information
# and tranform data to the right format
def transform_pvs_data_to_dataframe(pvs_informations: dict):
    df = pd.DataFrame.from_dict(pvs_informations)

    df["pv_size"] = df["pv_size"].apply(
        # remove any trailing charachters "m", "k"...
        lambda x: parse_size_number(x)
    )
    df["pv_free"] = df["pv_free"].apply(
        # remove any trailing charachters "m", "k"...
        lambda x: parse_size_number(x)
    )
    df["pv_used"] = df["pv_used"].apply(
        # remove any trailing charachters "m", "k"...
        lambda x: parse_size_number(x)
    )

    return df


# create a dataframe from the vgs information
# and tranform data to the right format
def transform_vgs_data_to_dataframe(vgs_informations: dict):
    df = pd.DataFrame.from_dict(vgs_informations)

    df["vg_size"] = df["vg_size"].apply(
        # remove any trailing charachters "m", "k"...
        lambda x: parse_size_number(x)
    )
    df["vg_free"] = df["vg_free"].apply(
        # remove any trailing charachters "m", "k"...
        lambda x: parse_size_number(x)
    )
    return df


def tranform_lvs_data_to_dataframe(lvs_informations: dict):
    # Create a DataFrame from the extracted information
    df = pd.DataFrame.from_dict(lvs_informations)

    df["lv_size"] = df["lv_size"].apply(
        # remove any trailing charachters "m", "k"...
        lambda x: parse_size_number(x)
    )
    df["seg_size"] = df["seg_size"].apply(
        # remove any trailing charachters "m", "k"...
        lambda x: parse_size_number(x)
    )
    df = parse_segment_ranges(df)
    return df


def tranform_lsblk_data_to_dataframe(lsblk_informations: dict):
    # Create a DataFrame from the extracted filesystem information
    df = pd.DataFrame.from_dict(lsblk_informations)
    # Filter out rows where 'mountpoint' is not null, drop NaN values
    df = df.dropna(subset=['mountpoint'])

    df['fssize'] = df['fssize'].apply(
        # convert bytes to MiB
        lambda x: convert_bytes_to_mib(
            # remove any trailing charachters "m", "k"...
            parse_size_number(x)
        )
    )
    df['fsused'] = df['fsused'].apply(
        # convert bytes to MiB
        lambda x: convert_bytes_to_mib(
            # remove any trailing charachters "m", "k"...
            parse_size_number(x)
        )
    )
    df['fsavail'] = df['fsavail'].apply(
        # convert bytes to MiB
        lambda x: convert_bytes_to_mib(
            # remove any trailing charachters "m", "k"...
            parse_size_number(x)
        )
    )
    return df


# Transform the lv_path and name columns to match the name of the lsblk name column
def transform_lv_path(path):
    return path.replace('/dev/', '').replace('/', '-')


# Transform the lvs and lsblk DataFrames into a one merged DataFrame
def transform_lvs_lsblk_data_to_dataframe(lvs: pd.DataFrame, lsblk: pd.DataFrame):
    # Transform the lv_path and name columns to match the name of the lsblk name column
    lvs['lv_path'] = lvs['lv_path'].apply(transform_lv_path)
    # Merge logical volume and filesystem DataFrames on 'lv_uuid'
    merged_df = pd.merge(
        lvs,
        lsblk,
        how="right",
        left_on='lv_path',
        right_on='name'
    )
    return merged_df


# Parse the host information into dataframes
def transform_lvm_data_to_dataframe(host_informations: dict, priorities: map) -> tuple[str, pd.DataFrame]:
    # Extract the hostname from the host information
    hostname = host_informations['host']
    # Extract the LVM information from the host information
    lvm_informations = host_informations["lvm_informations"]
    # Create a DataFrame from the extracted information
    pvs = transform_pvs_data_to_dataframe(lvm_informations['PVs'])
    vgs = transform_vgs_data_to_dataframe(lvm_informations['VGs'])
    lvs_fs = transform_lvs_lsblk_data_to_dataframe(
        tranform_lvs_data_to_dataframe(lvm_informations['LVs']),
        tranform_lsblk_data_to_dataframe(lvm_informations['lsblk'])
    )
    # Assign priority based on lv_name from the merged DataFrame
    lvs_fs['priority'] = lvs_fs['lv_name'].map(priorities)
    # Merge the DataFrames
    merge_pv_vg = pd.merge(
        vgs[['vg_uuid', 'vg_name', 'vg_size', 'vg_free']],
        pvs[['pv_uuid', 'pv_name', 'pv_size', 'pv_free', 'pv_used', 'vg_uuid']],
        how="inner",
        on="vg_uuid"
    )
    # Merge the DataFrames
    merge_pv_vg_lv_fs = pd.merge(
        merge_pv_vg,
        lvs_fs[['lv_uuid', 'lv_name', 'vg_uuid', 'lv_size', 'priority', 'seg_size', 'segment_range_start',
                'segment_range_end', 'lv_path', 'name', 'fstype', 'fssize', 'fsused', 'fsavail', 'mountpoint']],
        how="inner",
        on="vg_uuid"
    )

    # Drop duplicate columns
    merge_pv_vg_lv_fs = merge_pv_vg_lv_fs.loc[:,
                                              ~merge_pv_vg_lv_fs.columns.duplicated()]

    return hostname, merge_pv_vg_lv_fs


# Function to insert or update a segment in the database
def insert_or_update_segment(session: Session, logical_volume: LogicalVolume, physical_volume: PhysicalVolume, volume_group: VolumeGroup, host: Host, row):
    # Check if the segment already exists in the database
    segment: Segment = session.query(Segment).filter_by(
        physical_volume=physical_volume,
        logical_volume=logical_volume,
        volume_group=volume_group,
        host=host
    ).first()

    if segment is None:
        # Create a new segment if it doesn't exist
        segment: Segment = Segment(
            logical_volume=logical_volume,
            physical_volume=physical_volume,
            volume_group=volume_group,
            host=host,
        )
        session.add(segment)

    # Create new segment stats
    new_segment_stats = SegmentStats(
        segment_range_start=row['segment_range_start'],
        segment_range_end=row['segment_range_end'],
        segment_size=row['seg_size'],
    )
    new_segment_stats.segment = segment
    session.add(new_segment_stats)
    return segment


# Function to insert a logical volume and its stats in the database
def insert_logical_volume_and_stats(session: Session, lv_data: pd.DataFrame, host_id: int) -> LogicalVolume:
    # Insert or get the logical volume from the database
    logical_volume: LogicalVolume = insert_or_get_logical_volume(
        session=session, lv_uuid=lv_data["lv_uuid"], lv_name=lv_data["lv_name"], host_id=host_id,
        priority=insert_or_get_priority(
            session=session, value=lv_data['priority']
        )
    )

    # Create logical volume stats object
    logical_volume_stats: LogicalVolumeStats = LogicalVolumeStats(
        file_system_size=lv_data['fssize'],
        file_system_used_size=lv_data['fsused'],
        file_system_available_size=lv_data['fsavail'],
        file_system=insert_or_get_file_system(
            session=session, file_system_type=lv_data['fstype']
        ),
        mount_point=insert_or_get_mount_point(
            session=session, path=lv_data['mountpoint']
        )
    )

    # Add logical volume and logical volume stats to the session
    session.add(logical_volume)
    # Associate logical volume stats with the logical volume
    logical_volume_stats.logical_volume = logical_volume
    session.add(logical_volume_stats)

    return logical_volume


# Function to insert a physical volume and its stats in the database
def insert_physical_volume_and_stats(session: Session, pv_data: pd.DataFrame, host_id: int) -> PhysicalVolume:
    # Insert or get the physical volume from the database
    physical_volume = insert_or_get_physical_volume(
        session=session,
        pv_uuid=pv_data["pv_uuid"],
        pv_name=pv_data["pv_name"],
        host_id=host_id
    )
    session.add(physical_volume)

    # Create physical volume stats object
    physical_volume_stats = PhysicalVolumeStats(
        pv_size=pv_data["pv_size"],
        pv_free=pv_data["pv_free"],
        pv_used=pv_data["pv_used"]
    )

    # Associate physical volume stats with the physical volume
    physical_volume_stats.physical_volume = physical_volume
    session.add(physical_volume_stats)

    return physical_volume


# Function to insert a volume group and its stats in the database
def insert_volume_group_and_stats(session: Session, vg_data: pd.DataFrame, host_id: int) -> VolumeGroup:
    # insert or get the volume group from the database
    volume_group: VolumeGroup = insert_or_get_volume_group(
        session=session, vg_uuid=vg_data["vg_uuid"], vg_name=vg_data["vg_name"], host_id=host_id
    )
    # Add volume group to the session
    session.add(volume_group)
    # Create volume group stats object
    volume_group_stats: VolumeGroupStats = VolumeGroupStats(
        vg_size=vg_data["vg_size"],
        vg_free=vg_data["vg_free"],
    )
    # Associate volume group stats with the volume group
    volume_group_stats.volume_group = volume_group
    # Add volume group stats to the session
    session.add(volume_group_stats)
    return volume_group


# Function to insert LVM data into the database
def insert_lvm_data_to_database(session: Session, hostname: str, lvm_dataframe: pd.DataFrame):
    # Create a nested session to handle any errors
    with session.begin_nested():
        try:
            # Insert or get the host from the database
            host: Host = insert_or_get_host_entity(
                session=session, hostname=hostname
            )
            # Iterate over the rows in the DataFrame
            for index, row in lvm_dataframe.iterrows():
                # Insert the logical volume and its stats
                logical_volume: LogicalVolume = insert_logical_volume_and_stats(
                    session=session,
                    lv_data=row[['lv_uuid', 'lv_name', 'priority',
                                 'fssize', 'fsused', 'fsavail', 'fstype', 'mountpoint']],
                    host_id=host.id
                )
                # Insert the physical volume and its stats
                physical_volume: PhysicalVolume = insert_physical_volume_and_stats(
                    session=session,
                    pv_data=row[['pv_uuid', 'pv_name',
                                 'pv_size', 'pv_used', 'pv_free']],
                    host_id=host.id
                )
                # Insert the volume group and its stats
                volume_group: VolumeGroup = insert_volume_group_and_stats(
                    session=session,
                    vg_data=row[['vg_uuid', 'vg_name',
                                 'vg_size', 'vg_free', 'pv_used']],
                    host_id=host.id
                )
                # Insert or update the segment
                insert_or_update_segment(
                    session, logical_volume, physical_volume, volume_group, host, row
                )
            # Commit the session
            session.commit()
        except IntegrityError as e:
            # Handle any database integrity errors
            session.rollback()
            raise ValueError(f"Integrity error: {e}")
        except Exception as e:
            # Handle any other unexpected errors
            session.rollback()
            raise ValueError(f"Error: {e}")


def get_logical_volumes_list_per_volume_group_per_host(session: Session, hostname: str, vg_name: str, logical_volumes_names: list[str], lv_stats_limit: int):
    with session.begin_nested():
        rows = session.query(LogicalVolume, VolumeGroup, VolumeGroupInfo, LogicalVolumeInfo)\
            .join(LogicalVolumeInfo)\
            .join(Segment)\
            .join(Host)\
            .filter_by(hostname=hostname)\
            .join(VolumeGroup)\
            .join(VolumeGroupInfo)\
            .filter_by(vg_name=vg_name)\
            .order_by(VolumeGroupInfo.vg_name.asc())\
            .all()
        if len(rows) == 0:
            raise ValueError(
                f"No logical volumes found for volume group {vg_name} on host {hostname}")
        else:
            volume_groups_informations = {
                'hostname': hostname,
                'vg_name': rows[0].VolumeGroupInfo.vg_name,
                'vg_uuid': rows[0].VolumeGroup.vg_uuid,
                'vg_free': rows[0].VolumeGroup.stats[-1].vg_free,
            }
            logical_volumes_informations_list = []
            for row in rows:
                # get latest (depends on the number of the lookbacks) stats of one logical volumes
                if lv_stats_limit == 0:
                    lv_stat = session.query(LogicalVolumeStats).filter_by(
                        logical_volume_id_fk=row.LogicalVolume.id)\
                        .order_by(LogicalVolumeStats.created_at.desc())\
                        .limit(1)\
                        .all()
                else:
                    lv_stat = session.query(LogicalVolumeStats)\
                        .filter_by(logical_volume_id_fk=row.LogicalVolume.id)\
                        .order_by(LogicalVolumeStats.created_at.desc())\
                        .limit(lv_stats_limit)\
                        .all()

                if len(lv_stat) > 0:
                    # array of latest logical_volume fs usage
                    lv_fs_usage = [
                        stat.file_system_used_size for stat in lv_stat]
                    # # append the mean value | array has to be of 7 entries for the scaling
                    # # it is gonna be ignored during prediction
                    # lv_fs_usage.append(sum(lv_fs_usage) / len(lv_fs_usage))

                    # to reshaped numpy array
                    lv_fs_usage_allocation_reclaim_size = np.array(
                        # flip the order of the usage serie
                        # newest usage volume is at the end of the list (array)
                        np.flip(lv_fs_usage)
                    ).flatten()

                    # create a dictionary for each logical volume
                    lv_data = {
                        'lv_uuid': row.LogicalVolume.lv_uuid,
                        'lv_name': row.LogicalVolume.info.lv_name,
                        'priority': row.LogicalVolume.priority.value,
                        'file_system_size': lv_stat[0].file_system_size,
                        'file_system_type': lv_stat[0].file_system.file_system_type,
                        # Convert numpy array to list
                        'usage_serie': lv_fs_usage_allocation_reclaim_size.tolist(),
                    }
                    # add the dictionary to the list
                    logical_volumes_informations_list.append(lv_data)
            # add the list of logical volumes to the dictionary
            volume_groups_informations['logical_volumes'] = logical_volumes_informations_list
            return volume_groups_informations
