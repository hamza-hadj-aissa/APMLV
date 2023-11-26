import math
from numpy import double
import pandas as pd
from sqlalchemy.orm import sessionmaker, Session
from commands_utils import get_group_volumes, get_lv_lsblk, get_physical_volumes
from database.connect import connect_to_database
from database.utils import get_volume_entity, insert_volume_entity
from exceptions.InstanceNotFound import InstanceNotFound
from database.models import (
    LogicalVolumeStats,
    PhysicalVolumeStats,
    VolumeGroup,
    VolumeGroupInfo,
    PhysicalVolume,
    PhysicalVolumeInfo,
    LogicalVolume,
    LogicalVolumeInfo,
    Segment,
    SegmentStats,
    VolumeGroupStats
)


# Helper function to get or insert a volume group
def insert_or_get_volume_group(session: Session, vg_uuid, vg_name):
    found_vg = get_volume_entity(session, VolumeGroup, vg_uuid=vg_uuid)
    if found_vg is None or found_vg.info.vg_name != vg_name:
        found_vg = insert_volume_entity(
            session, VolumeGroup, VolumeGroupInfo, "vg_name", "vg_uuid", vg_name, vg_uuid)
        session.add(found_vg)
        session.commit()
    return found_vg


# Helper function to get or insert a physical volume
def insert_or_get_physical_volume(session: Session, pv_uuid, pv_name, vg_uuid):
    found_vg = get_volume_entity(session, VolumeGroup, vg_uuid=vg_uuid)
    if found_vg is None:
        raise InstanceNotFound("Volume group not found")

    found_pv = get_volume_entity(session, PhysicalVolume, pv_uuid=pv_uuid)
    if found_pv is None or found_pv.info.pv_name != pv_name:
        found_pv = insert_volume_entity(
            session, PhysicalVolume, PhysicalVolumeInfo, "pv_name", "pv_uuid", pv_name, pv_uuid)
        found_pv.volume_group_id_fk = found_vg.id
        session.add(found_pv)
        session.commit()
    return found_pv


# Helper function to get or insert a logical volume
def insert_or_get_logical_volume(session: Session, lv_uuid, lv_name):
    found_lv = get_volume_entity(session, LogicalVolume, lv_uuid=lv_uuid)
    if found_lv is None or found_lv.info.lv_name != lv_name:
        found_lv = insert_volume_entity(
            session, LogicalVolume, LogicalVolumeInfo, "lv_name", "lv_uuid", lv_name, lv_uuid)
        session.add(found_lv)
        session.commit()
    return found_lv


# Helper function to get or insert a segment
def insert_or_get_segment(session: Session, pv_name, lv_uuid):
    found_pv_info = get_volume_entity(
        session, PhysicalVolumeInfo, pv_name=pv_name)
    found_pv = get_volume_entity(
        session, PhysicalVolume, physical_volume_info_id_fk=found_pv_info.id)
    found_lv = get_volume_entity(session, LogicalVolume, lv_uuid=lv_uuid)

    if found_lv is None or found_pv is None:
        if found_lv is None:
            raise InstanceNotFound(f"Logical volume not found ({lv_uuid})")
        else:
            raise InstanceNotFound(f"Physical volume not found ({pv_name})")

    found_segment = session.query(Segment).filter_by(
        logical_volume_id_fk=found_lv.id,
        physical_volume_id_fk=found_pv.id
    ).first()

    if found_segment is None:
        found_segment = Segment(
            logical_volume_id_fk=found_lv.id,
            physical_volume_id_fk=found_pv.id
        )
        session.add(found_segment)
        session.commit()

    return found_segment


# Helper function to get or insert volume group stat
def insert_to_volume_group_stats(session: Session, vgs: pd.DataFrame):
    for index, new_vg in vgs.iterrows():
        # find and get volume group instance
        found_vg: VolumeGroup = insert_or_get_volume_group(
            session, new_vg["vg_uuid"], new_vg["vg_name"])
        new_vg_stat = VolumeGroupStats(
            vg_size=new_vg["vg_size"],
            # newely inserted vg id
            volume_group_id_fk=found_vg.id
        )
        # add volume group stat instance
        session.add(new_vg_stat)
        session.commit()


# Helper function to get or insert physical volume stat
def insert_to_physical_volume_stats(session: Session, pvs: pd.DataFrame):
    for _, new_pv in pvs.iterrows():
        # find and get physical volume instance
        found_pv = insert_or_get_physical_volume(
            session, new_pv["pv_uuid"], new_pv["pv_name"], new_pv["vg_uuid"])

        new_pv_stat = PhysicalVolumeStats(
            pv_size=new_pv["pv_size"],
            physical_volume_id_fk=found_pv.id
        )
        # add physical volume instance stat
        session.add(new_pv_stat)

        # insert new pv stats to DB
        session.commit()


# Helper function to get or insert logical volume stat
def insert_to_logical_volume_stats(session: Session, lvs: pd.DataFrame):
    for _, new_lv in lvs.iterrows():
        # find and get logical volume instance
        found_lv: LogicalVolume = insert_or_get_logical_volume(
            session, new_lv["lv_uuid"], new_lv["lv_name"])
        new_lv_stat = LogicalVolumeStats(
            file_system_type=new_lv["fstype"],
            file_system_size=math.ceil(double(new_lv["fssize"])),
            file_system_used_size=math.ceil(double(new_lv["fsused"])),
            file_system_available_size=math.ceil(
                double(new_lv["fsavail"])),
            logical_volume_id_fk=found_lv.id
        )
        # add logical volume instance stat
        session.add(new_lv_stat)
        session.commit()


# Helper function to get or insert segment stat
def insert_to_segment_stats(session: Session, segments: pd.DataFrame):
    for _, new_segment in segments.iterrows():
        found_segment = insert_or_get_segment(
            session, new_segment["pv_name"], new_segment["lv_uuid"])

        new_segment_stat = SegmentStats(
            segment_size=new_segment["seg_size"],
            segment_range_start=new_segment["segment_range_start"],
            segment_range_end=new_segment["segment_range_end"],
            segment_id_fk=found_segment.id
        )
        session.add(new_segment_stat)
        session.commit()


def start_extracting_data(session: Session):
    pvs = get_physical_volumes()
    vgs = get_group_volumes()
    lvs_fs = get_lv_lsblk()
    print(lvs_fs)
    insert_to_volume_group_stats(
        session, vgs[["vg_uuid", "vg_name", "vg_size"]].drop_duplicates())
    insert_to_physical_volume_stats(
        session, pvs[["pv_uuid", "pv_name", "pv_size", "vg_uuid"]].drop_duplicates())
    insert_to_logical_volume_stats(
        session, lvs_fs[["lv_uuid", "lv_name", "fstype", "fssize", "fsused", "fsavail"]])
    insert_to_segment_stats(
        session, lvs_fs[["seg_size", "segment_range_start", "segment_range_end", "pv_name", "lv_uuid"]])
    session.commit()
    session.close()


if __name__ == "__main__":
    # Connect to the database
    database_engine = connect_to_database()
    DBSession = sessionmaker(bind=database_engine)
    session = DBSession()

    start_extracting_data(session)
