from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, ForeignKey, DateTime, func, Enum
import enum
from typing import List
from sqlalchemy.orm import Session


class Base(DeclarativeBase):
    pass


class VolumeGroupStats(Base):
    __tablename__ = "volume_group_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    vg_size: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    volume_group_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "volume_group.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    volume_group: Mapped[List["VolumeGroup"]
                         ] = relationship("VolumeGroup", back_populates="stats")


class VolumeGroupInfo(Base):
    __tablename__ = "volume_group_info"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    vg_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Relationships
    volume_group: Mapped[List["VolumeGroup"]
                         ] = relationship("VolumeGroup", back_populates="info")

    # classMethods
    def get_or_create(cls, session: Session, vg_name: str) -> "VolumeGroupInfo":
        # Try to find an existing row based on the vg_uuid
        existing_row = session.query(cls).filter_by(vg_name=vg_name).first()

        if existing_row:
            # If the row exists, return it
            return existing_row
        else:
            # If the row doesn't exist, create a new one and return it
            new_row = cls(vg_name=vg_name)
            session.add(new_row)
            session.commit()  # Commit the new row to the database
            return new_row


class VolumeGroup(Base):
    __tablename__ = "volume_group"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # info
    vg_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    volume_group_info_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "volume_group_info.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    info: Mapped["VolumeGroupInfo"] = relationship(
        "VolumeGroupInfo", back_populates="volume_group")
    stats: Mapped["VolumeGroupStats"] = relationship(
        "VolumeGroupStats", back_populates="volume_group")
    physical_volumes: Mapped[List["PhysicalVolume"]
                             ] = relationship("PhysicalVolume", back_populates="volume_group")

    # classMethods
    def get_or_create(cls, session: Session, vg_uuid: str) -> "VolumeGroup":
        # Try to find an existing row based on the vg_uuid
        existing_row = session.query(cls).filter_by(vg_uuid=vg_uuid).first()

        if existing_row:
            # If the row exists, return it
            return existing_row
        else:
            # If the row doesn't exist, create a new one and return it
            new_row = cls(vg_uuid=vg_uuid)
            session.add(new_row)
            session.commit()  # Commit the new row to the database
            return new_row


class PhysicalVolumeStats(Base):
    __tablename__ = "physical_volume_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    pv_size: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    physical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "physical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    physical_volume: Mapped[List["PhysicalVolume"]
                            ] = relationship("PhysicalVolume", back_populates="stats")


class PhysicalVolumeInfo(Base):
    __tablename__ = "physical_volume_info"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pv_name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Relationships
    physical_volume: Mapped[List["PhysicalVolume"]
                            ] = relationship("PhysicalVolume", back_populates="info")

    # classMethods
    def get_or_create(cls, session: Session, pv_name: str) -> "PhysicalVolumeInfo":
        # Try to find an existing row based on the vg_uuid
        existing_row = session.query(cls).filter_by(pv_name=pv_name).first()

        if existing_row:
            # If the row exists, return it
            return existing_row
        else:
            # If the row doesn't exist, create a new one and return it
            new_row = cls(pv_name=pv_name)
            session.add(new_row)
            session.commit()  # Commit the new row to the database
            return new_row


class PhysicalVolume(Base):
    __tablename__ = "physical_volume"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # info

    pv_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    volume_group_id_fk: Mapped[int] = mapped_column(
        ForeignKey("volume_group.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Foreign keys
    physical_volume_info_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "physical_volume_info.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, unique=True)
    # Relationships
    info: Mapped["PhysicalVolumeInfo"] = relationship(
        "PhysicalVolumeInfo", back_populates="physical_volume")
    stats: Mapped["PhysicalVolumeStats"] = relationship(
        "PhysicalVolumeStats", back_populates="physical_volume")
    volume_group: Mapped["VolumeGroup"] = relationship(
        "VolumeGroup", back_populates="physical_volumes")
    segments: Mapped[List["Segment"]
                     ] = relationship(back_populates="physical_volume")

    # classMethods
    def get_or_create(cls, session: Session, pv_uuid: str) -> "PhysicalVolume":
        # Try to find an existing row based on the vg_uuid
        existing_row = session.query(cls).filter_by(pv_uuid=pv_uuid).first()

        if existing_row:
            # If the row exists, return it
            return existing_row
        else:
            # If the row doesn't exist, create a new one and return it
            new_row = cls(pv_uuid=pv_uuid)
            session.add(new_row)
            session.commit()  # Commit the new row to the database
            return new_row


class SegmentStats(Base):
    __tablename__ = "segment_stats"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    segment_range_start: Mapped[int] = mapped_column(nullable=False)
    segment_range_end: Mapped[int] = mapped_column(nullable=False)
    segment_size: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    segment_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "segment.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    segments: Mapped[List["Segment"]
                     ] = relationship("Segment", back_populates="stats")


class Segment(Base):
    __tablename__ = "segment"
    # info
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    physical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "physical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    stats: Mapped["SegmentStats"] = relationship(
        "SegmentStats", back_populates="segments")
    logical_volume: Mapped["LogicalVolume"] = relationship(
        back_populates="segments")
    physical_volume: Mapped["PhysicalVolume"] = relationship(
        back_populates="segments")

    # classMethods
    def get_or_create(cls, session: Session, logical_volume_id_fk: int, physical_volume_id_fk: int) -> "Segment":
        # Try to find an existing row based on the vg_uuid
        existing_row = session.query(cls).filter_by(
            logical_volume_id_fk=logical_volume_id_fk, physical_volume_id_fk=physical_volume_id_fk).first()

        if existing_row:
            # If the row exists, return it
            return existing_row
        else:
            # If the row doesn't exist, create a new one and return it
            new_row = cls(logical_volume_id_fk=logical_volume_id_fk,
                          physical_volume_id_fk=physical_volume_id_fk)
            session.add(new_row)
            session.commit()  # Commit the new row to the database
            return new_row


class LogicalVolumeStats(Base):
    __tablename__ = "logical_volume_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    file_system_type: Mapped[str] = mapped_column(String(10), nullable=False)
    file_system_size: Mapped[int] = mapped_column(nullable=False)
    file_system_used_size: Mapped[int] = mapped_column(nullable=False)
    file_system_available_size: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    logical_volume: Mapped[List["LogicalVolume"]
                           ] = relationship("LogicalVolume", back_populates="stats")


class LogicalVolumeInfo(Base):
    __tablename__ = "logical_volume_info"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lv_name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Relationships
    logical_volume: Mapped[List["LogicalVolume"]
                           ] = relationship("LogicalVolume", back_populates="info")

    # classMethods
    def get_or_create(cls, session: Session, lv_name: str) -> "LogicalVolumeInfo":
        # Try to find an existing row based on the vg_uuid
        existing_row = session.query(cls).filter_by(lv_name=lv_name).first()

        if existing_row:
            # If the row exists, return it
            return existing_row
        else:
            # If the row doesn't exist, create a new one and return it
            new_row = cls(lv_name=lv_name)
            session.add(new_row)
            session.commit()  # Commit the new row to the database
            return new_row


class LogicalVolume(Base):
    __tablename__ = "logical_volume"
    # info
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lv_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    logical_volume_info_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume_info.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, unique=True)
    # Relationships
    info: Mapped["LogicalVolumeInfo"] = relationship(
        "LogicalVolumeInfo", back_populates="logical_volume")
    stats: Mapped["LogicalVolumeStats"] = relationship(
        "LogicalVolumeStats", back_populates="logical_volume")
    lv_changes: Mapped[List["LvChange"]
                       ] = relationship("LvChange", back_populates="logical_volume")
    segments: Mapped[List["Segment"]
                     ] = relationship("Segment", back_populates="logical_volume")

    # classMethods
    def get_or_create(cls, session: Session, lv_uuid: str) -> "LogicalVolume":
        # Try to find an existing row based on the vg_uuid
        existing_row = session.query(cls).filter_by(lv_uuid=lv_uuid).first()

        if existing_row:
            # If the row exists, return it
            return existing_row
        else:
            # If the row doesn't exist, create a new one and return it
            new_row = cls(lv_uuid=lv_uuid)
            # session.add(new_row)
            # session.add(new_row)
            # session.commit()  # Commit the new row to the database
            return new_row


class LvChangeEnum(enum.Enum):
    pending = 0
    done = 1
    failed = 2


class LvChange(Base):
    __tablename__ = "lv_change"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    status: Mapped[str] = mapped_column(Enum(LvChangeEnum),
                                        default=LvChangeEnum.pending, nullable=False)
    extend: Mapped[bool] = mapped_column(nullable=False)
    size: Mapped[Float] = mapped_column(Float(), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)

    # Relationships
    logical_volume: Mapped["LogicalVolume"] = relationship("LogicalVolume",
                                                           back_populates="lv_changes")
