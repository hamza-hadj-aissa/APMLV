from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, ForeignKey, DateTime, UniqueConstraint, func, Enum
import enum
from typing import List


class Base(DeclarativeBase):
    pass


class Host(Base):
    __tablename__ = "host"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    hostname: Mapped[str] = mapped_column(nullable=False, unique=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Relationships
    segments: Mapped[List["Segment"]
                     ] = relationship(back_populates="host")


class VolumeGroupStats(Base):
    __tablename__ = "volume_group_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    vg_size: Mapped[int] = mapped_column(nullable=False)
    vg_free: Mapped[int] = mapped_column(nullable=False)
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
    stats: Mapped[List["VolumeGroupStats"]] = relationship(
        "VolumeGroupStats", back_populates="volume_group")
    segments: Mapped[List["Segment"]
                     ] = relationship(back_populates="volume_group")


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


class PhysicalVolume(Base):
    __tablename__ = "physical_volume"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # info

    pv_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    physical_volume_info_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "physical_volume_info.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    info: Mapped["PhysicalVolumeInfo"] = relationship(
        "PhysicalVolumeInfo", back_populates="physical_volume")
    stats: Mapped[List["PhysicalVolumeStats"]] = relationship(
        "PhysicalVolumeStats", back_populates="physical_volume")
    segments: Mapped[List["Segment"]
                     ] = relationship(back_populates="physical_volume")


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
    segment: Mapped["Segment"
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
    volume_group_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "volume_group.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    host_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "host.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    stats: Mapped[List["SegmentStats"]] = relationship(
        "SegmentStats", back_populates="segment")
    logical_volume: Mapped["LogicalVolume"] = relationship(
        back_populates="segments")
    physical_volume: Mapped["PhysicalVolume"] = relationship(
        back_populates="segments")
    volume_group: Mapped["VolumeGroup"] = relationship(
        back_populates="segments")
    host: Mapped["Host"] = relationship(back_populates="segments")
    __table_args__ = (
        UniqueConstraint('logical_volume_id_fk', 'physical_volume_id_fk', 'volume_group_id_fk', 'host_id_fk',
                         name='unique_LV_PV_VG_SEGMENT_per_host'),
    )


class FileSystem(Base):
    __tablename__ = "file_system"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_system_type: Mapped[str] = mapped_column(
        String(10), nullable=False, unique=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    logical_volume_stats: Mapped[List["LogicalVolumeStats"]
                                 ] = relationship("LogicalVolumeStats", back_populates="file_system")


class LogicalVolumeStats(Base):
    __tablename__ = "logical_volume_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    file_system_size: Mapped[int] = mapped_column(nullable=False)
    file_system_used_size: Mapped[int] = mapped_column(nullable=False)
    file_system_available_size: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    file_system_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "file_system.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    file_system: Mapped[List["FileSystem"]] = relationship(
        "FileSystem", back_populates="logical_volume_stats")
    logical_volume: Mapped[List["LogicalVolume"]
                           ] = relationship("LogicalVolume", back_populates="stats")


class LogicalVolumeInfo(Base):
    __tablename__ = "logical_volume_info"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lv_name: Mapped[str] = mapped_column(
        String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Relationships
    logical_volume: Mapped[List["LogicalVolume"]
                           ] = relationship("LogicalVolume", back_populates="info")


class LogicalVolume(Base):
    __tablename__ = "logical_volume"
    # info
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lv_uuid: Mapped[str] = mapped_column(
        String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    logical_volume_info_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume_info.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    priority_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "priority.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    info: Mapped["LogicalVolumeInfo"] = relationship(
        "LogicalVolumeInfo", back_populates="logical_volume")
    stats: Mapped[List["LogicalVolumeStats"]] = relationship(
        "LogicalVolumeStats", back_populates="logical_volume")
    adjustmens: Mapped[List["Adjustment"]
                       ] = relationship("Adjustment", back_populates="logical_volume")
    segments: Mapped[List["Segment"]
                     ] = relationship("Segment", back_populates="logical_volume")
    priority: Mapped["Priority"] = relationship(
        "Priority", back_populates="logical_volume")


class Priority(Base):
    __tablename__ = "priority"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(nullable=True, unique=True)
    value: Mapped[int] = mapped_column(nullable=False, unique=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Relationships
    logical_volume: Mapped[List["LogicalVolume"]] = relationship("LogicalVolume",
                                                                 back_populates="priority")


class AdjustmentStateEnum(enum.Enum):
    pending = 0
    done = 1
    failed = 2


class Adjustment(Base):
    __tablename__ = "adjustment"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    status: Mapped[str] = mapped_column(Enum(AdjustmentStateEnum),
                                        default=AdjustmentStateEnum.pending, nullable=False)
    extend: Mapped[bool] = mapped_column(nullable=False)
    size: Mapped[Float] = mapped_column(Float(), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)

    # Relationships
    logical_volume: Mapped["LogicalVolume"] = relationship("LogicalVolume",
                                                           back_populates="adjustmens")
