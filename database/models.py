from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, ForeignKey, DateTime, func, Enum
import enum
from typing import List


class Base(DeclarativeBase):
    pass


class VolumeGroupStats(Base):
    __tablename__ = "volume_group_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    vg_size: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    volume_group_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "volume_group.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    volume_group: Mapped[List["VolumeGroup"]
                         ] = relationship("VolumeGroup", back_populates="stats")


class VolumeGroup(Base):
    __tablename__ = "volume_group"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # info
    vg_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False)
    vg_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Relationships
    stats: Mapped["VolumeGroupStats"] = relationship(
        "VolumeGroupStats", back_populates="volume_group")
    physical_volumes: Mapped[List["PhysicalVolume"]
                             ] = relationship("PhysicalVolume", back_populates="volume_group")


class PhysicalVolumeStats(Base):
    __tablename__ = "physical_volume_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    pv_size: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    physical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "physical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    physical_volumes: Mapped[List["PhysicalVolume"]
                             ] = relationship("PhysicalVolume", back_populates="stats")


class PhysicalVolume(Base):
    __tablename__ = "physical_volume"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # info
    pv_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False)
    pv_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    # Foreign keys
    group_volume_id_fk: Mapped[int] = mapped_column(
        ForeignKey("volume_group.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)

    # Relationships
    stats: Mapped["PhysicalVolumeStats"] = relationship(
        "PhysicalVolumeStats", back_populates="physical_volumes")
    volume_group: Mapped["VolumeGroup"] = relationship(
        "VolumeGroup", back_populates="physical_volumes")
    segments: Mapped[List["Segment"]
                     ] = relationship(back_populates="physical_volume")


class SegmentStats(Base):
    __tablename__ = "segment_stats"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    segment_range_start: Mapped[int] = mapped_column(nullable=False)
    segment_range_end: Mapped[int] = mapped_column(nullable=False)
    segment_size: Mapped[float] = mapped_column(nullable=False)
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


class LogicalVolumeStats(Base):
    __tablename__ = "logical_volume_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # stats
    file_system_type: Mapped[str] = mapped_column(String(10), nullable=False)
    file_system_size: Mapped[float] = mapped_column(nullable=False)
    file_system_used_size: Mapped[float] = mapped_column(nullable=False)
    file_system_available_size: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    # Relationships
    logical_volumes: Mapped[List["LogicalVolume"]
                            ] = relationship("LogicalVolume", back_populates="stats")


class LogicalVolume(Base):
    __tablename__ = "logical_volume"
    # info
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lv_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False)
    lv_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    # Relationships
    stats: Mapped["LogicalVolumeStats"] = relationship(
        "LogicalVolumeStats", back_populates="logical_volumes")
    lv_changes: Mapped[List["LvChange"]
                       ] = relationship("LvChange", back_populates="logical_volume")
    segments: Mapped[List["Segment"]
                     ] = relationship("Segment", back_populates="logical_volume")


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

    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)

    # Relationships
    logical_volume: Mapped["LogicalVolume"] = relationship("LogicalVolume",
                                                           back_populates="lv_changes")
