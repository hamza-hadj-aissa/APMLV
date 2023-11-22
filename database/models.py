from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, ForeignKey, DateTime, func, Enum
from typing import List
import enum


class Base(DeclarativeBase):
    pass


class Group_volume(Base):
    __tablename__ = "group_volume"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    vg_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False)
    vg_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    vg_size: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    physical_volumes: Mapped[List["Physical_volume"]
                             ] = relationship("Physical_volume", back_populates="group_volume")


class Lv_change_enum(enum.Enum):
    pending = 0
    done = 1
    failed = 2


class Lv_change(Base):
    __tablename__ = "lv_change"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    status: Mapped[str] = mapped_column(Enum(Lv_change_enum),
                                        default=Lv_change_enum.pending, nullable=False)
    extend: Mapped[bool] = mapped_column(nullable=False)
    size: Mapped[Float] = mapped_column(Float(), nullable=False)

    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    logical_volume: Mapped["Logical_volume"] = relationship("Logical_volume",
                                                            back_populates="Lv_change")


class Logical_volume(Base):
    __tablename__ = "logical_volume"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lv_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False)
    lv_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    file_system_type: Mapped[str] = mapped_column(String(10), nullable=False)
    file_system_size: Mapped[float] = mapped_column(nullable=False)
    file_system_used_size: Mapped[float] = mapped_column(nullable=False)
    file_system_available_size: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    lv_changes: Mapped[List["Lv_change"]] = relationship("Lv_change",
                                                         back_populates="logical_volume")
    segments: Mapped[List["Segment"]
                     ] = relationship("Segment", back_populates="logical_volume")


class Segment(Base):
    __tablename__ = "segment"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    segment_size: Mapped[float] = mapped_column(nullable=False)

    # Foreign keys
    logical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "logical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    physical_volume_id_fk: Mapped[int] = mapped_column(ForeignKey(
        "physical_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())
    physical_volume: Mapped["Physical_volume"] = relationship(
        "Physical_volume", back_populates="segment")
    logical_volume: Mapped["Logical_volume"] = relationship(
        "Logical_volume", back_populates="segment")


class Physical_volume(Base):
    __tablename__ = "physical_volume"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pv_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False)
    pv_uuid: Mapped[str] = mapped_column(String(255), nullable=False)
    pv_size: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime,
                                                 nullable=False, default=func.now())

    # Foreign keys
    group_volume_id_fk: Mapped[int] = mapped_column(
        ForeignKey("group_volume.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    group_volume: Mapped["Group_volume"] = relationship(
        back_populates="Physical_volume")
    segments: Mapped[List["Segment"]
                     ] = relationship(back_populates="physical_volume")
