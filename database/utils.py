from sqlalchemy.orm import Session
from database.models.models import LogicalVolume, LogicalVolumeInfo, PhysicalVolume, VolumeGroup, VolumeGroupInfo


class DuplicateRowError(Exception):
    pass


def insert_volume_entity(session: Session, model: LogicalVolume | VolumeGroup, info_model: LogicalVolumeInfo | VolumeGroupInfo, name_column, uuid_column, name, uuid):
    # Check if name is already taken
    name_taken = session.query(model).join(
        getattr(model, 'info')).filter_by(**{name_column: name}).first()

    if name_taken is not None:
        # Raise an error if name is already taken
        raise DuplicateRowError(f"Name already taken ({name})")
    else:
        # Check if uuid already exists in the database
        entity_exists = session.query(model).filter_by(
            **{uuid_column: uuid}).first()

        # Check if info already exists in the database
        entity_info = session.query(info_model).filter_by(
            **{name_column: name}).first()

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


def get_volume_entity(session: Session, model: LogicalVolume | VolumeGroup | PhysicalVolume | None, **kwargs) -> LogicalVolume | VolumeGroup | PhysicalVolume | None:
    return session.query(model).filter_by(**kwargs).first()
