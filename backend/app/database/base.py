from sqlalchemy import Uuid
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Generic UUID keeps the same model definitions compatible with PostgreSQL and SQLite.
UUIDType = Uuid(as_uuid=True)
