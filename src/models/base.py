"""
Declarative base for all SQLAlchemy ORM models.
All models import Base from here to share the same metadata object.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
