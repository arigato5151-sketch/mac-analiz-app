"""Database configuration and session management."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models_db import Base


def create_engine_from_settings(echo: bool = False):
    """Create an engine without requiring unrelated API credentials."""
    database_url = os.getenv("DATABASE_URL", "sqlite:///./local.db").strip()
    if not database_url:
        raise ValueError("DATABASE_URL cannot be empty")

    return create_engine(database_url, echo=echo, future=True)


# Create engine and session factory
engine = create_engine_from_settings()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI/Starlette to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def drop_all_tables() -> None:
    """Drop all tables (for testing)."""
    Base.metadata.drop_all(bind=engine)
