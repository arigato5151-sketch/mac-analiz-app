"""Database configuration and session management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from db.models_db import Base


def create_engine_from_settings(echo: bool = False):
    """Create SQLAlchemy engine from settings."""
    settings = get_settings()
    # Convert Supabase URL to SQLAlchemy-compatible PostgreSQL URL
    # Supabase URL format: https://project.supabase.co
    # We need to convert to postgresql://user:password@host:port/db
    # For Supabase, we need to use the pooler or direct connection
    supabase_url = settings.supabase_url
    
    # Extract project reference from URL
    # https://project.supabase.co -> project
    if supabase_url.startswith("https://"):
        supabase_url = supabase_url[8:]  # Remove https://
    if supabase_url.endswith("/"):
        supabase_url = supabase_url[:-1]
    
    # Supabase database connection format
    # postgresql://postgres.{project_ref}:{password}@aws-0-{region}.pooler.supabase.com:6543/postgres
    # For simplicity, we'll use a connection string format
    # In practice, you'd use the Supabase connection string from the dashboard
    # For now, we'll use the Supabase REST client for production
    # and this for local development with SQLite
    import os
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Default to SQLite for local development
        database_url = "sqlite:///./local.db"
    
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