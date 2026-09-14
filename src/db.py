"""
Database connection for the resy EDA project.

Credentials are read exclusively from environment variables (via a local
.env file, never committed or shared). This module never hardcodes or logs
credentials.

Usage:
    from src.db import get_engine
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM \"DailyRoutes\" LIMIT 10", engine)
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text

load_dotenv()  # loads .env if present; no-op otherwise

REQUIRED_VARS = [
    "RESY_DB_HOST",
    "RESY_DB_PORT",
    "RESY_DB_NAME",
    "RESY_DB_USER",
    "RESY_DB_PASSWORD",
]


def _build_connection_url() -> str:
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            "Missing required DB environment variables: "
            f"{', '.join(missing)}. Copy .env.example to .env and fill it in."
        )

    host = os.environ["RESY_DB_HOST"]
    port = os.environ["RESY_DB_PORT"]
    name = os.environ["RESY_DB_NAME"]
    user = os.environ["RESY_DB_USER"]
    password = os.environ["RESY_DB_PASSWORD"]

    # psycopg2 driver; password is passed straight to SQLAlchemy's URL
    # object rather than interpolated as a raw string, so special
    # characters in the password are handled safely.
    from sqlalchemy.engine import URL

    return URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=name,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Returns a cached SQLAlchemy engine. Connection is lazy (not opened
    until the first query runs)."""
    url = _build_connection_url()
    return create_engine(url, pool_pre_ping=True)


def test_connection() -> None:
    """Quick sanity check you can run standalone: `python -m src.db`"""
    engine = get_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version();")).scalar_one()
        table_count = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'public';"
            )
        ).scalar_one()
    print("Connected OK.")
    print(f"Postgres version: {version}")
    print(f"Tables in public schema: {table_count}")


if __name__ == "__main__":
    test_connection()
