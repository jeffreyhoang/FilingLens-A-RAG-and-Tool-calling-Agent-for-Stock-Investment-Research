"""The Supabase (Postgres) connection, from DATABASE_URL in .env.

The owner's login can read and write; the read-only login can only
read, which is all the app needs.
"""

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings


def connect() -> psycopg.Connection:
    """A new autocommit connection that understands pgvector's halfvec type."""
    conn = psycopg.connect(
        get_settings().database_url, autocommit=True, connect_timeout=10
    )
    conn.execute("SET search_path = public, extensions")  # where pgvector lives
    register_vector(conn)
    return conn
