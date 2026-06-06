"""Async Postgres pool + migration runner.

The pool sets ``search_path`` to the configured schema on every connection,
so all unqualified DDL/DML lands in the Night Bandit schema and we can
share a database whose role lacks CREATEDB.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ..config import get_settings

_pool: AsyncConnectionPool | None = None


# --- NUL-byte / control sanitation (Postgres TEXT + JSONB reject U+0000) ---
# Built from code points so this source file stays free of bare control /
# surrogate / noncharacter bytes (the very bytes that break editors + PG).
def _build_forbidden() -> "re.Pattern[str]":
    def e(cp: int) -> str:
        return "\\u%04x" % cp

    cls = (
        e(0x00) + "-" + e(0x08)  # NUL..backspace
        + e(0x0B) + e(0x0C)  # vtab, formfeed (skip 0x09 tab, 0x0A LF, 0x0D CR)
        + e(0x0E) + "-" + e(0x1F)  # rest of C0
        + e(0xD800) + "-" + e(0xDFFF)  # lone surrogates
        + e(0xFDD0) + "-" + e(0xFDEF)  # Arabic-block noncharacters
        + e(0xFFFE) + e(0xFFFF)  # BMP noncharacters
    )
    return re.compile("[" + cls + "]")


_FORBIDDEN = _build_forbidden()


def _sanitize_str(s: str) -> str:
    return _FORBIDDEN.sub("", s) if _FORBIDDEN.search(s) else s


def sanitize_for_storage(value: Any) -> Any:
    """Recursively strip Postgres-hostile characters from any value."""
    if isinstance(value, str):
        return _sanitize_str(value)
    if isinstance(value, list):
        return [sanitize_for_storage(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_for_storage(v) for k, v in value.items()}
    return value


async def _configure(conn: AsyncConnection) -> None:
    schema = get_settings().pg_schema
    # Identifier can't be parameterized; schema comes from trusted config.
    await conn.execute(f"SET search_path TO {schema}, public")


def get_pool() -> AsyncConnectionPool:
    """Process-wide pool singleton. Call within the app lifespan."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = AsyncConnectionPool(
            settings.pg_dsn,
            min_size=1,
            max_size=8,
            kwargs={"row_factory": dict_row, "autocommit": True},
            configure=_configure,
            open=False,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def run_migrations(migrations_dir: Path) -> list[str]:
    """Create the schema + apply pending .sql migrations in order.

    Returns the filenames applied this run.
    """
    settings = get_settings()
    pool = get_pool()
    applied: list[str] = []
    async with pool.connection() as conn:
        await conn.execute(
            f"CREATE SCHEMA IF NOT EXISTS {settings.pg_schema} "
            f"AUTHORIZATION CURRENT_USER"
        )
        await conn.execute(f"SET search_path TO {settings.pg_schema}, public")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  filename text PRIMARY KEY,"
            "  applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cur = await conn.execute("SELECT filename FROM schema_migrations")
        done = {r["filename"] for r in await cur.fetchall()}
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in done:
                continue
            sql = path.read_text(encoding="utf-8")
            await conn.execute(sql)  # type: ignore[arg-type]
            await conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
            )
            applied.append(path.name)
    return applied
