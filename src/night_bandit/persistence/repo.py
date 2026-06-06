"""Session / message / summary repository over the async pool.

UUIDs are generated app-side. All string content is sanitized before it
hits Postgres (NUL bytes / lone surrogates / noncharacters would abort the
insert). Sequence numbers are assigned atomically per session.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.types.json import Jsonb

from .db import get_pool, sanitize_for_storage


@dataclass
class Session:
    id: str
    user_id: str
    title: str | None
    status: str
    client_metadata: dict[str, Any] | None
    total_token_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class Message:
    id: str
    session_id: str
    turn_id: str
    role: str
    content: Any  # str or list[parts]
    text: str
    seq: int
    metadata: dict[str, Any] | None
    created_at: datetime


@dataclass
class Summary:
    id: str
    session_id: str
    start_seq: int
    end_seq: int
    summary_text: str
    token_count: int
    summary_model: str
    created_at: datetime


def _row_to_session(r: dict) -> Session:
    return Session(
        id=str(r["id"]),
        user_id=r["user_id"],
        title=r["title"],
        status=r["status"],
        client_metadata=r["client_metadata"],
        total_token_count=int(r["total_token_count"]),
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def _row_to_message(r: dict) -> Message:
    return Message(
        id=str(r["id"]),
        session_id=str(r["session_id"]),
        turn_id=str(r["turn_id"]),
        role=r["role"],
        content=r["content"],
        text=r["text"],
        seq=int(r["seq"]),
        metadata=r["metadata"],
        created_at=r["created_at"],
    )


def _row_to_summary(r: dict) -> Summary:
    return Summary(
        id=str(r["id"]),
        session_id=str(r["session_id"]),
        start_seq=int(r["start_seq"]),
        end_seq=int(r["end_seq"]),
        summary_text=r["summary_text"],
        token_count=int(r["token_count"]),
        summary_model=r["summary_model"],
        created_at=r["created_at"],
    )


class Repo:
    """Stateless data-access façade over the shared pool."""

    # --- sessions ---

    async def create_session(
        self,
        user_id: str,
        *,
        title: str | None = None,
        client_metadata: dict[str, Any] | None = None,
    ) -> Session:
        sid = str(uuid.uuid4())
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "INSERT INTO sessions (id, user_id, title, client_metadata) "
                "VALUES (%s, %s, %s, %s) RETURNING *",
                (sid, user_id, sanitize_for_storage(title),
                 Jsonb(sanitize_for_storage(client_metadata)) if client_metadata else None),
            )
            return _row_to_session(await cur.fetchone())

    async def get_session(self, session_id: str, user_id: str) -> Session | None:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM sessions WHERE id = %s AND user_id = %s",
                (session_id, user_id),
            )
            row = await cur.fetchone()
            return _row_to_session(row) if row else None

    async def list_sessions(self, user_id: str, *, limit: int = 50) -> list[Session]:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM sessions WHERE user_id = %s AND status = 'active' "
                "ORDER BY updated_at DESC LIMIT %s",
                (user_id, limit),
            )
            return [_row_to_session(r) for r in await cur.fetchall()]

    async def archive_session(self, session_id: str, user_id: str) -> bool:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "UPDATE sessions SET status = 'archived', updated_at = now() "
                "WHERE id = %s AND user_id = %s AND status = 'active'",
                (session_id, user_id),
            )
            return cur.rowcount > 0

    async def set_title_if_unset(self, session_id: str, title: str) -> None:
        async with get_pool().connection() as conn:
            await conn.execute(
                "UPDATE sessions SET title = %s, updated_at = now() "
                "WHERE id = %s AND title IS NULL",
                (sanitize_for_storage(title), session_id),
            )

    async def touch_session(self, session_id: str, *, add_tokens: int = 0) -> None:
        async with get_pool().connection() as conn:
            await conn.execute(
                "UPDATE sessions SET updated_at = now(), "
                "total_token_count = total_token_count + %s WHERE id = %s",
                (add_tokens, session_id),
            )

    # --- messages ---

    async def append_message(
        self,
        session_id: str,
        turn_id: str,
        role: str,
        content: Any,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Append a message, assigning the next per-session seq atomically."""
        mid = str(uuid.uuid4())
        safe_content = sanitize_for_storage(content)
        safe_text = sanitize_for_storage(text) or ""
        safe_meta = sanitize_for_storage(metadata) if metadata else None
        async with get_pool().connection() as conn:
            async with conn.transaction():
                # Lock the session row to serialize seq assignment.
                await conn.execute(
                    "SELECT 1 FROM sessions WHERE id = %s FOR UPDATE", (session_id,)
                )
                cur = await conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages "
                    "WHERE session_id = %s",
                    (session_id,),
                )
                seq = int((await cur.fetchone())["next"])
                cur = await conn.execute(
                    "INSERT INTO messages "
                    "(id, session_id, turn_id, role, content, text, seq, metadata) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                    (
                        mid, session_id, turn_id, role,
                        Jsonb(safe_content), safe_text, seq,
                        Jsonb(safe_meta) if safe_meta is not None else None,
                    ),
                )
                return _row_to_message(await cur.fetchone())

    async def get_messages(self, session_id: str) -> list[Message]:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM messages WHERE session_id = %s ORDER BY seq ASC",
                (session_id,),
            )
            return [_row_to_message(r) for r in await cur.fetchall()]

    async def get_messages_range(
        self, session_id: str, start_seq: int, end_seq: int
    ) -> list[Message]:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM messages WHERE session_id = %s "
                "AND seq >= %s AND seq <= %s ORDER BY seq ASC",
                (session_id, start_seq, end_seq),
            )
            return [_row_to_message(r) for r in await cur.fetchall()]

    # --- summaries ---

    async def get_summaries(self, session_id: str) -> list[Summary]:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM session_summaries WHERE session_id = %s "
                "ORDER BY start_seq ASC",
                (session_id,),
            )
            return [_row_to_summary(r) for r in await cur.fetchall()]

    async def add_summary(
        self,
        session_id: str,
        start_seq: int,
        end_seq: int,
        summary_text: str,
        token_count: int,
        summary_model: str,
    ) -> Summary:
        sid = str(uuid.uuid4())
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "INSERT INTO session_summaries "
                "(id, session_id, start_seq, end_seq, summary_text, token_count, summary_model) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (sid, session_id, start_seq, end_seq,
                 sanitize_for_storage(summary_text), token_count, summary_model),
            )
            return _row_to_summary(await cur.fetchone())
