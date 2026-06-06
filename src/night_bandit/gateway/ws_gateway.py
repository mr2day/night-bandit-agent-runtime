"""One WebSocket connection: auth, session CRUD, and turn streaming.

A turn runs as a background task so the connection keeps receiving frames
(e.g. turn.cancel) while the ensemble works. Every transparency event the
graph emits is forwarded live as a ``turn.event`` frame and also collected
and persisted on the assistant message so a reloaded session replays it.
All sends go through a lock (multiple tasks may send on one socket).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ..compaction import assemble_context, maybe_compact
from ..config import get_settings
from ..orchestration.graph import build_graph
from ..persistence.repo import Message, Repo, Session, Summary
from .auth import AuthError, verify_token


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _public_session(s: Session) -> dict[str, Any]:
    return {
        "id": s.id,
        "title": s.title,
        "status": s.status,
        "clientMetadata": s.client_metadata,
        "totalTokenCount": s.total_token_count,
        "createdAt": _iso(s.created_at),
        "updatedAt": _iso(s.updated_at),
    }


def _public_message(m: Message) -> dict[str, Any]:
    return {
        "id": m.id,
        "turnId": m.turn_id,
        "role": m.role,
        "content": m.content,
        "text": m.text,
        "seq": m.seq,
        "metadata": m.metadata,
        "createdAt": _iso(m.created_at),
    }


def _public_summary(s: Summary) -> dict[str, Any]:
    return {
        "id": s.id,
        "startSeq": s.start_seq,
        "endSeq": s.end_seq,
        "summaryText": s.summary_text,
        "tokenCount": s.token_count,
        "summaryModel": s.summary_model,
        "createdAt": _iso(s.created_at),
    }


class Connection:
    def __init__(self, ws: WebSocket, app_state: Any) -> None:
        self._ws = ws
        self._state = app_state
        self._repo: Repo = app_state.repo
        self._user_id: str | None = None
        self._send_lock = asyncio.Lock()
        self._turns: dict[str, asyncio.Task] = {}

    async def _send(self, frame: dict[str, Any]) -> None:
        async with self._send_lock:
            try:
                await self._ws.send_json(frame)
            except (RuntimeError, WebSocketDisconnect):
                pass

    async def run(self) -> None:
        await self._ws.accept()
        try:
            while True:
                frame = await self._ws.receive_json()
                await self._dispatch(frame)
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001 - never crash the socket loop
            await self._send({"type": "error", "code": "internal", "message": str(e)})
        finally:
            for task in list(self._turns.values()):
                task.cancel()

    async def _dispatch(self, frame: dict[str, Any]) -> None:
        ftype = frame.get("type")
        reply_id = frame.get("id")

        if ftype == "auth":
            try:
                self._user_id = verify_token(frame.get("token"))
            except AuthError as e:
                await self._send({"type": "error", "code": "auth_failed", "message": str(e)})
                await self._ws.close()
                return
            await self._send({"type": "ready", "userId": self._user_id})
            return

        if self._user_id is None:
            await self._send({"type": "error", "code": "unauthenticated", "message": "auth first"})
            return

        if ftype == "session.create":
            s = await self._repo.create_session(
                self._user_id,
                title=frame.get("title"),
                client_metadata=frame.get("clientMetadata"),
            )
            await self._send({"type": "session.created", "inReplyTo": reply_id, "session": _public_session(s)})
        elif ftype == "session.list":
            rows = await self._repo.list_sessions(self._user_id)
            await self._send({"type": "session.list", "inReplyTo": reply_id, "sessions": [_public_session(x) for x in rows]})
        elif ftype == "session.get":
            await self._handle_session_get(frame, reply_id)
        elif ftype == "session.archive":
            ok = await self._repo.archive_session(frame.get("sessionId", ""), self._user_id)
            await self._send({"type": "session.archived", "inReplyTo": reply_id, "sessionId": frame.get("sessionId"), "ok": ok})
        elif ftype == "messages.fetch_range":
            await self._handle_fetch_range(frame, reply_id)
        elif ftype == "turn.submit":
            await self._handle_turn_submit(frame)
        elif ftype == "turn.cancel":
            task = self._turns.get(frame.get("turnId", ""))
            if task:
                task.cancel()
        else:
            await self._send({"type": "error", "code": "bad_frame", "message": f"unknown type: {ftype}"})

    async def _handle_session_get(self, frame: dict[str, Any], reply_id: Any) -> None:
        session_id = frame.get("sessionId", "")
        s = await self._repo.get_session(session_id, self._user_id)
        if s is None:
            await self._send({"type": "error", "code": "not_found", "message": "session not found"})
            return
        messages = await self._repo.get_messages(session_id)
        summaries = await self._repo.get_summaries(session_id)
        await self._send(
            {
                "type": "session.detail",
                "inReplyTo": reply_id,
                "session": _public_session(s),
                "messages": [_public_message(m) for m in messages],
                "summaries": [_public_summary(x) for x in summaries],
            }
        )

    async def _handle_fetch_range(self, frame: dict[str, Any], reply_id: Any) -> None:
        session_id = frame.get("sessionId", "")
        s = await self._repo.get_session(session_id, self._user_id)
        if s is None:
            await self._send({"type": "error", "code": "not_found", "message": "session not found"})
            return
        msgs = await self._repo.get_messages_range(
            session_id, int(frame.get("startSeq", 0)), int(frame.get("endSeq", 0))
        )
        await self._send(
            {
                "type": "messages.range",
                "inReplyTo": reply_id,
                "sessionId": session_id,
                "messages": [_public_message(m) for m in msgs],
            }
        )

    async def _handle_turn_submit(self, frame: dict[str, Any]) -> None:
        session_id = frame.get("sessionId", "")
        message = frame.get("message", "")
        turn_id = frame.get("turnId") or str(uuid.uuid4())
        s = await self._repo.get_session(session_id, self._user_id)
        if s is None:
            await self._send({"type": "turn.error", "turnId": turn_id, "error": "session not found"})
            return
        if not message.strip():
            await self._send({"type": "turn.error", "turnId": turn_id, "error": "empty message"})
            return
        await self._send({"type": "turn.accepted", "turnId": turn_id})
        task = asyncio.create_task(self._run_turn(s, message, turn_id))
        self._turns[turn_id] = task
        task.add_done_callback(lambda _t: self._turns.pop(turn_id, None))

    async def _run_turn(self, session: Session, message: str, turn_id: str) -> None:
        settings = get_settings()
        tz = (session.client_metadata or {}).get("timezone")
        try:
            # Assemble prior context BEFORE persisting the new user message,
            # so history holds only earlier turns.
            summary_text, history = await assemble_context(self._repo, session.id)
            await self._repo.append_message(session.id, turn_id, "user", message, message)
            await self._repo.set_title_if_unset(session.id, message)

            events_q: asyncio.Queue = asyncio.Queue()
            collected: list[dict[str, Any]] = []

            def sink(ev: dict[str, Any]) -> None:
                # Stream everything live, but don't persist the high-volume
                # text deltas — the final draft text already captures them.
                if ev.get("phase") != "proposer.text_delta":
                    collected.append(ev)
                events_q.put_nowait(ev)

            async def drain() -> None:
                while True:
                    ev = await events_q.get()
                    if ev is None:
                        return
                    await self._send({"type": "turn.event", "turnId": turn_id, **ev})

            drain_task = asyncio.create_task(drain())

            graph = build_graph(
                self._state.proposer_client,
                self._state.verifier_client,
                checkpointer=self._state.checkpointer,
                event_sink=sink,
                skills=self._state.skills,
            )
            result = await graph.ainvoke(
                {
                    "user_message": message,
                    "timezone": tz,
                    "history": history,
                    "summary_text": summary_text,
                    "revisions": 0,
                },
                {"configurable": {"thread_id": turn_id}},
            )
            events_q.put_nowait(None)
            await drain_task

            final = result.get("final", "")
            agreed = bool(result.get("agreed"))
            verdict = result.get("verdict")
            assistant = await self._repo.append_message(
                session.id,
                turn_id,
                "assistant",
                final,
                final,
                metadata={
                    "proposer_model": settings.proposer_model_id,
                    "verifier_model": settings.verifier_model_id,
                    "agreed": agreed,
                    "revisions": result.get("revisions", 0),
                    "verdict": verdict,
                    "events": collected,
                },
            )
            await self._repo.touch_session(session.id)

            # Opportunistic compaction (no-op when under budget).
            try:
                await maybe_compact(
                    self._repo,
                    session.id,
                    self._state.proposer_client,
                    settings.proposer_model_id,
                    chunk_tokens=settings.compaction_chunk_tokens,
                )
            except Exception:  # noqa: BLE001 - compaction must never fail a turn
                pass

            await self._send(
                {
                    "type": "turn.done",
                    "turnId": turn_id,
                    "final": final,
                    "agreed": agreed,
                    "messageId": assistant.id,
                    "seq": assistant.seq,
                    "createdAt": _iso(assistant.created_at),
                }
            )
        except asyncio.CancelledError:
            await self._send({"type": "turn.error", "turnId": turn_id, "error": "cancelled"})
            raise
        except Exception as e:  # noqa: BLE001 - surface any turn failure
            await self._send({"type": "turn.error", "turnId": turn_id, "error": str(e)})
