"""Async client for the swirlock-llm-host v5 WebSocket protocol.

The host is a thin relay around ``ollama.chat()``: raw turns in, raw
deltas out (text, thinking, and — since the native-tool-calling fix —
tool calls). This client owns one persistent connection per host and
multiplexes concurrent inference requests over it by ``correlationId``.

It exposes:
- :meth:`stream_infer` — an async generator of typed stream events for
  one inference turn (text deltas, thinking deltas, tool calls, done).
- :meth:`model_status` / :meth:`health` — single request/reply calls.
- :meth:`cancel` — abort an in-flight turn.

Everything above the wire (tool execution, the agent loop, verification)
lives in the runtime, never in the host.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection


# --- Typed stream events yielded by stream_infer -------------------------


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ThinkingDelta:
    text: str


@dataclass(slots=True)
class ToolCallDelta:
    """A native tool call relayed by the host, in Ollama's shape:
    ``{ id?, function: { name, arguments } }`` where ``arguments`` is an
    already-parsed dict. One host ``tool_call`` event may carry several."""

    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str


@dataclass(slots=True)
class Done:
    finish_reason: str  # 'stop' | 'length' | 'error'


@dataclass(slots=True)
class HostError(Exception):
    code: str
    message: str
    retryable: bool

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


StreamEvent = TextDelta | ThinkingDelta | ToolCallDelta | Done


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class _Pending:
    """A registered in-flight request keyed by correlationId."""

    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)


class LlmHostClient:
    """One persistent connection to a single swirlock-llm-host instance."""

    def __init__(self, ws_url: str, *, caller_service: str) -> None:
        self._ws_url = ws_url
        self._caller = caller_service
        self._conn: ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[str, _Pending] = {}
        self._connect_lock = asyncio.Lock()

    @property
    def ws_url(self) -> str:
        return self._ws_url

    async def _ensure_connected(self) -> ClientConnection:
        if self._conn is not None:
            return self._conn
        async with self._connect_lock:
            if self._conn is not None:  # re-check inside the lock
                return self._conn
            conn = await websockets.connect(
                self._ws_url, max_size=64 * 1024 * 1024, ping_interval=20
            )
            self._conn = conn
            self._reader_task = asyncio.create_task(self._read_loop(conn))
            return conn

    async def _read_loop(self, conn: ClientConnection) -> None:
        """Route every inbound frame to the queue for its correlationId."""
        try:
            async for raw in conn:
                try:
                    frame = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                cid = frame.get("correlationId")
                pending = self._pending.get(cid) if cid else None
                if pending is not None:
                    await pending.queue.put(frame)
        except (websockets.ConnectionClosed, OSError):
            pass
        finally:
            # Connection dropped: wake every waiter with a synthetic error
            # so no stream_infer hangs forever.
            for pending in list(self._pending.values()):
                await pending.queue.put(
                    {
                        "type": "error",
                        "error": {
                            "code": "upstream_unavailable",
                            "message": "llm-host connection closed",
                            "retryable": True,
                        },
                    }
                )
            if self._conn is conn:
                self._conn = None

    async def _send(self, frame: dict[str, Any]) -> None:
        conn = await self._ensure_connected()
        await conn.send(json.dumps(frame))

    # --- Streaming inference --------------------------------------------

    async def stream_infer(
        self,
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_format: str | None = None,
        thinking: bool | None = None,
        ollama_options: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one inference turn, yielding typed stream events.

        ``messages`` are ``{role, content}`` dicts (system/user/assistant).
        ``tools`` are Ollama-shaped tool definitions. The generator
        completes after yielding :class:`Done`; on a host error it raises
        :class:`HostError`.
        """
        cid = str(uuid.uuid4())
        pending = _Pending()
        self._pending[cid] = pending

        options: dict[str, Any] = {}
        if response_format is not None:
            options["responseFormat"] = response_format
        if thinking is not None:
            options["thinking"] = thinking
        if ollama_options:
            options["ollama"] = ollama_options
        if tools:
            options["tools"] = tools

        request: dict[str, Any] = {
            "requestContext": {
                "callerService": self._caller,
                "requestedAt": _now_iso(),
            },
            "input": {"messages": messages},
        }
        if options:
            request["options"] = options

        try:
            await self._send(
                {"type": "infer", "correlationId": cid, "payload": {"request": request}}
            )
            while True:
                frame = await pending.queue.get()
                ftype = frame.get("type")
                payload = frame.get("payload") or {}
                if ftype == "chunk":
                    text = payload.get("text", "")
                    if text:
                        yield TextDelta(text)
                elif ftype == "thinking":
                    text = payload.get("text", "")
                    if text:
                        yield ThinkingDelta(text)
                elif ftype == "tool_call":
                    for call in payload.get("toolCalls", []):
                        fn = call.get("function", {}) if isinstance(call, dict) else {}
                        name = fn.get("name")
                        if not name:
                            continue
                        args = fn.get("arguments")
                        if not isinstance(args, dict):
                            args = {}
                        yield ToolCallDelta(
                            tool_name=name,
                            arguments=args,
                            tool_call_id=call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        )
                elif ftype == "done":
                    yield Done(finish_reason=payload.get("finishReason", "stop"))
                    return
                elif ftype == "error":
                    err = frame.get("error") or {}
                    raise HostError(
                        code=err.get("code", "internal_error"),
                        message=err.get("message", "unknown host error"),
                        retryable=bool(err.get("retryable", False)),
                    )
                # accepted / queued / started carry no agent-visible payload
        finally:
            self._pending.pop(cid, None)

    async def cancel(self, correlation_id: str) -> None:
        """Best-effort abort of an in-flight turn."""
        try:
            await self._send({"type": "cancel", "correlationId": correlation_id})
        except (HostError, OSError, websockets.ConnectionClosed):
            pass

    # --- Request/reply calls --------------------------------------------

    async def _request_reply(
        self, msg_type: str, reply_type: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cid = str(uuid.uuid4())
        pending = _Pending()
        self._pending[cid] = pending
        try:
            await self._send(
                {"type": msg_type, "correlationId": cid, "payload": payload or {}}
            )
            while True:
                frame = await pending.queue.get()
                if frame.get("type") == reply_type:
                    return frame.get("payload") or {}
                if frame.get("type") == "error":
                    err = frame.get("error") or {}
                    raise HostError(
                        code=err.get("code", "internal_error"),
                        message=err.get("message", "unknown host error"),
                        retryable=bool(err.get("retryable", False)),
                    )
        finally:
            self._pending.pop(cid, None)

    async def model_status(self) -> dict[str, Any]:
        return await self._request_reply("model.status", "model.status")

    async def health(self) -> dict[str, Any]:
        return await self._request_reply("health.get", "health")

    async def aclose(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
