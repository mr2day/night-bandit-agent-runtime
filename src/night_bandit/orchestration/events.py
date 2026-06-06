"""Transparency events emitted from inside graph nodes.

The Night Bandit's user is the operator, who wants to see everything the
ensemble does. Nodes emit structured events through an ``EventSink`` — a
plain callable closed over by the graph at build time — which the gateway
forwards to the UI verbatim so the operator sees the proposer/verifier
dance live.

A closure sink is used rather than LangGraph's ``get_stream_writer()`` /
injected ``StreamWriter`` because both route through ``get_config()``,
whose contextvar machinery has an async footgun on Python < 3.11. The
sink works on any Python version and keeps transparency independent of the
graph's own stream channel.
"""

from __future__ import annotations

from typing import Any, Protocol


class EventSink(Protocol):
    def __call__(self, event: dict[str, Any]) -> None: ...


def emit(sink: EventSink | None, phase: str, **data: Any) -> None:
    """Emit one transparency event. No-op if no sink is wired."""
    if sink is None:
        return
    sink({"phase": phase, **data})
