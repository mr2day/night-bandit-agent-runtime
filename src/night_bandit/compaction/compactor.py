"""Assemble per-turn context from stored history, and compact when the
verbatim tail outgrows the proposer's effective window.

Originals are never deleted: a summary covers a contiguous seq range while
the underlying messages stay in the table (for the UI's expand affordance
and any future re-summarization).
"""

from __future__ import annotations

from typing import Any

from ..llm.host_client import Done, LlmHostClient, TextDelta
from ..persistence.repo import Message, Repo
from .context_window import estimate_tokens, threshold

_SUMMARY_SYSTEM = (
    "You compress a slice of a conversation into a compact summary that "
    "preserves everything a later turn would need: facts the user gave, "
    "decisions, tasks, conclusions, and unresolved threads. Keep it terse "
    "and in the conversation's own language. No preamble — output only the "
    "summary."
)


async def assemble_context(
    repo: Repo, session_id: str
) -> tuple[str | None, list[dict[str, str]]]:
    """Return (summary_text, verbatim_history) for the next turn.

    summary_text concatenates all stored summaries (oldest first).
    verbatim_history is the user/assistant turns after the last summarized
    seq, as ``{role, content}`` dicts.
    """
    summaries = await repo.get_summaries(session_id)
    messages = await repo.get_messages(session_id)
    last_summarized = max((s.end_seq for s in summaries), default=0)
    summary_text = (
        "\n\n".join(s.summary_text for s in summaries) if summaries else None
    )
    verbatim = [
        {"role": m.role, "content": m.text}
        for m in messages
        if m.seq > last_summarized and m.role in ("user", "assistant") and m.text
    ]
    return summary_text, verbatim


def _transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        who = m.role.upper()
        lines.append(f"{who}: {m.text}")
    return "\n\n".join(lines)


async def _summarize(client: LlmHostClient, transcript: str) -> str:
    chunks: list[str] = []
    async for ev in client.stream_infer(
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": f"Conversation slice:\n\n{transcript}"},
        ],
    ):
        if isinstance(ev, TextDelta):
            chunks.append(ev.text)
        elif isinstance(ev, Done):
            break
    return "".join(chunks).strip()


async def maybe_compact(
    repo: Repo,
    session_id: str,
    client: LlmHostClient,
    model_id: str,
    *,
    chunk_tokens: int,
) -> dict[str, Any] | None:
    """Summarize the oldest verbatim chunk if the session is over budget.

    Returns a small dict describing the new summary, or None if nothing was
    compacted. Safe to call after every turn; it no-ops when under budget.
    """
    summaries = await repo.get_summaries(session_id)
    messages = await repo.get_messages(session_id)
    last_summarized = max((s.end_seq for s in summaries), default=0)
    summary_tokens = sum(s.token_count for s in summaries)
    verbatim = [m for m in messages if m.seq > last_summarized and m.text]
    verbatim_tokens = sum(estimate_tokens(m.text) for m in verbatim)

    if summary_tokens + verbatim_tokens < threshold():
        return None
    if not verbatim:
        return None

    # Take the oldest contiguous slice whose tokens reach chunk_tokens.
    acc = 0
    cutoff = 0
    for i, m in enumerate(verbatim):
        acc += estimate_tokens(m.text)
        if acc >= chunk_tokens:
            cutoff = i + 1
            break
    if cutoff == 0:
        cutoff = len(verbatim)
    slice_ = verbatim[:cutoff]

    summary_text = await _summarize(client, _transcript(slice_))
    if not summary_text:
        return None

    start_seq = slice_[0].seq
    end_seq = slice_[-1].seq
    tokens = estimate_tokens(summary_text)
    await repo.add_summary(session_id, start_seq, end_seq, summary_text, tokens, model_id)
    return {
        "start_seq": start_seq,
        "end_seq": end_seq,
        "token_count": tokens,
        "summary_model": model_id,
    }
