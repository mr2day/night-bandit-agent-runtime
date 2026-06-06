"""Convert stored conversation history (plain dicts) into Pydantic AI
message history for the proposer.

State stays JSON-simple ({role, content} dicts + a summary string) so the
LangGraph Postgres checkpointer can serialize it; we materialize Pydantic
AI ModelMessage objects only at run time inside the proposer node.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)


def to_model_history(
    summary_text: str | None, history: list[dict[str, Any]] | None
) -> list[ModelMessage]:
    """Build message_history: optional leading summary, then prior turns.

    ``history`` items are ``{"role": "user"|"assistant", "content": str}``.
    Only user/assistant turns are replayed (tool rows are internal to a
    past turn's agent loop and aren't needed cross-turn).
    """
    out: list[ModelMessage] = []
    if summary_text:
        out.append(
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content=(
                            "[Summary of earlier conversation, for context]\n"
                            + summary_text
                        )
                    )
                ]
            )
        )
    for item in history or []:
        role = item.get("role")
        content = item.get("content") or ""
        if role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            out.append(ModelResponse(parts=[TextPart(content=content)]))
    return out
