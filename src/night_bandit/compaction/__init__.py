"""Conversation compaction: keep a session under the proposer model's
effective context window by rolling old turns into summaries."""

from .compactor import assemble_context, maybe_compact
from .context_window import estimate_tokens, threshold

__all__ = ["assemble_context", "maybe_compact", "estimate_tokens", "threshold"]
