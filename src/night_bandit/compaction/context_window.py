"""Context-window math. The proposer is the landmark — it carries the
conversation history, so its effective window drives compaction."""

from __future__ import annotations

import math

from ..config import get_settings


def estimate_tokens(text: str) -> int:
    """char-count / 4 heuristic — accurate enough to trigger a background
    summarization, never used for billing or context construction."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def effective_window() -> int:
    s = get_settings()
    return max(1024, s.proposer_num_ctx - s.context_reserve_tokens)


def threshold() -> int:
    """When summaries+verbatim exceed this, compact the oldest chunk."""
    return math.floor(effective_window() * get_settings().context_compact_ratio)
