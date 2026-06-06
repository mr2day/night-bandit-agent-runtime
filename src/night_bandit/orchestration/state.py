"""The ensemble graph's state."""

from __future__ import annotations

from typing import Any, TypedDict


class EnsembleState(TypedDict, total=False):
    # Input
    user_message: str
    timezone: str | None
    history: list[dict[str, str]]  # prior turns as {role, content}

    # Working state
    draft: str  # proposer's current draft
    verdict: dict[str, Any] | None  # verifier's Verdict, model_dump()
    revisions: int  # how many revise cycles have happened

    # Output
    final: str  # the answer to return to the user
    agreed: bool  # did the verifier agree on the final draft?
