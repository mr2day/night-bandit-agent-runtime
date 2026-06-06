"""Agent factories: the proposer (Box A / Ministral) and the verifier
(Box B / Qwen), as typed Pydantic AI agents."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext

from ..config import get_settings
from ..llm.host_client import LlmHostClient
from ..llm.ws_model import WebSocketModel
from ..skills import Skill, build_use_skill_tool
from ..tools import builtin_tools
from .persona import NIGHT_BANDIT_SYSTEM, VERIFIER_SYSTEM, substitute


@dataclass
class BanditDeps:
    """Per-run dependencies passed to the proposer agent."""

    timezone: str | None = None


class Verdict(BaseModel):
    """The verifier's structured judgement of the proposer's draft."""

    agrees: bool = Field(
        description="True if the draft is correct and complete enough to send as-is."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems found in the draft (empty if none).",
    )
    corrected_answer: str | None = Field(
        default=None,
        description="A corrected answer, set only when real issues were found.",
    )


def _repair_format_for(model_id: str) -> str | None:
    """Pick the tool-call-text repair format from the model family.

    Role-agnostic: only Mistral-family models leak the ``name[ARGS]{...}``
    pattern the repair targets, so the format follows the model id rather
    than the proposer/verifier role (which we may swap)."""
    mid = model_id.lower()
    if "ministral" in mid or "mistral" in mid:
        return "mistral"
    return None


def build_proposer(
    client: LlmHostClient, skills: list[Skill] | None = None
) -> Agent[BanditDeps, str]:
    """The proposer: drafts answers, calls tools, can invoke skills."""
    settings = get_settings()
    model = WebSocketModel(
        client,
        model_name=settings.proposer_model_id,
        tool_call_text_format=_repair_format_for(settings.proposer_model_id),
    )
    tools = builtin_tools()
    skill_tool = build_use_skill_tool(skills or [])
    if skill_tool is not None:
        tools.append(skill_tool)
    agent: Agent[BanditDeps, str] = Agent(
        model=model,
        deps_type=BanditDeps,
        tools=tools,
        retries=2,
    )

    @agent.instructions
    def _persona(ctx: RunContext[BanditDeps]) -> str:
        return substitute(
            NIGHT_BANDIT_SYSTEM,
            proposer_model=settings.proposer_model_id,
            verifier_model=settings.verifier_model_id,
            timezone=ctx.deps.timezone,
        )

    return agent


def build_verifier(client: LlmHostClient, model_id: str | None = None) -> Agent[None, str]:
    """The verifier: independently checks the proposer's draft.

    Small local models are unreliable at Pydantic AI's tool-based structured
    output, so the verifier runs in Ollama JSON mode (``force_json``) with the
    Verdict schema spelled out in its instructions, returns the JSON as plain
    text, and we parse it with :func:`parse_verdict` (tolerant fallback). This
    is markedly more reliable than ToolOutput for a small model.

    ``model_id`` overrides the configured verifier model — used by the
    verifier benchmark to test candidates head-to-head.
    """
    settings = get_settings()
    mid = model_id or settings.verifier_model_id
    model = WebSocketModel(client, model_name=mid, force_json=True)
    agent: Agent[None, str] = Agent(
        model=model,
        instructions=VERIFIER_SYSTEM,
    )
    return agent


def parse_verdict(text: str) -> Verdict:
    """Parse the verifier's JSON text into a Verdict, tolerantly.

    On any parse/validation failure we fail SAFE toward agreement (treat the
    draft as acceptable) rather than blocking the turn, but record the raw
    text as an issue so the operator can see the verifier misbehaved.
    """
    raw = (text or "").strip()
    # Strip a ```json ... ``` fence if the model added one despite JSON mode.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
        return Verdict.model_validate(data)
    except (ValueError, TypeError, ValidationError):
        return Verdict(
            agrees=True,
            issues=[f"verifier returned unparseable output: {raw[:200]}"],
            corrected_answer=None,
        )
