"""The ensemble graph: proposer -> verifier -> (revise | finalize).

The proposer (Box A / Ministral) drafts an answer, calling tools as
needed. The verifier (Box B / Qwen), a different model family, judges the
draft independently and returns a structured Verdict. If it agrees (or the
revision budget is spent) we finalize; otherwise the proposer revises once
with the verifier's critique in hand.

Durability + HITL come from the checkpointer passed to ``build_graph``:
with an ``AsyncPostgresSaver`` the run survives process restarts and an
``interrupt()`` can pause for human approval (the natural insertion point
is a gate node before any future write/side-effecting tool — read-only
tools like search/fetch/browse don't need it). Every node emits structured
transparency events on the custom stream channel.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
)

from ..agents.agents import BanditDeps, build_proposer, build_verifier, parse_verdict
from ..config import get_settings
from ..llm.host_client import LlmHostClient
from .events import EventSink, emit
from .history import to_model_history
from .state import EnsembleState

import json as _json


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return _json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _evidence_block(evidence: list[dict[str, Any]] | None) -> str:
    if not evidence:
        return ""
    lines = ["\n\nEvidence the proposer gathered (trust these over your own training):"]
    for e in evidence:
        lines.append(
            f"- {e.get('tool_name')}({_stringify(e.get('args'))}) -> {e.get('result')}"
        )
    return "\n".join(lines)


def build_graph(
    proposer_client: LlmHostClient,
    verifier_client: LlmHostClient,
    *,
    checkpointer: Any | None = None,
    event_sink: EventSink | None = None,
    skills: Any | None = None,
) -> Any:
    """Build + compile the ensemble graph.

    ``checkpointer`` (Postgres in production, InMemorySaver in tests)
    enables durability + HITL. ``event_sink`` receives every transparency
    event as it happens; the gateway forwards them to the UI. ``skills`` is
    the loaded skill list exposed to the proposer via the use_skill tool.
    """
    settings = get_settings()
    proposer = build_proposer(proposer_client, skills or [])
    verifier = build_verifier(verifier_client)
    max_revisions = settings.ensemble_max_revisions

    async def proposer_node(state: EnsembleState) -> dict[str, Any]:
        revisions = state.get("revisions", 0)
        emit(event_sink, "proposer.start", model=settings.proposer_model_id, revision=revisions)

        prompt = state["user_message"]
        verdict = state.get("verdict")
        if revisions > 0 and verdict:
            issues = "; ".join(verdict.get("issues") or [])
            prompt = (
                f"{state['user_message']}\n\n"
                f"[Your previous draft was reviewed by the verifier model "
                f"({settings.verifier_model_id}). Issues raised: {issues}. "
                f"Revise your answer to address them.]"
            )

        deps = BanditDeps(timezone=state.get("timezone"))
        message_history = to_model_history(state.get("summary_text"), state.get("history"))

        # Stream the proposer's internals so the operator sees text appear
        # live and every tool call/result as it happens (max transparency).
        # Also capture tool evidence so the verifier can judge against facts
        # the proposer gathered, not just the draft text.
        draft = ""
        calls: dict[str, dict[str, Any]] = {}
        evidence: list[dict[str, Any]] = []
        async with proposer.iter(
            prompt, deps=deps, message_history=message_history or None
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for ev in stream:
                            if isinstance(ev, PartDeltaEvent) and isinstance(
                                ev.delta, TextPartDelta
                            ):
                                if ev.delta.content_delta:
                                    emit(event_sink, "proposer.text_delta", delta=ev.delta.content_delta)
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for ev in stream:
                            if isinstance(ev, FunctionToolCallEvent):
                                calls[ev.part.tool_call_id] = {
                                    "tool_name": ev.part.tool_name,
                                    "args": ev.part.args,
                                }
                                emit(
                                    event_sink,
                                    "proposer.tool_call",
                                    tool_name=ev.part.tool_name,
                                    args=ev.part.args,
                                )
                            elif isinstance(ev, FunctionToolResultEvent):
                                res = getattr(ev, "result", None)
                                name = getattr(res, "tool_name", None) or calls.get(
                                    getattr(ev, "tool_call_id", ""), {}
                                ).get("tool_name")
                                result_str = _stringify(getattr(res, "content", None))
                                evidence.append(
                                    {
                                        "tool_name": name,
                                        "args": calls.get(
                                            getattr(ev, "tool_call_id", ""), {}
                                        ).get("args"),
                                        "result": result_str,
                                    }
                                )
                                emit(
                                    event_sink,
                                    "proposer.tool_result",
                                    tool_name=name,
                                    result=result_str[:500],
                                )
        draft = run.result.output if run.result is not None else draft
        emit(event_sink, "proposer.draft", text=draft)
        return {"draft": draft, "evidence": evidence}

    async def verifier_node(state: EnsembleState) -> dict[str, Any]:
        emit(event_sink, "verifier.start", model=settings.verifier_model_id)
        prompt = (
            f"User question:\n{state['user_message']}\n\n"
            f"Proposer's draft answer:\n{state['draft']}"
            f"{_evidence_block(state.get('evidence'))}\n\n"
            f"Judge the draft independently. Do you agree it is correct and "
            f"complete enough to send? List concrete issues; if there are real "
            f"problems, provide a corrected answer."
        )
        result = await verifier.run(prompt)
        vd = parse_verdict(result.output).model_dump()
        emit(
            event_sink,
            "verifier.verdict",
            agrees=vd["agrees"],
            issues=vd["issues"],
            has_correction=vd.get("corrected_answer") is not None,
        )
        return {"verdict": vd}

    def route_after_verify(state: EnsembleState) -> str:
        verdict = state.get("verdict") or {}
        revisions = state.get("revisions", 0)
        if verdict.get("agrees") or revisions >= max_revisions:
            return "finalize"
        return "revise"

    async def revise_node(state: EnsembleState) -> dict[str, Any]:
        nxt = state.get("revisions", 0) + 1
        emit(event_sink, "revise", revision=nxt)
        return {"revisions": nxt}

    async def finalize_node(state: EnsembleState) -> dict[str, Any]:
        verdict = state.get("verdict") or {}
        agreed = bool(verdict.get("agrees"))
        if not agreed and verdict.get("corrected_answer"):
            final = verdict["corrected_answer"]
        else:
            final = state.get("draft", "")
        emit(event_sink, "final", text=final, agreed=agreed)
        return {"final": final, "agreed": agreed}

    g: StateGraph = StateGraph(EnsembleState)
    g.add_node("proposer", proposer_node)
    g.add_node("verifier", verifier_node)
    g.add_node("revise", revise_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "proposer")
    g.add_edge("proposer", "verifier")
    g.add_conditional_edges(
        "verifier", route_after_verify, {"finalize": "finalize", "revise": "revise"}
    )
    g.add_edge("revise", "proposer")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
