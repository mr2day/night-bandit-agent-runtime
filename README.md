# Night Bandit — Agent Runtime

A capable, **local-only**, two-LLM ensemble agent. Not a chatbot — the
chat UI is just one surface. The runtime does the heavy lifting; the
model hosts are thin relays.

## Architecture

```
UI  ──WS /v6/bandit──▶  night-bandit-agent-runtime (Python)
                          │
                          │  LangGraph spine: supervisor routing,
                          │  durable execution (Postgres checkpointer),
                          │  human-in-the-loop, conditional/cyclic flow
                          │
                          │  Pydantic AI typed agent nodes:
                          │   • proposer  (drafts, calls tools)
                          │   • verifier  (cross-checks the draft)
                          │
                          ▼
        ┌────────────────────────┐   ┌────────────────────────┐
        │ swirlock-llm-host A     │   │ swirlock-llm-host B     │
        │ ws://127.0.0.1:3213     │   │ ws://192.168.0.194:3213 │
        │ ministral-3:14b (32k)   │   │ qwen3.5:9b              │
        │  = PROPOSER             │   │  = VERIFIER            │
        └────────────────────────┘   └────────────────────────┘
```

The two models come from **different families** so their hallucinations
are uncorrelated: the proposer drafts, the verifier independently checks.

## Stack

- **LangGraph 1.2.4** — orchestration spine (StateGraph, `AsyncPostgresSaver`
  for durable execution, `interrupt()` for HITL, custom stream events for
  the transparency UI).
- **Pydantic AI 1.106 (slim)** — typed agent nodes with a **custom Model**
  (`night_bandit.llm.ws_model.WebSocketModel`) that speaks the
  swirlock-llm-host v5 WebSocket protocol. Native tool-calling round-trips
  through the host; a Mistral-format text-leak repair provides tail
  reliability.
- **FastAPI + websockets** — the gateway the UI connects to.
- **Postgres** — app tables (sessions / messages / summaries) + the
  LangGraph checkpointer.

## Status

Built and live-verified:
- `llm/host_client.py` — async v5 client (infer stream, native tool calls,
  multi-step tool loop, health/status). Verified against Box A + Box B.
- `llm/ws_model.py` — Pydantic AI custom Model. Verified: non-streaming +
  streaming runs, native tool loop (5/5), tool-leak repair net.
- `config.py` — env-driven settings.

Next increments: tools (search_web / fetch_page / browse / get_current_time),
skills loader, proposer/verifier agents + Night Bandit persona, the LangGraph
ensemble graph, persistence + compactor, the FastAPI gateway.

## Develop

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install -e .   # Windows
cp .env.example .env                          # then edit
```

Requires the two swirlock-llm-host instances reachable (Box A local,
Box B on the LAN). Python ≥ 3.10.
