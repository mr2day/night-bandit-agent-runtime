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

The backend is complete and live-verified end-to-end:
- `llm/` — v5 host client + Pydantic AI custom Model (native tool calls,
  streaming, JSON mode, Mistral tool-leak repair).
- `tools/` — get_current_time, search_web (Exa), fetch_page (Exa), browse
  (Playwright).
- `skills/` — SKILL.md loader + `use_skill` tool (example skills:
  deep_research, check_current_info).
- `agents/` — Night Bandit persona, proposer (Box A), verifier (Box B,
  JSON-mode structured verdicts).
- `orchestration/` — LangGraph proposer → verifier → (revise | finalize),
  live transparency events (text deltas, tool calls/results, verdicts),
  tool **evidence** passed to the verifier so it judges against facts.
- `persistence/` — async pool, schema migrations, sessions/messages/
  summaries repo (dedicated `night_bandit` schema).
- `compaction/` — context-window math + idle compactor.
- `gateway/` — FastAPI WebSocket gateway (IdP auth, session CRUD, turn
  streaming) + Postgres checkpointer (durable execution).

Verified: auth, session CRUD, multi-turn history, native tool use with
correct result handling, the proposer/verifier ensemble (agreement +
disagreement + revise), full live transparency, persistence + reload,
durable checkpoints.

Remaining (separate repo): the Angular UI (`night-bandit-agent-ui`).

## Run

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install -e .   # Windows
cp .env.example .env                          # then edit secrets
./.venv/Scripts/python -m night_bandit.main   # starts ws://127.0.0.1:3220/v6/bandit
```

Requires the two swirlock-llm-host instances reachable (Box A local, Box B
on the LAN) and Postgres reachable via `NB_PG_DSN`. Windows runs on the
Selector event loop automatically (psycopg-async requirement). For `browse`,
also run `./.venv/Scripts/playwright install chromium` once.

## Develop

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install -e .   # Windows
cp .env.example .env                          # then edit
```

Requires the two swirlock-llm-host instances reachable (Box A local,
Box B on the LAN). Python ≥ 3.10.
