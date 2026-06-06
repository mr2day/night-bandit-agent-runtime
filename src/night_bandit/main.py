"""Entry point: FastAPI app + WebSocket gateway + lifespan wiring.

On Windows the whole runtime must run on the Selector event loop
(psycopg-async is incompatible with the Proactor loop), so we set the
policy before anything creates a loop.
"""

from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":  # must precede any loop creation (uvicorn, psycopg)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import AsyncExitStack, asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import quote  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket  # noqa: E402

from .config import get_settings  # noqa: E402
from .gateway.ws_gateway import Connection  # noqa: E402
from .llm.host_client import LlmHostClient  # noqa: E402
from .persistence import Repo, close_pool, get_pool, run_migrations  # noqa: E402
from .skills import load_skills  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_dsn() -> str:
    """The app DSN with search_path set so the checkpointer's tables land
    in our schema too."""
    s = get_settings()
    opt = quote(f"-c search_path={s.pg_schema},public")
    sep = "&" if "?" in s.pg_dsn else "?"
    return f"{s.pg_dsn}{sep}options={opt}"


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _REPO_ROOT / p


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pool = get_pool()
    await pool.open()
    applied = await run_migrations(_REPO_ROOT / "migrations")
    if applied:
        print(f"[migrate] applied: {applied}")

    app.state.repo = Repo()
    app.state.skills = load_skills(_resolve(settings.skills_dir))
    print(f"[skills] loaded: {[s.name for s in app.state.skills]}")
    app.state.proposer_client = LlmHostClient(
        settings.proposer_ws_url, caller_service=settings.caller_service
    )
    app.state.verifier_client = LlmHostClient(
        settings.verifier_ws_url, caller_service=settings.caller_service
    )

    stack = AsyncExitStack()
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        cp = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(_checkpoint_dsn())
        )
        await cp.setup()
        app.state.checkpointer = cp
        print("[checkpointer] AsyncPostgresSaver ready")
    except Exception as e:  # noqa: BLE001 - fall back so the gateway still runs
        from langgraph.checkpoint.memory import InMemorySaver

        app.state.checkpointer = InMemorySaver()
        print(f"[checkpointer] WARN: Postgres unavailable ({e}); using InMemorySaver")

    app.state.exit_stack = stack
    print(
        f"[bandit] proposer={settings.proposer_model_id} verifier={settings.verifier_model_id} "
        f"listening on ws://{settings.host}:{settings.port}{settings.ws_path}"
    )
    try:
        yield
    finally:
        await app.state.proposer_client.aclose()
        await app.state.verifier_client.aclose()
        await stack.aclose()
        await close_pool()


app = FastAPI(lifespan=lifespan)


@app.websocket(get_settings().ws_path)
async def ws_endpoint(ws: WebSocket) -> None:
    conn = Connection(ws, ws.app.state)
    await conn.run()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    s = get_settings()
    # Run uvicorn under our own asyncio.run so it does NOT reconfigure the
    # event loop. On Windows uvicorn's "asyncio" setup installs a Proactor
    # loop, which psycopg-async can't use; loop="none" + asyncio.run (under
    # the Selector policy set at import) keeps us on the Selector loop.
    config = uvicorn.Config(app, host=s.host, port=s.port, loop="none", lifespan="on")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    run()
