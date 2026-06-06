"""Runtime configuration, sourced entirely from the environment.

Nothing here is hardcoded business logic — every value flows from an env
var (loaded from a local ``.env`` in development via python-dotenv) with a
non-secret default where one is safe. Secrets (Postgres DSN, Exa key) have
no default and must be supplied.

The two model hosts are first-class: the Night Bandit is a two-LLM
ensemble, so the proposer host (Box A) and verifier host (Box B) each get
their own URL + display model id.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # no-op if .env is absent (production sets real env vars)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NB_", extra="ignore")

    # --- HTTP / WS server (the gateway the UI connects to) ---
    host: str = "127.0.0.1"
    port: int = 3220
    # Path the UI's WebSocket connects to. Path-mounted so the whole app
    # lives under gigi-the-robot.com/bandit with no Cloudflare DNS work.
    ws_path: str = "/v6/bandit"

    # --- Proposer LLM host (Box B, LAN / qwen) ---
    # Roles reversed 2026-06-07: the proposer (easier job: drafting) runs on
    # the 9B; the verifier (harder job: judging) runs on the 14B Ministral.
    proposer_ws_url: str = "ws://192.168.0.194:3213/v5/model"
    proposer_model_id: str = "qwen3.5:9b"

    # --- Verifier LLM host (Box A, local / ministral) ---
    verifier_ws_url: str = "ws://127.0.0.1:3213/v5/model"
    verifier_model_id: str = "ministral-3:14b"

    # Caller identity stamped on every llm-host requestContext.
    caller_service: str = "night-bandit-agent-runtime"

    # --- Persistence ---
    # Postgres DSN for both the app tables (sessions/messages/summaries)
    # and the LangGraph Postgres checkpointer (durable execution). No
    # default — must be supplied.
    pg_dsn: str = Field(default=..., description="postgresql://user:pass@host:port/db")

    # --- Tools ---
    exa_api_key: str = Field(default=..., description="Exa API key for search_web / fetch_page")
    fetch_page_max_chars: int = 12000
    browse_max_text_chars: int = 12000
    browse_max_links: int = 40
    browse_nav_timeout_ms: int = 30000

    # --- Auth (IdP, OIDC) ---
    idp_issuer: str = "https://idpbase.swirlock.com/oidc"
    idp_audience: str = "https://api.gigi-the-robot.com"
    idp_jwks_uri: str = "https://idpbase.swirlock.com/oidc/jwks"
    # Dev-only escape hatch for localhost smoke tests. NEVER true in prod.
    dev_bypass_auth: bool = False

    # --- Agent loop ---
    max_steps: int = 8
    max_output_tokens: int = 4096

    # --- Ensemble ---
    # How many proposer<->verifier revise cycles before finalizing on
    # disagreement. 1 = at most one revision, then ship the best draft
    # (or the verifier's correction) and surface the disagreement.
    ensemble_max_revisions: int = 1

    # --- Context window / compaction (proposer host is the landmark) ---
    proposer_num_ctx: int = 32768
    context_reserve_tokens: int = 3000
    context_compact_ratio: float = 0.70
    compaction_chunk_tokens: int = 6000

    # --- Skills ---
    # Directory scanned for SKILL.md capability definitions, relative to
    # the repo root. Absolute paths are honoured as-is.
    skills_dir: str = "skills"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Import and call where config is needed."""
    return Settings()
