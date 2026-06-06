-- Night Bandit initial schema. Applied inside the configured schema
-- (search_path is set by the pool), so table names are unqualified.
-- UUIDs are generated app-side (uuid4) so no pgcrypto/extension is needed.

CREATE TABLE IF NOT EXISTS sessions (
  id                uuid PRIMARY KEY,
  user_id           text NOT NULL,
  title             text,
  status            text NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active', 'archived', 'deleted')),
  client_metadata   jsonb,
  total_token_count bigint NOT NULL DEFAULT 0,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sessions_user_updated_idx
  ON sessions (user_id, updated_at DESC)
  WHERE status = 'active';

-- One row per persisted message. A turn typically writes the user
-- message and the assistant's final answer; the assistant row's
-- `metadata` carries the full ensemble trace (proposer/verifier models,
-- verdict, agreement, revision count, and the transparency event list)
-- so a reloaded session can render exactly what the operator saw live.
CREATE TABLE IF NOT EXISTS messages (
  id          uuid PRIMARY KEY,
  session_id  uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  turn_id     uuid NOT NULL,
  role        text NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'system')),
  content     jsonb NOT NULL,
  text        text NOT NULL DEFAULT '',
  seq         bigint NOT NULL,
  metadata    jsonb,
  created_at  timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT messages_session_seq_unique UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS messages_session_seq_idx ON messages (session_id, seq);

CREATE TABLE IF NOT EXISTS session_summaries (
  id            uuid PRIMARY KEY,
  session_id    uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  start_seq     bigint NOT NULL,
  end_seq       bigint NOT NULL,
  summary_text  text NOT NULL,
  token_count   integer NOT NULL,
  summary_model text NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT summaries_seq_range CHECK (end_seq >= start_seq)
);

CREATE INDEX IF NOT EXISTS summaries_session_start_idx
  ON session_summaries (session_id, start_seq);
