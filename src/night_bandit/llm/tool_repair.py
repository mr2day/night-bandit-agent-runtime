"""Repair for tool calls that small local models occasionally leak as
text instead of triggering the host's native tool-call path.

Observed on Mistral-family models (Ministral): instead of leading with
the ``[TOOL_CALLS]`` sentinel Ollama's parser extracts, the model emits
just the body — ``add_numbers[ARGS]{"a": 17, "b": 25}`` — as plain text,
so the call leaks through as assistant content. Native tool-calling
works the large majority of the time (measured 5/5 in steady state); this
is the tail-reliability net for the cold-start / awkward-prompt cases.

The streaming repair mirrors the incremental-buffering design proven in
swirlock-agent-runtime: hold text deltas only while the buffer could
still grow into the malformed pattern; the instant it can't, flush and
stream normally (≈ one token of perceived delay for ordinary prose).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Full malformed pattern: <identifier>[ARGS]{<json>}
_FULL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\[ARGS\](\{.*\})\s*$", re.DOTALL)

# "Could this buffer still become the malformed prefix?" — identifier,
# then optionally the [ARGS] sentinel being typed, then optionally {json.
_PARTIAL = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\[(?:A(?:R(?:G(?:S(?:\](?:\{.*)?)?)?)?)?)?)?$",
    re.DOTALL,
)


def match_tool_call_text(text: str) -> tuple[str, dict] | None:
    """Return (tool_name, args) if `text` is a leaked tool call, else None."""
    m = _FULL.match(text)
    if not m:
        return None
    name, args_json = m.group(1), m.group(2)
    try:
        args = json.loads(args_json)
    except (ValueError, TypeError):
        return None
    if not isinstance(args, dict):
        return None
    return name, args


def _could_still_match(buffer: str) -> bool:
    stripped = buffer.lstrip()
    if not stripped:
        return True
    return bool(_PARTIAL.match(stripped))


@dataclass
class StreamingToolRepair:
    """Incremental gate over a single text stream.

    Feed each text delta; it returns the text that is safe to emit now
    (empty while still inspecting a possible malformed prefix). Call
    :meth:`finish` at stream end to get the final disposition.
    """

    enabled: bool = True
    _mode: str = "inspecting"  # 'inspecting' | 'passthrough'
    _buffer: str = ""
    _emitted_any: bool = field(default=False)

    def feed(self, delta: str) -> str:
        if not self.enabled or self._mode == "passthrough":
            return delta
        self._buffer += delta
        if _could_still_match(self._buffer):
            return ""  # keep buffering
        # Broke the prefix → ordinary prose. Flush buffer, go passthrough.
        self._mode = "passthrough"
        out, self._buffer = self._buffer, ""
        return out

    def finish(self) -> tuple[str, str | tuple[str, dict]]:
        """Return ('text', remaining_text) or ('toolcall', (name, args))."""
        if self._mode == "passthrough" or not self.enabled:
            out, self._buffer = self._buffer, ""
            return ("text", out)
        # Still inspecting at end: full match → tool call, else text.
        match = match_tool_call_text(self._buffer)
        if match is not None:
            self._buffer = ""
            return ("toolcall", match)
        out, self._buffer = self._buffer, ""
        return ("text", out)
