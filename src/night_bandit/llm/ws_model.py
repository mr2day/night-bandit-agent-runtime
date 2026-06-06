"""Pydantic AI custom Model backed by the swirlock-llm-host v5 protocol.

Pydantic AI has no built-in provider for our WebSocket host, so we
implement the sanctioned extension point: a :class:`Model` subclass
(template: the in-tree ``FunctionModel``). It maps Pydantic AI's typed
``ModelMessage`` / ``ToolDefinition`` shapes onto the host's wire shape
and maps the host's text deltas + native tool-call events back onto
Pydantic AI's parts manager.

Because both Ministral (proposer) and Qwen (verifier) are tool-trained
and the host now relays native tool calls + multi-step tool messages,
this is a clean native-tool-calling adapter — no prompted/text tool
parsing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .host_client import Done, LlmHostClient, TextDelta, ThinkingDelta, ToolCallDelta
from .tool_repair import StreamingToolRepair, match_tool_call_text

PROVIDER = "swirlock-llm-host"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _content_to_text(content: Any) -> str:
    """Flatten a UserPromptPart content (str | sequence of parts) to text.

    Images are dropped for now (the host supports image parts via a
    different input shape; wiring multimodal through is a later step).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(content)


def _tool_return_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def encode_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Map Pydantic AI ModelMessages onto host wire messages.

    Host messages are ``{role, content, toolCalls?, toolName?}`` with roles
    system/user/assistant/tool, matching Ollama's native shape so the
    multi-step tool loop round-trips.
    """
    out: list[dict[str, Any]] = []

    # Pydantic AI v1 delivers the agent's instructions (static, and from
    # @agent.instructions) via ModelRequest.instructions — a string field,
    # NOT a SystemPromptPart. It's set on the most recent request. Hoist it
    # to a single leading system message; without this the persona and the
    # verifier's JSON contract never reach the model.
    instructions: str | None = None
    for message in messages:
        if isinstance(message, ModelRequest):
            instr = getattr(message, "instructions", None)
            if instr:
                instructions = instr
    if instructions:
        out.append({"role": "system", "content": instructions})

    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    out.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    out.append({"role": "user", "content": _content_to_text(part.content)})
                elif isinstance(part, ToolReturnPart):
                    out.append(
                        {
                            "role": "tool",
                            "content": _tool_return_to_text(part.content),
                            "toolName": part.tool_name,
                        }
                    )
                elif isinstance(part, RetryPromptPart):
                    # A validation/retry instruction — surface to the model
                    # as a user turn carrying the error guidance.
                    out.append({"role": "user", "content": part.model_response()})
        elif isinstance(message, ModelResponse):
            text_chunks: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    text_chunks.append(part.content)
                elif isinstance(part, ThinkingPart):
                    continue  # thinking is not replayed to the model
                elif isinstance(part, ToolCallPart):
                    tool_calls.append(
                        {
                            "id": part.tool_call_id,
                            "function": {
                                "name": part.tool_name,
                                "arguments": part.args_as_dict()
                                if hasattr(part, "args_as_dict")
                                else (part.args if isinstance(part.args, dict) else {}),
                            },
                        }
                    )
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_chunks)}
            if tool_calls:
                msg["toolCalls"] = tool_calls
            out.append(msg)
    return out


def encode_tools(mrp: ModelRequestParameters) -> list[dict[str, Any]] | None:
    """Map Pydantic AI ToolDefinitions to Ollama tool defs."""
    defs: list[ToolDefinition] = [*mrp.function_tools, *mrp.output_tools]
    if not defs:
        return None
    tools: list[dict[str, Any]] = []
    for d in defs:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description or "",
                    "parameters": d.parameters_json_schema,
                },
            }
        )
    return tools


def _ollama_options(model_settings: ModelSettings | None) -> dict[str, Any] | None:
    if not model_settings:
        return None
    opts: dict[str, Any] = {}
    temperature = model_settings.get("temperature")
    if temperature is not None:
        opts["temperature"] = temperature
    top_p = model_settings.get("top_p")
    if top_p is not None:
        opts["top_p"] = top_p
    max_tokens = model_settings.get("max_tokens")
    if max_tokens is not None:
        opts["num_predict"] = max_tokens
    return opts or None


class WebSocketModel(Model):
    """A Pydantic AI model that runs turns on one swirlock-llm-host."""

    def __init__(
        self,
        client: LlmHostClient,
        *,
        model_name: str,
        settings: ModelSettings | None = None,
        tool_call_text_format: str | None = None,
        force_json: bool = False,
    ) -> None:
        """``tool_call_text_format`` enables tail-reliability repair of
        tool calls leaked as text. Pass ``"mistral"`` for Ministral
        (proposer); leave ``None`` to disable.

        ``force_json`` sets the host's ``responseFormat: 'json'`` so Ollama
        constrains the model to emit valid JSON. Use for a structured
        single-shot call (the verifier) where small models are far more
        reliable in JSON mode than via tool-based structured output.
        """
        super().__init__(settings=settings)
        self._client = client
        self._model_name = model_name
        self._tool_format = tool_call_text_format
        self._force_json = force_json

    @property
    def provider(self) -> None:
        return None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return PROVIDER

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        text_chunks: list[str] = []
        parts: list[Any] = []
        async for event in self._client.stream_infer(
            messages=encode_messages(messages),
            tools=encode_tools(model_request_parameters),
            ollama_options=_ollama_options(model_settings),
            response_format="json" if self._force_json else None,
        ):
            if isinstance(event, TextDelta):
                text_chunks.append(event.text)
            elif isinstance(event, ToolCallDelta):
                parts.append(
                    ToolCallPart(
                        tool_name=event.tool_name,
                        args=event.arguments,
                        tool_call_id=event.tool_call_id,
                    )
                )
            elif isinstance(event, (ThinkingDelta, Done)):
                pass
        text = "".join(text_chunks)
        # Tail-reliability: if the model leaked a tool call as text and
        # produced no native call, recover it.
        if text and self._tool_format == "mistral" and not parts:
            leaked = match_tool_call_text(text)
            if leaked is not None:
                name, args = leaked
                parts.append(ToolCallPart(tool_name=name, args=args))
                text = ""
        if text:
            parts.insert(0, TextPart(content=text))
        return ModelResponse(
            parts=parts,
            model_name=self._model_name,
            usage=RequestUsage(),
            provider_name=PROVIDER,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        stream = self._client.stream_infer(
            messages=encode_messages(messages),
            tools=encode_tools(model_request_parameters),
            ollama_options=_ollama_options(model_settings),
            response_format="json" if self._force_json else None,
        )
        response = WebSocketStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _stream=stream,
            _repair_format=self._tool_format,
        )
        try:
            yield response
        finally:
            await stream.aclose()


@dataclass
class WebSocketStreamedResponse(StreamedResponse):
    _model_name: str
    _stream: AsyncIterator[Any]
    _repair_format: str | None = None
    _timestamp: datetime = field(default_factory=_now)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        tool_index = 0
        repair = StreamingToolRepair(enabled=self._repair_format == "mistral")
        async for event in self._stream:
            if isinstance(event, TextDelta):
                safe = repair.feed(event.text)
                if safe:
                    for e in self._parts_manager.handle_text_delta(
                        vendor_part_id="content", content=safe
                    ):
                        yield e
            elif isinstance(event, ToolCallDelta):
                # A native call arrived. Flush any buffered text first so
                # ordering holds, then emit the tool call.
                kind, payload = repair.finish()
                if kind == "text" and payload:
                    for e in self._parts_manager.handle_text_delta(
                        vendor_part_id="content", content=payload
                    ):
                        yield e
                repair = StreamingToolRepair(enabled=self._repair_format == "mistral")
                maybe = self._parts_manager.handle_tool_call_part(
                    vendor_part_id=tool_index,
                    tool_name=event.tool_name,
                    args=event.arguments,
                    tool_call_id=event.tool_call_id,
                )
                tool_index += 1
                if maybe is not None:
                    yield maybe
            elif isinstance(event, Done):
                self.finish_reason = "stop" if event.finish_reason == "stop" else "length"
            # ThinkingDelta: not surfaced as a model part here

        # Stream ended: resolve any buffered text — either flush as prose
        # or recover a leaked tool call.
        kind, payload = repair.finish()
        if kind == "text" and payload:
            for e in self._parts_manager.handle_text_delta(
                vendor_part_id="content", content=payload
            ):
                yield e
        elif kind == "toolcall":
            name, args = payload  # type: ignore[misc]
            maybe = self._parts_manager.handle_tool_call_part(
                vendor_part_id=tool_index, tool_name=name, args=args
            )
            if maybe is not None:
                yield maybe

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str | None:
        return PROVIDER

    @property
    def provider_url(self) -> str | None:
        return None

    @property
    def timestamp(self) -> datetime:
        return self._timestamp
