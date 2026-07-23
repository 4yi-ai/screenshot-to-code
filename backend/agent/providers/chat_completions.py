# pyright: reportUnknownVariableType=false
import base64
import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.state import ensure_str
from agent.providers.anthropic.image import process_image
from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)
from agent.tools import CanonicalToolDefinition, ToolCall, parse_json_arguments


def _get_attr(value: Any, key: str, default: Any = None) -> Any:
    if hasattr(value, key):
        return getattr(value, key)
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def _process_data_url_image(image_url: str) -> str:
    if not image_url.startswith("data:image/") or ";base64," not in image_url:
        return image_url

    media_type, base64_data = process_image(image_url)
    return f"data:{media_type};base64,{base64_data}"


def _tool_image_ref(part: Any) -> str | None:
    if part.image_url:
        return part.image_url
    if part.data is None:
        return None
    encoded = base64.b64encode(part.data).decode("ascii")
    return _process_data_url_image(f"data:{part.mime_type};base64,{encoded}")


def _prepare_prompt_messages(
    prompt_messages: List[ChatCompletionMessageParam],
) -> List[ChatCompletionMessageParam]:
    prepared_messages = copy.deepcopy(prompt_messages)
    for message in prepared_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = cast(object, image_url.get("url"))
            if isinstance(url, str):
                image_url["url"] = _process_data_url_image(url)
    return prepared_messages


def serialize_chat_completions_tools(
    tools: List[CanonicalToolDefinition],
) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": copy.deepcopy(tool.parameters),
            },
        }
        for tool in tools
    ]


@dataclass
class ChatCompletionsParseState:
    assistant_text: str = ""
    tool_calls_by_index: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def _tool_call_wire_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    call_id = ensure_str(entry.get("id")) or f"call-{uuid.uuid4().hex[:8]}"
    name = ensure_str(entry.get("name")) or "unknown_tool"
    arguments = ensure_str(entry.get("arguments"))
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _tool_call_from_wire(wire_tool_call: Dict[str, Any]) -> ToolCall:
    function = wire_tool_call.get("function") or {}
    raw_args = function.get("arguments")
    args, error = parse_json_arguments(raw_args)
    if error:
        args = {"INVALID_JSON": ensure_str(raw_args)}
    return ToolCall(
        id=ensure_str(wire_tool_call.get("id")) or f"call-{uuid.uuid4().hex[:8]}",
        name=ensure_str(function.get("name")) or "unknown_tool",
        arguments=args,
    )


def _build_provider_turn(
    assistant_text: str,
    raw_tool_calls: List[Dict[str, Any]],
) -> ProviderTurn:
    wire_tool_calls = [
        _tool_call_wire_entry(tool_call)
        for tool_call in raw_tool_calls
        if ensure_str(tool_call.get("name"))
    ]
    tool_calls = [_tool_call_from_wire(tool_call) for tool_call in wire_tool_calls]
    assistant_turn = (
        {
            "role": "assistant",
            "content": assistant_text or None,
            "tool_calls": wire_tool_calls,
        }
        if tool_calls
        else None
    )
    return ProviderTurn(
        assistant_text=assistant_text,
        tool_calls=tool_calls,
        assistant_turn=assistant_turn,
    )


async def _parse_stream_chunk(
    chunk: Any,
    state: ChatCompletionsParseState,
    on_event: EventSink,
) -> None:
    choices = _get_attr(chunk, "choices", None) or []
    if not choices:
        return

    delta = _get_attr(choices[0], "delta", None)
    if delta is None:
        return

    piece = _get_attr(delta, "content", None)
    if piece:
        text = ensure_str(piece)
        state.assistant_text += text
        await on_event(StreamEvent(type="assistant_delta", text=text))

    tool_call_deltas = _get_attr(delta, "tool_calls", None) or []
    for fallback_index, tool_call_delta in enumerate(tool_call_deltas):
        index = _get_attr(tool_call_delta, "index", fallback_index)
        if not isinstance(index, int):
            index = fallback_index
        entry = state.tool_calls_by_index.setdefault(
            index,
            {
                "id": None,
                "name": None,
                "arguments": "",
            },
        )

        call_id = _get_attr(tool_call_delta, "id", None)
        if call_id:
            entry["id"] = call_id

        function = _get_attr(tool_call_delta, "function", None)
        if function:
            name = _get_attr(function, "name", None)
            if name:
                entry["name"] = name
            arguments_delta = _get_attr(function, "arguments", None)
            if arguments_delta:
                entry["arguments"] = ensure_str(entry.get("arguments")) + ensure_str(
                    arguments_delta
                )

        await on_event(
            StreamEvent(
                type="tool_call_delta",
                tool_call_id=ensure_str(entry.get("id")) or None,
                tool_name=ensure_str(entry.get("name")) or None,
                tool_arguments=ensure_str(entry.get("arguments")),
            )
        )


def _extract_response_tool_calls(message: Any) -> List[Dict[str, Any]]:
    raw_tool_calls = _get_attr(message, "tool_calls", None) or []
    tool_calls: List[Dict[str, Any]] = []
    for raw_tool_call in raw_tool_calls:
        function = _get_attr(raw_tool_call, "function", None) or {}
        tool_calls.append(
            {
                "id": _get_attr(raw_tool_call, "id", None),
                "name": _get_attr(function, "name", None),
                "arguments": _get_attr(function, "arguments", ""),
            }
        )
    return tool_calls


class ChatCompletionsProviderSession(ProviderSession):
    """OpenAI-compatible chat/completions session used to route generation through
    the 4yi gateway.

    The gateway often routes to Bedrock Claude models, which reject images with
    dimensions >= 8000 px. Native Anthropic calls already normalize screenshots;
    the OpenAI-compatible gateway payload needs the same treatment.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        prompt_messages: List[ChatCompletionMessageParam],
        tools: List[Any],
    ):
        self._client = client
        self._model_name = model_name
        self._messages = _prepare_prompt_messages(prompt_messages)
        self._tools = tools

    async def _complete_without_stream(self, kwargs: dict[str, Any]) -> ProviderTurn:
        retry_kwargs = {**kwargs, "stream": False}
        response = await self._client.chat.completions.create(**retry_kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ProviderTurn(assistant_text="", tool_calls=[], assistant_turn=None)
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        text = content if isinstance(content, str) else ""
        return _build_provider_turn(text, _extract_response_tool_calls(message))

    async def stream_turn(self, on_event: EventSink) -> ProviderTurn:
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._messages,
            "stream": True,
            "temperature": 0,
            "max_tokens": 8192,
        }
        if self._tools:
            kwargs["tools"] = self._tools

        try:
            state = ChatCompletionsParseState()
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                await _parse_stream_chunk(chunk, state, on_event)
            return _build_provider_turn(
                state.assistant_text,
                [
                    state.tool_calls_by_index[index]
                    for index in sorted(state.tool_calls_by_index.keys())
                ],
            )
        except httpx.RemoteProtocolError as exc:
            print(
                "[gateway] streaming response ended early; retrying without stream",
                exc,
            )
            return await self._complete_without_stream(kwargs)

    async def append_tool_results(
        self, turn: ProviderTurn, executed_tool_calls: List[ExecutedToolCall]
    ) -> None:
        if turn.assistant_turn:
            self._messages.append(cast(ChatCompletionMessageParam, turn.assistant_turn))

        image_parts: List[Dict[str, Any]] = []
        for executed in executed_tool_calls:
            self._messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": "tool",
                        "tool_call_id": executed.tool_call.id,
                        "content": json.dumps(executed.result.result),
                    },
                )
            )
            if not executed.result.ok:
                continue
            for part in executed.result.multimodal_parts or []:
                image_url = _tool_image_ref(part)
                if image_url is None:
                    continue
                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    }
                )

        if image_parts:
            self._messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Tool output images from the previous calls.",
                            },
                            *image_parts,
                        ],
                    },
                )
            )

    async def close(self) -> None:
        return None
