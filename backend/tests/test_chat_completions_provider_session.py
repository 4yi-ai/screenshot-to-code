# pyright: reportUnknownVariableType=false
import base64
import io
from typing import Any, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from agent.providers.base import ExecutedToolCall
from agent.providers.chat_completions import ChatCompletionsProviderSession
from agent.tools.types import ToolCall, ToolExecutionResult, ToolMultimodalPart


def _png_data_url(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _data_url_dimensions(data_url: str) -> tuple[int, int]:
    _, encoded = data_url.split(",", 1)
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    return image.size


class _Delta:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, content, tool_calls=None):
        self.delta = _Delta(content, tool_calls)


class _Chunk:
    def __init__(self, content, empty=False, tool_calls=None):
        self.choices = [] if empty else [_Choice(content, tool_calls)]


class _Message:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _ResponseChoice:
    def __init__(self, content, tool_calls=None):
        self.message = _Message(content, tool_calls)


class _Response:
    def __init__(self, content, tool_calls=None):
        self.choices = [_ResponseChoice(content, tool_calls)]


class _Stream:
    def __init__(self, chunks, *, fail: bool = False):
        self._it = iter(chunks)
        self._fail = fail

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._fail:
            self._fail = False
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Completions:
    def __init__(
        self,
        chunks,
        *,
        fallback_content: str = "",
        fallback_tool_calls=None,
        fail_stream_once: bool = False,
    ):
        self._chunks = chunks
        self._fallback_content = fallback_content
        self._fallback_tool_calls = fallback_tool_calls
        self._fail_stream_once = fail_stream_once
        self.kwargs: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        if kwargs.get("stream") is False:
            return _Response(self._fallback_content, self._fallback_tool_calls)
        fail = self._fail_stream_once
        self._fail_stream_once = False
        return _Stream(self._chunks, fail=fail)


class _Chat:
    def __init__(
        self,
        chunks,
        *,
        fallback_content: str = "",
        fallback_tool_calls=None,
        fail_stream_once: bool = False,
    ):
        self.completions = _Completions(
            chunks,
            fallback_content=fallback_content,
            fallback_tool_calls=fallback_tool_calls,
            fail_stream_once=fail_stream_once,
        )


class _Client:
    def __init__(
        self,
        chunks,
        *,
        fallback_content: str = "",
        fallback_tool_calls=None,
        fail_stream_once: bool = False,
    ):
        self.chat = _Chat(
            chunks,
            fallback_content=fallback_content,
            fallback_tool_calls=fallback_tool_calls,
            fail_stream_once=fail_stream_once,
        )


def _captured_kwargs(client: _Client) -> dict[str, Any]:
    assert client.chat.completions.kwargs is not None
    return client.chat.completions.kwargs


async def test_streams_assistant_text_and_returns_turn():
    chunks = [_Chunk("<!DOCTYPE "), _Chunk("html>"), _Chunk("", empty=True)]
    client = _Client(chunks)
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, client),
        model_name="anthropic.claude-sonnet-4-6",
        prompt_messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        tools=[],
    )
    events = []

    async def sink(e):
        events.append(e)

    turn = await session.stream_turn(sink)

    assert turn.assistant_text == "<!DOCTYPE html>"
    assert turn.tool_calls == []
    assert [e.text for e in events if e.type == "assistant_delta"] == [
        "<!DOCTYPE ",
        "html>",
    ]
    kwargs = _captured_kwargs(client)
    assert kwargs["model"] == "anthropic.claude-sonnet-4-6"
    assert kwargs["stream"] is True
    # tools-less: no tools sent
    assert "tools" not in kwargs


async def test_empty_choices_chunk_is_ignored():
    client = _Client([_Chunk("", empty=True), _Chunk("hi")])
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, client),
        model_name="m",
        prompt_messages=[{"role": "user", "content": "u"}],
        tools=[],
    )
    events = []
    turn = await session.stream_turn(lambda e: events.append(e) or _noop())
    assert turn.assistant_text == "hi"


async def test_resizes_large_data_url_images_for_gateway():
    image_url = _png_data_url(width=1, height=8001)
    prompt_messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Build from this screenshot."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            }
        ],
    )
    client = _Client([_Chunk("ok")])
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, client),
        model_name="m",
        prompt_messages=prompt_messages,
        tools=[],
    )

    await session.stream_turn(lambda e: _noop())

    messages = cast(list[dict[str, Any]], _captured_kwargs(client)["messages"])
    sent_image_url = messages[0]["content"][1]["image_url"]["url"]

    assert sent_image_url.startswith("data:image/jpeg;base64,")
    assert max(_data_url_dimensions(sent_image_url)) < 8000
    original_messages = cast(list[dict[str, Any]], prompt_messages)
    assert original_messages[0]["content"][1]["image_url"]["url"] == image_url


async def test_keeps_remote_image_urls_unchanged_for_gateway():
    remote_url = "https://example.com/screenshot.png"
    client = _Client([_Chunk("ok")])
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, client),
        model_name="m",
        prompt_messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": remote_url, "detail": "high"},
                    }
                ],
            }
        ],
        tools=[],
    )

    await session.stream_turn(lambda e: _noop())

    messages = cast(list[dict[str, Any]], _captured_kwargs(client)["messages"])
    sent_image_url = messages[0]["content"][0]["image_url"]["url"]
    assert sent_image_url == remote_url


async def test_retries_non_streaming_when_gateway_stream_closes_early():
    fallback_html = "<!DOCTYPE html><html><body><h1>Updated</h1></body></html>"
    client = _Client(
        [_Chunk("partial")],
        fallback_content=fallback_html,
        fail_stream_once=True,
    )
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, client),
        model_name="m",
        prompt_messages=[{"role": "user", "content": "u"}],
        tools=[],
    )
    events = []

    turn = await session.stream_turn(lambda e: events.append(e) or _noop())

    assert turn.assistant_text == fallback_html
    assert events == []
    assert [call["stream"] for call in client.chat.completions.calls] == [True, False]


async def test_streams_tool_calls_and_appends_tool_results():
    chunks = [
        _Chunk(
            "",
            tool_calls=[
                {
                    "index": 0,
                    "id": "call-1",
                    "function": {
                        "name": "create_file",
                        "arguments": '{"path":"index.html","content":"<!DOCTYPE ',
                    },
                }
            ],
        ),
        _Chunk(
            "",
            tool_calls=[
                {
                    "index": 0,
                    "function": {
                        "arguments": 'html><html></html>"}',
                    },
                }
            ],
        ),
    ]
    client = _Client(chunks)
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, client),
        model_name="m",
        prompt_messages=[{"role": "user", "content": "u"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "create_file",
                    "description": "Create a file.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    events = []

    turn = await session.stream_turn(lambda e: events.append(e) or _noop())

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0] == ToolCall(
        id="call-1",
        name="create_file",
        arguments={
            "path": "index.html",
            "content": "<!DOCTYPE html><html></html>",
        },
    )
    assert [
        e.tool_arguments for e in events if e.type == "tool_call_delta"
    ] == [
        '{"path":"index.html","content":"<!DOCTYPE ',
        '{"path":"index.html","content":"<!DOCTYPE html><html></html>"}',
    ]

    await session.append_tool_results(
        turn,
        [
            ExecutedToolCall(
                tool_call=turn.tool_calls[0],
                result=ToolExecutionResult(
                    ok=True,
                    result={"content": "created"},
                    summary={"content": "created"},
                ),
            )
        ],
    )

    messages = cast(list[dict[str, Any]], session._messages)
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["id"] == "call-1"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"content": "created"}',
    }


async def test_non_streaming_retry_preserves_tool_calls():
    client = _Client(
        [_Chunk("partial")],
        fallback_tool_calls=[
            {
                "id": "call-2",
                "function": {
                    "name": "edit_file",
                    "arguments": '{"old_text":"a","new_text":"b"}',
                },
            }
        ],
        fail_stream_once=True,
    )
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, client),
        model_name="m",
        prompt_messages=[{"role": "user", "content": "u"}],
        tools=[],
    )

    turn = await session.stream_turn(lambda e: _noop())

    assert turn.tool_calls == [
        ToolCall(
            id="call-2",
            name="edit_file",
            arguments={"old_text": "a", "new_text": "b"},
        )
    ]
    assert [call["stream"] for call in client.chat.completions.calls] == [True, False]


async def test_append_tool_results_adds_multimodal_tool_images():
    session = ChatCompletionsProviderSession(
        client=cast(AsyncOpenAI, _Client([])),
        model_name="m",
        prompt_messages=[{"role": "user", "content": "u"}],
        tools=[],
    )
    turn = await session.stream_turn(lambda e: _noop())
    tool_call = ToolCall(id="call-3", name="screenshot_preview", arguments={})
    turn.tool_calls.append(tool_call)
    turn.assistant_turn = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-3",
                "type": "function",
                "function": {"name": "screenshot_preview", "arguments": "{}"},
            }
        ],
    }

    await session.append_tool_results(
        turn,
        [
            ExecutedToolCall(
                tool_call=tool_call,
                result=ToolExecutionResult(
                    ok=True,
                    result={"content": "rendered"},
                    summary={"content": "rendered"},
                    multimodal_parts=[
                        ToolMultimodalPart(
                            display_name="desktop.png",
                            mime_type="image/png",
                            image_url="https://example.com/desktop.png",
                        )
                    ],
                ),
            )
        ],
    )

    messages = cast(list[dict[str, Any]], session._messages)
    assert messages[-1]["role"] == "user"
    content = messages[-1]["content"]
    assert content[1]["image_url"]["url"] == "https://example.com/desktop.png"


async def _noop():
    return None
