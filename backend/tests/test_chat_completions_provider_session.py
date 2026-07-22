# pyright: reportUnknownVariableType=false
import base64
import io
from typing import Any, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from agent.providers.chat_completions import ChatCompletionsProviderSession


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
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content, empty=False):
        self.choices = [] if empty else [_Choice(content)]


class _Message:
    def __init__(self, content):
        self.content = content


class _ResponseChoice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_ResponseChoice(content)]


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
        fail_stream_once: bool = False,
    ):
        self._chunks = chunks
        self._fallback_content = fallback_content
        self._fail_stream_once = fail_stream_once
        self.kwargs: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        if kwargs.get("stream") is False:
            return _Response(self._fallback_content)
        fail = self._fail_stream_once
        self._fail_stream_once = False
        return _Stream(self._chunks, fail=fail)


class _Chat:
    def __init__(
        self,
        chunks,
        *,
        fallback_content: str = "",
        fail_stream_once: bool = False,
    ):
        self.completions = _Completions(
            chunks,
            fallback_content=fallback_content,
            fail_stream_once=fail_stream_once,
        )


class _Client:
    def __init__(
        self,
        chunks,
        *,
        fallback_content: str = "",
        fail_stream_once: bool = False,
    ):
        self.chat = _Chat(
            chunks,
            fallback_content=fallback_content,
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


async def _noop():
    return None
