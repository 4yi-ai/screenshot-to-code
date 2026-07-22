# pyright: reportUnknownVariableType=false
import copy
from typing import Any, List, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.providers.anthropic.image import process_image
from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)


def _process_data_url_image(image_url: str) -> str:
    if not image_url.startswith("data:image/") or ";base64," not in image_url:
        return image_url

    media_type, base64_data = process_image(image_url)
    return f"data:{media_type};base64,{base64_data}"


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


class ChatCompletionsProviderSession(ProviderSession):
    """OpenAI-compatible chat/completions session used to route generation through
    the 4yi gateway. v1 is tools-less: a single streaming vision call that returns
    raw HTML (recovered downstream by extract_html_content).

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

        stream = await self._client.chat.completions.create(**kwargs)

        text = ""
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                text += piece
                await on_event(StreamEvent(type="assistant_delta", text=piece))

        return ProviderTurn(assistant_text=text, tool_calls=[], assistant_turn=None)

    async def append_tool_results(
        self, turn: ProviderTurn, executed_tool_calls: List[ExecutedToolCall]
    ) -> None:
        # v1 is tools-less; there is never a tool continuation.
        return None

    async def close(self) -> None:
        return None
