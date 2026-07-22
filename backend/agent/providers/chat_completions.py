from typing import Any, List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)


class ChatCompletionsProviderSession(ProviderSession):
    """OpenAI-compatible chat/completions session used to route generation through
    the 4yi gateway. v1 is tools-less: a single streaming vision call that returns
    raw HTML (recovered downstream by extract_html_content).
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
        self._messages = prompt_messages
        self._tools = tools

    async def stream_turn(self, on_event: EventSink) -> ProviderTurn:
        kwargs: dict = {
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
