from agent.providers.chat_completions import ChatCompletionsProviderSession


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content, empty=False):
        self.choices = [] if empty else [_Choice(content)]


class _Stream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Completions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _Stream(self._chunks)


class _Chat:
    def __init__(self, chunks):
        self.completions = _Completions(chunks)


class _Client:
    def __init__(self, chunks):
        self.chat = _Chat(chunks)


async def test_streams_assistant_text_and_returns_turn():
    chunks = [_Chunk("<!DOCTYPE "), _Chunk("html>"), _Chunk("", empty=True)]
    client = _Client(chunks)
    session = ChatCompletionsProviderSession(
        client=client,
        model_name="anthropic.claude-sonnet-4-6",
        prompt_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        tools=[],
    )
    events = []

    async def sink(e):
        events.append(e)

    turn = await session.stream_turn(sink)

    assert turn.assistant_text == "<!DOCTYPE html>"
    assert turn.tool_calls == []
    assert [e.text for e in events if e.type == "assistant_delta"] == ["<!DOCTYPE ", "html>"]
    assert client.chat.completions.kwargs["model"] == "anthropic.claude-sonnet-4-6"
    assert client.chat.completions.kwargs["stream"] is True
    # tools-less: no tools sent
    assert "tools" not in client.chat.completions.kwargs


async def test_empty_choices_chunk_is_ignored():
    client = _Client([_Chunk("", empty=True), _Chunk("hi")])
    session = ChatCompletionsProviderSession(
        client=client, model_name="m",
        prompt_messages=[{"role": "user", "content": "u"}], tools=[])
    events = []
    turn = await session.stream_turn(lambda e: events.append(e) or _noop())
    assert turn.assistant_text == "hi"


async def _noop():
    return None
