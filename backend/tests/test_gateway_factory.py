from agent.providers.chat_completions import ChatCompletionsProviderSession
from agent.providers.factory import create_provider_session
from llm import Llm


def test_gateway_model_returns_chat_completions_session(monkeypatch):
    import config
    monkeypatch.setattr(config, "TEXT_MODEL", "anthropic.claude-sonnet-4-6", raising=False)
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "https://gw.test/api/v1", raising=False)
    # Also patch the names already imported into factory's namespace.
    import agent.providers.factory as factory
    monkeypatch.setattr(factory, "TEXT_MODEL", "anthropic.claude-sonnet-4-6", raising=False)
    monkeypatch.setattr(factory, "OPENAI_BASE_URL", "https://gw.test/api/v1", raising=False)

    session = create_provider_session(
        model=Llm.GATEWAY,
        prompt_messages=[{"role": "user", "content": "u"}],
        should_generate_images=True,
        openai_api_key="xck-test",
        openai_base_url=None,      # simulate IS_PROD suppressing the per-request base url
        anthropic_api_key=None,
        gemini_api_key=None,
        replicate_api_key=None,
    )
    assert isinstance(session, ChatCompletionsProviderSession)
    assert session._model_name == "anthropic.claude-sonnet-4-6"
    assert session._tools == []
    # base_url falls back to config even when the per-request one is None
    assert str(session._client.base_url).startswith("https://gw.test/api/v1")


def test_gateway_raises_when_base_url_missing(monkeypatch):
    import pytest

    import agent.providers.factory as factory
    monkeypatch.setattr(factory, "TEXT_MODEL", "anthropic.claude-sonnet-4-6", raising=False)
    monkeypatch.setattr(factory, "OPENAI_BASE_URL", "", raising=False)  # not injected

    # With no config base_url and the per-request one suppressed (IS_PROD), the
    # gateway branch must fail fast rather than leak the token to api.openai.com.
    with pytest.raises(Exception, match="base URL"):
        create_provider_session(
            model=Llm.GATEWAY,
            prompt_messages=[{"role": "user", "content": "u"}],
            should_generate_images=True,
            openai_api_key="xck-test",
            openai_base_url=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            replicate_api_key=None,
        )
