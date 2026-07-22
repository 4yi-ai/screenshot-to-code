import importlib, os
from llm import Llm, OPENAI_MODELS, ANTHROPIC_MODELS, GEMINI_MODELS


def test_gateway_enum_exists_and_is_unclassified():
    assert Llm.GATEWAY.value == "gateway"
    # The sentinel must NOT be in any native-provider set (so the factory's
    # native branches skip it and the gateway branch handles it).
    assert Llm.GATEWAY not in OPENAI_MODELS
    assert Llm.GATEWAY not in ANTHROPIC_MODELS
    assert Llm.GATEWAY not in GEMINI_MODELS


def test_text_model_read_from_env(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL", "anthropic.claude-sonnet-4-6")
    import config
    importlib.reload(config)
    assert config.TEXT_MODEL == "anthropic.claude-sonnet-4-6"
