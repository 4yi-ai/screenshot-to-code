from routes.generate_code import ModelSelectionStage
from llm import Llm


async def _throw(msg):
    raise AssertionError(f"should not error: {msg}")


async def test_selects_only_gateway_regardless_of_keys():
    stage = ModelSelectionStage(throw_error=_throw)
    models = await stage.select_models(
        generation_type="create", input_mode="image",
        openai_api_key=None, anthropic_api_key=None, gemini_api_key=None,
    )
    assert models == [Llm.GATEWAY] * len(models)
    assert len(models) >= 1
    assert set(models) == {Llm.GATEWAY}
