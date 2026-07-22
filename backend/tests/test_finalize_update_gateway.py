from typing import Any, Dict, Optional

import pytest

from agent.engine import AgentEngine


def _make_engine(
    *,
    stream_text_as_code: bool,
    initial_file_state: Optional[Dict[str, str]] = None,
) -> AgentEngine:
    async def send_message(
        msg_type: str,
        value: Optional[str],
        variant_index: int,
        data: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> None:
        return None

    return AgentEngine(
        send_message=send_message,
        variant_index=0,
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        replicate_api_key=None,
        should_generate_images=False,
        initial_file_state=initial_file_state,
        stream_text_as_code=stream_text_as_code,
    )


NEW_HTML = "<!DOCTYPE html><html><body><h1>New content</h1></body></html>"


@pytest.mark.asyncio
async def test_tools_less_update_returns_freshly_generated_html_not_stale_seed() -> None:
    """Regression test for the Critical bug: on the gateway (tools-less) path,
    _finalize_response used to short-circuit on pre-seeded (old) file_state.content
    for UPDATE generations, discarding the newly generated HTML.
    """
    engine = _make_engine(
        stream_text_as_code=True,
        initial_file_state={"path": "index.html", "content": "<old>"},
    )
    assert engine.file_state.content == "<old>"

    result = await engine._finalize_response(NEW_HTML)

    assert result == NEW_HTML
    assert engine.file_state.content == NEW_HTML


@pytest.mark.asyncio
async def test_tools_less_create_mode_unaffected() -> None:
    """Create-mode (no seeded file_state) on the tools-less path should still
    return the freshly generated HTML."""
    engine = _make_engine(stream_text_as_code=True, initial_file_state=None)
    assert engine.file_state.content == ""

    result = await engine._finalize_response(NEW_HTML)

    assert result == NEW_HTML
    assert engine.file_state.content == NEW_HTML


@pytest.mark.asyncio
async def test_tools_less_path_falls_back_to_seed_when_model_returns_nothing() -> None:
    """If the model produced no output at all on the tools-less path
    (extract_html_content -> ""), fall back to whatever is already in
    file_state.content rather than clobbering it with empty."""
    engine = _make_engine(
        stream_text_as_code=True,
        initial_file_state={"path": "index.html", "content": "<old>"},
    )

    result = await engine._finalize_response("")

    assert result == "<old>"
    assert engine.file_state.content == "<old>"


@pytest.mark.asyncio
async def test_non_tools_less_path_behavior_unchanged_prefers_seeded_content() -> None:
    """The non-tools-less (tool-calling) path must keep its exact current
    behavior: if file_state.content is already seeded, it wins regardless of
    assistant_text."""
    engine = _make_engine(
        stream_text_as_code=False,
        initial_file_state={"path": "index.html", "content": "<old>"},
    )

    result = await engine._finalize_response(NEW_HTML)

    assert result == "<old>"
    assert engine.file_state.content == "<old>"


@pytest.mark.asyncio
async def test_non_tools_less_path_create_mode_extracts_html() -> None:
    """The non-tools-less path with no seeded content should still extract
    HTML from assistant_text (existing behavior)."""
    engine = _make_engine(stream_text_as_code=False, initial_file_state=None)

    result = await engine._finalize_response(NEW_HTML)

    assert result == NEW_HTML
    assert engine.file_state.content == NEW_HTML
