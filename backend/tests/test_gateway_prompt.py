from prompts.system_prompt import SYSTEM_PROMPT


def test_prompt_instructs_raw_html_and_drops_tool_binding():
    assert "<!DOCTYPE html>" in SYSTEM_PROMPT
    assert "create_file" not in SYSTEM_PROMPT
    assert "Do not output raw HTML" not in SYSTEM_PROMPT
    # stack instructions preserved
    assert "Tailwind" in SYSTEM_PROMPT
