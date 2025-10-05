import pytest

from core.utils import extract_eot_from_chat_template


@pytest.mark.parametrize(
    "template, expected_contains",
    [
        # XML-like marker
        ("User: {{ message.content }}\n</think>\n{{ assistant }}", "</think>"),
        # im_start style with trailing newline
        ("User: {{ message.content }}<|im_start|>assistant\n{{ assistant }}", "<|im_start|>assistant"),
        # paired start/end header-style markers
        ("User: {{ message.content }}<|start_header_id|>assistant<|end_header_id|>{{ assistant }}", "<|start_header_id|>assistant<|end_header_id|>"),
        # marker at end of template (no trailing {{)
        ("User: {{ message.content }}<|im_start|>assistant\n", "<|im_start|>assistant"),
    ],
)
def test_extract_eot_various_markers(template, expected_contains):
    marker = extract_eot_from_chat_template(template)
    assert marker is not None, f"Expected to find marker in template: {template!r}"
    assert expected_contains in marker


def test_extract_eot_none_when_no_marker():
    tmpl = "User: {{ message.content }} {{ some_other }}"
    assert extract_eot_from_chat_template(tmpl) is None
