# tests/unit/test_analyser.py
"""
Tests for the pure helper functions inside app.services.analyser.

_parse_json() is a great unit test target: it's deterministic, has clear
error conditions, and is called by every LLM response path.
"""

import pytest
from app.services.analyser import _parse_json


class TestParseJson:
    def test_plain_json_parsed(self):
        raw = '{"key": "value"}'
        result = _parse_json(raw, "test")
        assert result == {"key": "value"}

    def test_strips_triple_backtick_fence(self):
        raw = "```\n{\"key\": \"value\"}\n```"
        result = _parse_json(raw, "test")
        assert result["key"] == "value"

    def test_strips_json_language_tag(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = _parse_json(raw, "test")
        assert result["key"] == "value"

    def test_raises_value_error_on_invalid_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_json("this is not json", "Pass 1")

    def test_error_message_includes_label(self):
        """The label helps you identify which LLM pass failed."""
        with pytest.raises(ValueError, match="Pass 2"):
            _parse_json("{bad json", "Pass 2")

    def test_nested_json_preserved(self):
        raw = '{"skills": ["python", "sql"], "score": 0.8}'
        result = _parse_json(raw, "test")
        assert result["skills"] == ["python", "sql"]
        assert result["score"] == pytest.approx(0.8)

    def test_whitespace_around_json_is_ok(self):
        raw = '   {"key": "value"}   '
        result = _parse_json(raw, "test")
        assert result["key"] == "value"
