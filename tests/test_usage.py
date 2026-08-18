import pytest

from llm_cost import UsageFormatError, load_usage


ANTHROPIC_LINE = (
    '{"model":"claude-opus-5","date":"2026-06-01T09:12:00Z","team":"agents",'
    '"usage":{"input_tokens":18400,"output_tokens":2100,'
    '"cache_read_input_tokens":52000,"cache_creation_input_tokens":9000}}'
)

OPENAI_LINE = (
    '{"model":"gpt-4o-mini","date":"2026-06-01T10:03:00Z","team":"search",'
    '"usage":{"prompt_tokens":31000,"completion_tokens":420,'
    '"prompt_tokens_details":{"cached_tokens":24000}}}'
)


def test_parses_anthropic_shaped_usage():
    records, problems = load_usage(ANTHROPIC_LINE)
    assert not problems
    assert len(records) == 1
    record = records[0]
    assert record.model == "claude-opus-5"
    assert record.input_tokens == 18400
    assert record.output_tokens == 2100
    assert record.cached_input_tokens == 52000
    assert record.cache_write_tokens == 9000
    assert record.fields["team"] == "agents"


def test_parses_openai_shaped_usage_and_splits_cached_prefix():
    records, problems = load_usage(OPENAI_LINE)
    assert not problems
    record = records[0]
    assert record.model == "gpt-4o-mini"
    # prompt_tokens includes the cached prefix; input_tokens should not
    assert record.input_tokens == 31000 - 24000
    assert record.cached_input_tokens == 24000
    assert record.output_tokens == 420
    assert record.cache_write_tokens == 0


def test_blank_lines_are_skipped():
    text = f"\n{ANTHROPIC_LINE}\n\n{OPENAI_LINE}\n"
    records, problems = load_usage(text)
    assert len(records) == 2
    assert not problems


def test_malformed_json_becomes_a_problem_not_an_exception():
    records, problems = load_usage("not json at all")
    assert not records
    assert len(problems) == 1
    assert problems[0].line_number == 1
    assert "invalid JSON" in problems[0].reason


def test_missing_model_field_becomes_a_problem():
    records, problems = load_usage('{"usage": {"input_tokens": 1, "output_tokens": 1}}')
    assert not records
    assert len(problems) == 1
    assert "model" in problems[0].reason


def test_missing_usage_shape_becomes_a_problem():
    records, problems = load_usage('{"model": "claude-opus-5", "usage": {"weird_field": 1}}')
    assert not records
    assert len(problems) == 1


def test_strict_raises_when_problems_exist():
    with pytest.raises(UsageFormatError):
        load_usage("not json at all", strict=True)


def test_strict_does_not_raise_when_clean():
    records, problems = load_usage(ANTHROPIC_LINE, strict=True)
    assert len(records) == 1
    assert not problems
