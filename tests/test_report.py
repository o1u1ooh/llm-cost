import pytest

from llm_cost import build_report, compare_models, default_pricing, load_usage, parse_pricing, render_table


def test_build_report_groups_by_model_and_sums_costs():
    text = "\n".join([
        '{"model":"claude-opus-5","team":"agents","usage":{"input_tokens":1000,"output_tokens":100,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}',
        '{"model":"claude-opus-5","team":"search","usage":{"input_tokens":2000,"output_tokens":200,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}',
        '{"model":"gpt-4o-mini","team":"search","usage":{"prompt_tokens":5000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":0}}}',
    ])
    records, problems = load_usage(text)
    assert not problems

    report = build_report(records, default_pricing(), group_by="model")
    assert report.total_calls == 3

    by_key = {row.key: row for row in report.rows}
    assert by_key["claude-opus-5"].calls == 2
    assert by_key["claude-opus-5"].input_tokens == 3000
    assert by_key["gpt-4o-mini"].calls == 1
    assert report.total_cost == pytest.approx(sum(row.cost for row in report.rows))


def test_build_report_tracks_unknown_models_without_pricing_them_as_zero():
    text = '{"model":"internal-router-v3","usage":{"input_tokens":100,"output_tokens":10}}'
    records, problems = load_usage(text)
    assert not problems

    report = build_report(records, default_pricing(), group_by="model")
    assert report.rows == []
    assert report.unknown_models == {"internal-router-v3"}
    assert report.skipped_calls == 1


def test_build_report_can_group_by_arbitrary_field():
    text = "\n".join([
        '{"model":"claude-opus-5","team":"agents","usage":{"input_tokens":1000,"output_tokens":100}}',
        '{"model":"gpt-4o-mini","team":"agents","usage":{"prompt_tokens":500,"completion_tokens":50}}',
    ])
    records, problems = load_usage(text)
    assert not problems

    report = build_report(records, default_pricing(), group_by="team")
    assert len(report.rows) == 1
    assert report.rows[0].key == "agents"
    assert report.rows[0].calls == 2


def test_compare_models_orders_cheapest_first_and_computes_ratio():
    table = parse_pricing({
        "replace": True,
        "models": {
            "cheap": {"provider": "x", "input": 1.0, "output": 1.0},
            "pricey": {"provider": "x", "input": 3.0, "output": 3.0},
        },
    })
    rows = compare_models(table, input_tokens=1_000_000, output_tokens=0)
    assert [row.model for row in rows] == ["cheap", "pricey"]
    assert rows[0].vs_cheapest == pytest.approx(1.0)
    assert rows[1].vs_cheapest == pytest.approx(3.0)


def test_compare_models_filters_by_provider():
    table = default_pricing()
    rows = compare_models(table, input_tokens=10_000, output_tokens=1_000, provider="anthropic")
    assert rows
    assert all(row.provider == "anthropic" for row in rows)


def test_render_table_right_aligns_numeric_columns():
    text = render_table(["item", "tokens"], [["input", "12,000"], ["total", "800"]])
    lines = text.splitlines()
    header, separator, *body = lines
    assert header.startswith("item")
    # numeric column is right-aligned, so shorter numbers get left padding
    assert body[1].endswith("800")
    assert len(body[0]) == len(body[1])
