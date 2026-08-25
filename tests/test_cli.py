import io
import json

from llm_cost.cli import main


def test_estimate_prints_table_and_exits_zero(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "12000", "--output", "800"])
    out = capsys.readouterr().out
    assert code == 0
    assert "claude-opus-5" in out
    assert "cost per call: $0.0800" in out


def test_estimate_json_output_is_valid_json(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "0", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["model"] == "claude-opus-5"


def test_estimate_unknown_model_exits_3(capsys):
    code = main(["estimate", "--model", "not-a-real-model", "--input", "1", "--output", "1"])
    err = capsys.readouterr().err
    assert code == 3
    assert "not-a-real-model" in err


def test_estimate_total_tokens_includes_cached_and_cache_write(capsys):
    code = main([
        "estimate", "--model", "claude-opus-5",
        "--input", "12000", "--output", "800",
        "--cached", "52000", "--cache-write", "9000",
    ])
    out = capsys.readouterr().out
    assert code == 0
    total_line = next(line for line in out.splitlines() if line.startswith("total"))
    assert "73,800" in total_line


def test_report_missing_file_exits_2(capsys):
    code = main(["report", "/tmp/does-not-exist-llm-cost.jsonl"])
    err = capsys.readouterr().err
    assert code == 2
    assert "cannot read" in err


def test_report_json_over_a_written_log(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        '{"model":"claude-opus-5","usage":{"input_tokens":1000,"output_tokens":100}}\n'
        '{"model":"claude-opus-5","usage":{"input_tokens":2000,"output_tokens":200}}\n'
    )
    code = main(["report", str(log), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["total_calls"] == 2
    assert data["rows"][0]["key"] == "claude-opus-5"


def test_report_reads_stdin_when_path_omitted(monkeypatch, capsys):
    log = (
        '{"model":"claude-opus-5","usage":{"input_tokens":1000,"output_tokens":100}}\n'
        '{"model":"claude-opus-5","usage":{"input_tokens":2000,"output_tokens":200}}\n'
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(log))
    code = main(["report", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["total_calls"] == 2


def test_report_reads_stdin_when_path_is_dash(monkeypatch, capsys):
    log = '{"model":"claude-opus-5","usage":{"input_tokens":1000,"output_tokens":100}}\n'
    monkeypatch.setattr("sys.stdin", io.StringIO(log))
    code = main(["report", "-", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["total_calls"] == 1


def test_report_since_until_narrows_to_the_date_window(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text(
        '{"model":"claude-opus-5","date":"2026-05-01","usage":{"input_tokens":1000,"output_tokens":100}}\n'
        '{"model":"claude-opus-5","date":"2026-06-15","usage":{"input_tokens":2000,"output_tokens":200}}\n'
    )
    code = main(["report", str(log), "--since", "2026-06-01", "--until", "2026-06-30", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["total_calls"] == 1
    assert data["rows"][0]["input_tokens"] == 2000


def test_report_bad_since_exits_2(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text('{"model":"claude-opus-5","usage":{"input_tokens":1000,"output_tokens":100}}\n')
    code = main(["report", str(log), "--since", "not-a-date"])
    err = capsys.readouterr().err
    assert code == 2


def test_report_pricing_override_flag_works_before_and_after_subcommand(tmp_path, capsys):
    log = tmp_path / "usage.jsonl"
    log.write_text('{"model":"internal-router-v3","usage":{"input_tokens":1000,"output_tokens":100}}\n')
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"models": {"internal-router-v3": {"input": 1.0, "output": 1.0}}}))

    before = main(["--pricing", str(prices), "report", str(log), "--json"])
    out_before = json.loads(capsys.readouterr().out)

    after = main(["report", str(log), "--pricing", str(prices), "--json"])
    out_after = json.loads(capsys.readouterr().out)

    assert before == 0
    assert after == 0
    assert out_before["rows"][0]["key"] == "internal-router-v3"
    assert out_after == out_before


def test_compare_orders_cheapest_first(capsys):
    code = main(["compare", "--input", "10000", "--output", "1000", "--provider", "anthropic", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    rows = json.loads(out)
    costs = [row["cost"] for row in rows]
    assert costs == sorted(costs)


def test_estimate_with_non_usd_pricing_uses_symbol_and_precision(tmp_path, capsys):
    prices = tmp_path / "eur.json"
    prices.write_text(json.dumps({
        "currency": "EUR",
        "precision": 2,
        "replace": True,
        "models": {"claude-opus-5": {"input": 4.0, "output": 20.0}},
    }))
    code = main(["--pricing", str(prices), "estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "€0.00" in out
    assert "cost per call: €0.00" in out


def test_estimate_json_includes_currency(tmp_path, capsys):
    prices = tmp_path / "jpy.json"
    prices.write_text(json.dumps({"currency": "JPY", "precision": 0}))
    code = main(["--pricing", str(prices), "estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "0", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert data["currency"] == "JPY"


def test_models_lists_the_built_in_table(capsys):
    code = main(["models"])
    out = capsys.readouterr().out
    assert code == 0
    assert "13 models" in out
    assert "claude-opus-5" in out
