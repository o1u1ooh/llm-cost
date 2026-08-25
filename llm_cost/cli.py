"""The `llm-cost` command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .pricing import PricingError, UnknownModelError, default_pricing, load_pricing
from .cost import estimate_cost
from .usage import UsageFormatError, load_usage
from .report import build_report, compare_models, filter_by_date_range, render_table


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_price(p: float) -> str:
    return f"{p:.2f}"


def _fmt_money(x: float, table) -> str:
    return f"{table.symbol}{x:.{table.precision}f}"


def _plural(n: int) -> str:
    return "" if n == 1 else "s"


def _build_parser() -> argparse.ArgumentParser:
    global_opts = argparse.ArgumentParser(add_help=False)
    global_opts.add_argument("--pricing", metavar="PATH", help="JSON file overriding or replacing the built-in price table")
    global_opts.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")

    parser = argparse.ArgumentParser(prog="llm-cost", parents=[global_opts])
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", parents=[global_opts], help="cost of one call, or a batch of identical calls")
    estimate.add_argument("--model", required=True)
    estimate.add_argument("--input", type=int, default=0, help="input tokens per call")
    estimate.add_argument("--output", type=int, default=0, help="output tokens per call")
    estimate.add_argument("--cached", type=int, default=0, help="cached input tokens per call")
    estimate.add_argument("--cache-write", type=int, default=0, dest="cache_write", help="cache write tokens per call")
    estimate.add_argument("--calls", type=int, default=1)

    report = subparsers.add_parser("report", parents=[global_opts], help="cost of a JSONL usage log")
    report.add_argument("path", nargs="?", default="-", help="usage log path, or - to read stdin (default)")
    report.add_argument("--group-by", default="model")
    report.add_argument("--strict", action="store_true", help="fail instead of skipping malformed lines")
    report.add_argument("--since", help="only include records with a 'date' on or after this (YYYY-MM-DD)")
    report.add_argument("--until", help="only include records with a 'date' on or before this (YYYY-MM-DD)")

    compare = subparsers.add_parser("compare", parents=[global_opts], help="rank models by cost for a fixed workload")
    compare.add_argument("--input", type=int, required=True)
    compare.add_argument("--output", type=int, required=True)
    compare.add_argument("--calls", type=int, default=1)
    compare.add_argument("--provider")
    compare.add_argument("--models", help="comma-separated shortlist, otherwise every priced model")

    subparsers.add_parser("models", parents=[global_opts], help="list the price table")

    return parser


def _run_estimate(args, table) -> int:
    price = table.resolve(args.model)
    result = estimate_cost(
        price,
        input_tokens=args.input,
        output_tokens=args.output,
        cached_input_tokens=args.cached,
        cache_write_tokens=args.cache_write,
        calls=args.calls,
    )

    if args.json:
        data = result.to_dict()
        data["currency"] = table.currency
        print(json.dumps(data, indent=2))
        return 0

    print(f"{price.name}  ({args.calls} call{_plural(args.calls)}, prices as of {table.as_of})")
    print()
    rows = [
        ["input", _fmt_int(result.input_tokens), _fmt_price(price.input), _fmt_money(result.input_cost, table)],
        ["cached input", _fmt_int(result.cached_input_tokens), _fmt_price(price.cached_input), _fmt_money(result.cached_input_cost, table)],
        ["cache write", _fmt_int(result.cache_write_tokens), _fmt_price(price.cache_write), _fmt_money(result.cache_write_cost, table)],
        ["output", _fmt_int(result.output_tokens), _fmt_price(price.output), _fmt_money(result.output_cost, table)],
        [
            "total",
            _fmt_int(result.input_tokens + result.output_tokens + result.cached_input_tokens + result.cache_write_tokens),
            "",
            _fmt_money(result.total_cost, table),
        ],
    ]
    print(render_table(["item", "tokens", f"{table.currency}/1M", "cost"], rows))
    print()
    print(f"cost per call: {_fmt_money(result.cost_per_call, table)}")
    return 0


def _run_report(args, table) -> int:
    if args.path == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(args.path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise OSError(f"cannot read {args.path}: {exc.strerror}") from exc

    records, problems = load_usage(text, strict=args.strict)
    records = filter_by_date_range(records, since=args.since, until=args.until)
    report = build_report(records, table, group_by=args.group_by)

    if args.json:
        print(json.dumps(
            {
                "group_by": report.group_by,
                "currency": table.currency,
                "rows": [
                    {
                        "key": row.key,
                        "calls": row.calls,
                        "input_tokens": row.input_tokens,
                        "cached_input_tokens": row.cached_input_tokens,
                        "output_tokens": row.output_tokens,
                        "cache_write_tokens": row.cache_write_tokens,
                        "cost": round(row.cost, 6),
                    }
                    for row in report.rows
                ],
                "total_calls": report.total_calls,
                "total_cost": round(report.total_cost, 6),
                "unknown_models": sorted(report.unknown_models),
                "skipped_calls": report.skipped_calls,
                "malformed_lines": [p.line_number for p in problems],
            },
            indent=2,
        ))
        return 0

    headers = [args.group_by, "calls", "input", "cached", "output", "cost", f"{table.symbol}/call"]
    rows = [
        [
            row.key,
            _fmt_int(row.calls),
            _fmt_int(row.input_tokens),
            _fmt_int(row.cached_input_tokens),
            _fmt_int(row.output_tokens),
            _fmt_money(row.cost, table),
            _fmt_money(row.cost / row.calls, table),
        ]
        for row in report.rows
    ]
    rows.append([
        "TOTAL",
        _fmt_int(report.total_calls),
        _fmt_int(report.total_input_tokens),
        _fmt_int(report.total_cached_input_tokens),
        _fmt_int(report.total_output_tokens),
        _fmt_money(report.total_cost, table),
        "",
    ])
    print(render_table(headers, rows))

    if problems:
        print()
        lines = ", ".join(str(p.line_number) for p in problems)
        print(f"skipped {len(problems)} malformed line(s): {lines}")
    if report.unknown_models:
        print()
        names = ", ".join(sorted(report.unknown_models))
        print(f"skipped {report.skipped_calls} record(s) with no price: {names}")
    return 0


def _run_compare(args, table) -> int:
    models = None
    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]

    rows = compare_models(table, args.input, args.output, provider=args.provider, models=models, calls=args.calls)

    if args.json:
        print(json.dumps([
            {
                "model": row.model,
                "provider": row.provider,
                "input_price": row.input_price,
                "output_price": row.output_price,
                "cost": round(row.cost, 6),
                "vs_cheapest": round(row.vs_cheapest, 4),
                "currency": table.currency,
            }
            for row in rows
        ], indent=2))
        return 0

    print(f"{_fmt_int(args.input)} in + {_fmt_int(args.output)} out, {args.calls} call{_plural(args.calls)}, prices as of {table.as_of}")
    print()
    table_rows = [
        [row.model, row.provider, _fmt_price(row.input_price), _fmt_price(row.output_price), _fmt_money(row.cost, table), f"{row.vs_cheapest:.1f}x"]
        for row in rows
    ]
    print(render_table(["model", "provider", f"{table.currency}/1M in", f"{table.currency}/1M out", "cost", "vs cheapest"], table_rows))
    return 0


def _run_models(args, table) -> int:
    names = sorted(table.models)

    if args.json:
        print(json.dumps(
            {
                "currency": table.currency,
                "models": {
                    name: {
                        "provider": table.models[name].provider,
                        "input": table.models[name].input,
                        "output": table.models[name].output,
                        "cached_input": table.models[name].cached_input,
                        "cache_write": table.models[name].cache_write,
                    }
                    for name in names
                },
            },
            indent=2,
        ))
        return 0

    print(f"{len(names)} models, {table.currency} per 1M tokens, as of {table.as_of} (source: {table.source})")
    print()
    rows = [
        [
            name,
            table.models[name].provider,
            _fmt_price(table.models[name].input),
            _fmt_price(table.models[name].output),
            _fmt_price(table.models[name].cached_input),
            _fmt_price(table.models[name].cache_write),
        ]
        for name in names
    ]
    print(render_table(["model", "provider", "input", "output", "cached input", "cache write"], rows))
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        table = load_pricing(args.pricing) if args.pricing else default_pricing()
    except PricingError as exc:
        print(f"llm-cost: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "estimate": _run_estimate,
        "report": _run_report,
        "compare": _run_compare,
        "models": _run_models,
    }

    try:
        return handlers[args.command](args, table)
    except UnknownModelError as exc:
        print(f"llm-cost: {exc}", file=sys.stderr)
        return 3
    except (OSError, UsageFormatError, ValueError) as exc:
        print(f"llm-cost: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
