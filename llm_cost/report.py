"""Aggregate priced usage records, rank models against each other, and
render the aligned tables both the estimate and report commands print."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as _date

from .cost import estimate_cost
from .pricing import UnknownModelError


@dataclass
class ReportRow:
    key: str
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0


@dataclass
class Report:
    group_by: str
    rows: list
    total_calls: int
    total_input_tokens: int
    total_cached_input_tokens: int
    total_output_tokens: int
    total_cache_write_tokens: int
    total_cost: float
    unknown_models: set = field(default_factory=set)
    skipped_calls: int = 0


def _group_key(record, group_by: str) -> str:
    if group_by == "model":
        return record.model
    if group_by == "date":
        date = record.fields.get("date", "")
        return date[:10] if date else "unknown"
    value = record.fields.get(group_by)
    return str(value) if value is not None else "unknown"


def filter_by_date_range(records, *, since: str = None, until: str = None):
    """Keep only records whose 'date' field falls within [since, until]
    (inclusive, YYYY-MM-DD). Records with no date are dropped whenever a
    bound is given, since they cannot be placed inside or outside the range.
    `since` and `until` are validated as ISO dates so a typo fails loudly
    instead of silently matching nothing."""
    if since is None and until is None:
        return records

    if since is not None:
        _date.fromisoformat(since)
    if until is not None:
        _date.fromisoformat(until)

    kept = []
    for record in records:
        day = (record.fields.get("date") or "")[:10]
        if not day:
            continue
        if since is not None and day < since:
            continue
        if until is not None and day > until:
            continue
        kept.append(record)
    return kept


def build_report(records, table, *, group_by: str = "model") -> Report:
    groups = {}
    unknown_models = set()
    skipped_calls = 0

    for record in records:
        try:
            price = table.resolve(record.model)
        except UnknownModelError:
            unknown_models.add(record.model)
            skipped_calls += 1
            continue

        result = estimate_cost(
            price,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens,
        )

        key = _group_key(record, group_by)
        row = groups.setdefault(key, ReportRow(key))
        row.calls += 1
        row.input_tokens += result.input_tokens
        row.cached_input_tokens += result.cached_input_tokens
        row.output_tokens += result.output_tokens
        row.cache_write_tokens += result.cache_write_tokens
        row.cost += result.total_cost

    rows = sorted(groups.values(), key=lambda row: row.cost, reverse=True)

    return Report(
        group_by=group_by,
        rows=rows,
        total_calls=sum(row.calls for row in rows),
        total_input_tokens=sum(row.input_tokens for row in rows),
        total_cached_input_tokens=sum(row.cached_input_tokens for row in rows),
        total_output_tokens=sum(row.output_tokens for row in rows),
        total_cache_write_tokens=sum(row.cache_write_tokens for row in rows),
        total_cost=sum(row.cost for row in rows),
        unknown_models=unknown_models,
        skipped_calls=skipped_calls,
    )


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    provider: str
    input_price: float
    output_price: float
    cost: float
    vs_cheapest: float


def compare_models(table, input_tokens: int, output_tokens: int, *, provider=None, models=None, calls: int = 1):
    names = models if models else sorted(table.models)

    priced = []
    for name in names:
        price = table.resolve(name)
        if provider and price.provider != provider:
            continue
        result = estimate_cost(price, input_tokens=input_tokens, output_tokens=output_tokens, calls=calls)
        priced.append((price, result.total_cost))

    priced.sort(key=lambda item: item[1])
    if not priced:
        return []

    cheapest = priced[0][1]
    return [
        ComparisonRow(
            model=price.name,
            provider=price.provider,
            input_price=price.input,
            output_price=price.output,
            cost=cost,
            vs_cheapest=(cost / cheapest) if cheapest > 0 else 1.0,
        )
        for price, cost in priced
    ]


_CURRENCY_PREFIX_RE = re.compile(r"^([$€£¥]|[A-Z]{3} )")


def _is_numeric_cell(cell: str) -> bool:
    stripped = _CURRENCY_PREFIX_RE.sub("", cell.replace(",", ""))
    if stripped.endswith("x"):
        stripped = stripped[:-1]
    if stripped in ("", "-"):
        return True
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def render_table(headers, rows) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    aligns = []
    for i in range(len(headers)):
        values = [row[i] for row in str_rows if i < len(row) and row[i] != ""]
        aligns.append("r" if values and all(_is_numeric_cell(v) for v in values) else "l")

    def format_row(cells):
        parts = [
            cell.rjust(width) if align == "r" else cell.ljust(width)
            for cell, width, align in zip(cells, widths, aligns)
        ]
        return "  ".join(parts).rstrip()

    lines = [format_row([str(header) for header in headers])]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(format_row(row) for row in str_rows)
    return "\n".join(lines)
