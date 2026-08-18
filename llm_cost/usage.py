"""Parse OpenAI- and Anthropic-shaped usage logs into a common record.

OpenAI's `prompt_tokens` includes the cached prefix; Anthropic's
`input_tokens` does not. Both are normalised here so that downstream code
only ever deals with input / cached input / cache write / output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


class UsageFormatError(Exception):
    """A single record, or the whole log under --strict, is malformed."""


@dataclass(frozen=True)
class UsageRecord:
    model: str
    fields: dict
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True)
class UsageProblem:
    line_number: int
    reason: str
    raw: str


def _parse_record(data) -> UsageRecord:
    if not isinstance(data, dict):
        raise UsageFormatError("record is not a JSON object")

    model = data.get("model")
    if not model:
        raise UsageFormatError("missing 'model' field")

    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise UsageFormatError("missing 'usage' object")

    if "input_tokens" in usage:
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cached_input_tokens = int(usage.get("cache_read_input_tokens", 0))
        cache_write_tokens = int(usage.get("cache_creation_input_tokens", 0))
    elif "prompt_tokens" in usage:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        cached_input_tokens = int(details.get("cached_tokens", 0) or 0)
        input_tokens = prompt_tokens - cached_input_tokens
        output_tokens = int(usage.get("completion_tokens", 0))
        cache_write_tokens = 0
    else:
        raise UsageFormatError("usage object has neither Anthropic nor OpenAI token fields")

    return UsageRecord(
        model=model,
        fields=data,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def load_usage(text: str, *, strict: bool = False):
    """Parse JSONL usage records. Returns (records, problems); malformed
    lines are collected as problems rather than raised, unless strict."""
    records = []
    problems = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(UsageProblem(line_number, f"invalid JSON: {exc.msg}", raw_line))
            continue

        try:
            records.append(_parse_record(data))
        except UsageFormatError as exc:
            problems.append(UsageProblem(line_number, str(exc), raw_line))

    if strict and problems:
        raise UsageFormatError(f"{len(problems)} malformed record(s), starting at line {problems[0].line_number}")

    return records, problems
