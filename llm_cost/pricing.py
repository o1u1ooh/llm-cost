"""The price table: a built-in snapshot, JSON overrides, and name resolution.

Prices are USD per 1,000,000 tokens. `cached_input` and `cache_write` fall
back to the `input` price when a model doesn't specify them, which
over-states rather than under-states a bill for models we know less about.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

BUILTIN_AS_OF = "2026-06-24"

_BUILTIN_MODELS = {
    "claude-haiku-4-5": {"provider": "anthropic", "input": 1.00, "output": 5.00, "cached_input": 0.10, "cache_write": 1.25},
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write": 3.75},
    "claude-sonnet-5": {"provider": "anthropic", "input": 3.00, "output": 15.00, "cached_input": 0.30, "cache_write": 3.75},
    "claude-opus-4-6": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-opus-4-7": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-opus-4-8": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-opus-5": {"provider": "anthropic", "input": 5.00, "output": 25.00, "cached_input": 0.50, "cache_write": 6.25},
    "claude-fable-5": {"provider": "anthropic", "input": 10.00, "output": 50.00, "cached_input": 1.00, "cache_write": 12.50},
    "gpt-4o": {"provider": "openai", "input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"provider": "openai", "input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"provider": "openai", "input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gemini-2.5-flash": {"provider": "google", "input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"provider": "google", "input": 1.25, "output": 10.00},
}

_PROVIDER_PREFIXES = ("anthropic", "openai", "google")
_DATE_SUFFIX_RE = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")

_CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
DEFAULT_PRECISION = 4


class PricingError(Exception):
    """The pricing table or an override file could not be built."""


class UnknownModelError(Exception):
    def __init__(self, name):
        super().__init__(f"no price for model {name!r}")
        self.name = name


@dataclass(frozen=True)
class ModelPrice:
    name: str
    provider: str
    input: float
    output: float
    cached_input: float
    cache_write: float


@dataclass
class PricingTable:
    models: dict
    as_of: str
    source: str
    currency: str = "USD"
    precision: int = DEFAULT_PRECISION

    @property
    def symbol(self) -> str:
        """A short prefix for rendering an amount: a currency symbol for
        currencies we know, otherwise the ISO code followed by a space."""
        return _CURRENCY_SYMBOLS.get(self.currency, f"{self.currency} ")

    def resolve(self, name: str) -> ModelPrice:
        """Look up a model, tolerating a leading provider prefix and a
        trailing dated-snapshot suffix (e.g. "anthropic.claude-opus-5-20260101")."""
        candidate = name.strip()
        if candidate in self.models:
            return self.models[candidate]

        if "." in candidate:
            prefix, _, rest = candidate.partition(".")
            if prefix in _PROVIDER_PREFIXES:
                if rest in self.models:
                    return self.models[rest]
                candidate = rest

        stripped = _DATE_SUFFIX_RE.sub("", candidate)
        if stripped in self.models:
            return self.models[stripped]

        for known in sorted(self.models, key=len, reverse=True):
            if stripped == known or stripped.startswith(known + "-"):
                return self.models[known]

        raise UnknownModelError(name)


def _build_models(raw: dict) -> dict:
    models = {}
    for name, data in raw.items():
        if "input" not in data or "output" not in data:
            raise PricingError(f"pricing entry for {name!r} needs both 'input' and 'output'")
        models[name] = ModelPrice(
            name=name,
            provider=data.get("provider", "unknown"),
            input=float(data["input"]),
            output=float(data["output"]),
            cached_input=float(data.get("cached_input", data["input"])),
            cache_write=float(data.get("cache_write", data["input"])),
        )
    return models


def default_pricing() -> PricingTable:
    return PricingTable(models=_build_models(_BUILTIN_MODELS), as_of=BUILTIN_AS_OF, source="built-in")


def parse_pricing(obj: dict, *, source: str = "override") -> PricingTable:
    if not isinstance(obj, dict):
        raise PricingError("pricing override must be a JSON object")

    as_of = obj.get("as_of", BUILTIN_AS_OF)
    replace = bool(obj.get("replace", False))
    overrides = obj.get("models", {})
    if not isinstance(overrides, dict):
        raise PricingError("'models' must be a JSON object")

    currency = obj.get("currency", "USD")
    if not isinstance(currency, str) or not currency:
        raise PricingError("'currency' must be a non-empty string")

    precision = obj.get("precision", DEFAULT_PRECISION)
    if not isinstance(precision, int) or isinstance(precision, bool) or precision < 0:
        raise PricingError("'precision' must be a non-negative integer")

    base = {} if replace else {name: dict(data) for name, data in _BUILTIN_MODELS.items()}

    for name, data in overrides.items():
        if not isinstance(data, dict):
            raise PricingError(f"pricing entry for {name!r} must be a JSON object")
        merged = dict(base.get(name, {}))
        merged.update(data)
        base[name] = merged

    return PricingTable(models=_build_models(base), as_of=as_of, source=source, currency=currency, precision=precision)


def load_pricing(path: str) -> PricingTable:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise PricingError(f"cannot read {path}: {exc.strerror}") from exc

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PricingError(f"invalid JSON in {path}: {exc.msg}") from exc

    return parse_pricing(obj, source=path)
