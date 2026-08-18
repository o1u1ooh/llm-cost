import pytest

from llm_cost import PricingError, UnknownModelError, default_pricing, parse_pricing


def test_default_pricing_matches_documented_opus_price():
    table = default_pricing()
    price = table.resolve("claude-opus-5")
    assert price.input == 5.00
    assert price.output == 25.00
    assert price.cached_input == 0.50
    assert price.cache_write == 6.25


def test_resolve_tolerates_provider_prefix_and_dated_suffix():
    table = default_pricing()
    price = table.resolve("anthropic.claude-opus-5-20260101")
    assert price.name == "claude-opus-5"


def test_resolve_exact_match_still_works():
    table = default_pricing()
    assert table.resolve("gpt-4o-mini").provider == "openai"


def test_resolve_unknown_model_raises():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("some-model-nobody-has-priced")


def test_cached_input_and_cache_write_fall_back_to_input_price():
    table = default_pricing()
    price = table.resolve("gemini-2.5-flash")
    assert price.cached_input == price.input
    assert price.cache_write == price.input


def test_parse_pricing_merges_onto_builtin_by_default():
    table = parse_pricing({
        "as_of": "2026-07-01",
        "models": {
            "claude-opus-5": {"input": 4.25, "output": 21.0, "cached_input": 0.42, "cache_write": 5.3},
            "internal-router-v3": {"input": 0.2, "output": 0.8, "provider": "internal"},
        },
    })

    assert table.as_of == "2026-07-01"

    opus = table.resolve("claude-opus-5")
    assert opus.input == 4.25
    assert opus.provider == "anthropic"

    router = table.resolve("internal-router-v3")
    assert router.provider == "internal"

    # untouched built-in entries survive the merge
    assert table.resolve("claude-haiku-4-5").input == 1.00


def test_parse_pricing_replace_drops_untouched_builtins():
    table = parse_pricing({
        "replace": True,
        "models": {"internal-router-v3": {"input": 0.2, "output": 0.8}},
    })

    table.resolve("internal-router-v3")
    with pytest.raises(UnknownModelError):
        table.resolve("claude-opus-5")


def test_parse_pricing_rejects_non_object():
    with pytest.raises(PricingError):
        parse_pricing(["not", "an", "object"])


def test_parse_pricing_rejects_entry_missing_prices():
    with pytest.raises(PricingError):
        parse_pricing({"models": {"broken-model": {"provider": "internal"}}})
