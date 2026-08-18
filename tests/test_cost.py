import pytest

from llm_cost import default_pricing, estimate_cost


def test_estimate_cost_matches_readme_example():
    price = default_pricing().resolve("claude-opus-5")
    result = estimate_cost(price, input_tokens=12000, output_tokens=800, cached_input_tokens=52000)
    assert result.total_cost == pytest.approx(0.106)


def test_estimate_cost_breaks_down_by_token_class():
    price = default_pricing().resolve("claude-opus-5")
    result = estimate_cost(price, input_tokens=12000, output_tokens=800)
    assert result.input_cost == pytest.approx(0.06)
    assert result.output_cost == pytest.approx(0.02)
    assert result.cached_input_cost == 0.0
    assert result.cache_write_cost == 0.0


def test_estimate_cost_scales_with_calls():
    price = default_pricing().resolve("claude-opus-5")
    one = estimate_cost(price, input_tokens=1000, output_tokens=100)
    many = estimate_cost(price, input_tokens=1000, output_tokens=100, calls=50)
    assert many.total_cost == pytest.approx(one.total_cost * 50)
    assert many.input_tokens == one.input_tokens * 50


def test_cost_per_call():
    price = default_pricing().resolve("claude-opus-5")
    result = estimate_cost(price, input_tokens=1000, output_tokens=1000, calls=4)
    assert result.cost_per_call == pytest.approx(result.total_cost / 4)


def test_estimate_cost_rejects_zero_calls():
    price = default_pricing().resolve("claude-opus-5")
    with pytest.raises(ValueError):
        estimate_cost(price, input_tokens=1, calls=0)


def test_to_dict_is_json_ready_and_rounded():
    price = default_pricing().resolve("claude-opus-5")
    result = estimate_cost(price, input_tokens=12000, output_tokens=800, cached_input_tokens=52000)
    data = result.to_dict()
    assert data["model"] == "claude-opus-5"
    assert data["total_cost"] == pytest.approx(0.106)
    assert isinstance(data["input_tokens"], int)
