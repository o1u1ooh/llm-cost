"""Turn a price and a token count into a bill."""

from __future__ import annotations

from dataclasses import dataclass

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class CostResult:
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    input_cost: float
    output_cost: float
    cached_input_cost: float
    cache_write_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost + self.cached_input_cost + self.cache_write_cost

    @property
    def cost_per_call(self) -> float:
        return self.total_cost / self.calls if self.calls else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "cached_input_cost": round(self.cached_input_cost, 6),
            "cache_write_cost": round(self.cache_write_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "cost_per_call": round(self.cost_per_call, 6),
        }


def estimate_cost(
    price,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    calls: int = 1,
) -> CostResult:
    if calls < 1:
        raise ValueError("calls must be >= 1")

    return CostResult(
        model=price.name,
        calls=calls,
        input_tokens=input_tokens * calls,
        output_tokens=output_tokens * calls,
        cached_input_tokens=cached_input_tokens * calls,
        cache_write_tokens=cache_write_tokens * calls,
        input_cost=input_tokens * calls * price.input / _PER_MILLION,
        output_cost=output_tokens * calls * price.output / _PER_MILLION,
        cached_input_cost=cached_input_tokens * calls * price.cached_input / _PER_MILLION,
        cache_write_cost=cache_write_tokens * calls * price.cache_write / _PER_MILLION,
    )
