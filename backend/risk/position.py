"""
Multi-leg option position: aggregates price and Greeks across legs.

A "leg" is one option contract with a quantity (+ve = long, -ve = short).
Position-level Greeks are just the quantity-weighted sum of per-leg Greeks --
this is true because Black-Scholes price is linear in the sense that a
portfolio's Greeks are the sum of its components' Greeks (each Greek is a
partial derivative of the total price, and differentiation is linear).
"""

from dataclasses import dataclass

from ..pricing.black_scholes import greeks


@dataclass
class Leg:
    strike: float
    T: float
    option_type: str  # "call" or "put"
    quantity: float    # +1 = long 1 contract, -2 = short 2 contracts, etc.


class Position:
    def __init__(self, legs: list[Leg]):
        if not legs:
            raise ValueError("Position must have at least one leg")
        self.legs = legs

    def aggregate_greeks(self, S: float, r: float, vol_surface, q: float = 0.0) -> dict:
        totals = {"price": 0.0, "delta": 0.0, "gamma": 0.0,
                  "vega": 0.0, "theta": 0.0, "rho": 0.0}

        for leg in self.legs:
            sigma = vol_surface.get_iv(leg.strike, leg.T)
            leg_greeks = greeks(S, leg.strike, leg.T, r, sigma, leg.option_type, q)

            for key in totals:
                totals[key] += leg.quantity * leg_greeks[key]

        return totals

    def leg_breakdown(self, S: float, r: float, vol_surface, q: float = 0.0) -> list[dict]:
        breakdown = []
        for leg in self.legs:
            sigma = vol_surface.get_iv(leg.strike, leg.T)
            leg_greeks = greeks(S, leg.strike, leg.T, r, sigma, leg.option_type, q)
            row = {
                "strike": leg.strike, "T": leg.T,
                "option_type": leg.option_type, "quantity": leg.quantity,
                "iv": sigma,
            }
            row.update({k: leg.quantity * v for k, v in leg_greeks.items()})
            breakdown.append(row)
        return breakdown


if __name__ == "__main__":
    from ..pricing.black_scholes import price

    class _FlatSurface:
        def get_iv(self, K, T):
            return 0.20

    surface = _FlatSurface()
    S, r = 100, 0.03

    spread = Position([
        Leg(strike=100, T=0.5, option_type="call", quantity=1),
        Leg(strike=110, T=0.5, option_type="call", quantity=-1),
    ])

    agg = spread.aggregate_greeks(S, r, surface)
    print("Bull call spread Greeks:", {k: round(v, 4) for k, v in agg.items()})

    assert agg["delta"] > 0, "Bull call spread should have positive delta"
    assert agg["price"] > 0, "Net debit spread should cost money upfront"

    long_call_price = price(S, 100, 0.5, r, 0.20, "call")
    short_call_price = price(S, 110, 0.5, r, 0.20, "call")
    expected_price = long_call_price - short_call_price
    assert abs(agg["price"] - expected_price) < 1e-8, "Aggregation doesn't match manual calc"

    print(f"Manual price check: {expected_price:.4f} vs aggregated: {agg['price']:.4f}")
    print("All position aggregation sanity checks passed.")