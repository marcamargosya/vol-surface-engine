"""
Pytest suite for multi-leg position aggregation.

Run with: pytest tests/test_position.py -v
"""

import pytest

from backend.pricing.black_scholes import price
from backend.risk.position import Position, Leg


class _FlatSurface:
    """Stub surface with constant IV, so tests don't depend on real market data."""
    def get_iv(self, K, T):
        return 0.20


def test_position_requires_at_least_one_leg():
    with pytest.raises(ValueError):
        Position([])


def test_bull_call_spread_has_positive_delta():
    surface = _FlatSurface()
    S, r = 100, 0.03

    spread = Position([
        Leg(strike=100, T=0.5, option_type="call", quantity=1),
        Leg(strike=110, T=0.5, option_type="call", quantity=-1),
    ])

    greeks = spread.aggregate_greeks(S, r, surface)
    assert greeks["delta"] > 0


def test_bull_call_spread_price_matches_manual_calc():
    surface = _FlatSurface()
    S, r = 100, 0.03

    spread = Position([
        Leg(strike=100, T=0.5, option_type="call", quantity=1),
        Leg(strike=110, T=0.5, option_type="call", quantity=-1),
    ])

    greeks = spread.aggregate_greeks(S, r, surface)

    long_call = price(S, 100, 0.5, r, 0.20, "call")
    short_call = price(S, 110, 0.5, r, 0.20, "call")
    expected_price = long_call - short_call

    assert abs(greeks["price"] - expected_price) < 1e-8


def test_long_straddle_has_near_zero_delta():
    # A straddle (long call + long put, same strike/expiry) is close to
    # delta-neutral when struck at the strike -- but not exactly zero,
    # since with r > 0 the forward price sits above spot, so a straddle
    # struck at spot (not the forward) has a small positive delta tilt.
    surface = _FlatSurface()
    S, r = 100, 0.03

    straddle = Position([
        Leg(strike=100, T=0.25, option_type="call", quantity=1),
        Leg(strike=100, T=0.25, option_type="put", quantity=1),
    ])

    greeks = straddle.aggregate_greeks(S, r, surface)
    assert abs(greeks["delta"]) < 0.15


def test_long_straddle_has_positive_vega():
    surface = _FlatSurface()
    S, r = 100, 0.03

    straddle = Position([
        Leg(strike=100, T=0.25, option_type="call", quantity=1),
        Leg(strike=100, T=0.25, option_type="put", quantity=1),
    ])

    greeks = straddle.aggregate_greeks(S, r, surface)
    assert greeks["vega"] > 0


def test_short_position_flips_sign_of_greeks():
    surface = _FlatSurface()
    S, r = 100, 0.03

    long_call = Position([Leg(strike=100, T=0.5, option_type="call", quantity=1)])
    short_call = Position([Leg(strike=100, T=0.5, option_type="call", quantity=-1)])

    long_greeks = long_call.aggregate_greeks(S, r, surface)
    short_greeks = short_call.aggregate_greeks(S, r, surface)

    assert abs(long_greeks["delta"] + short_greeks["delta"]) < 1e-10
    assert abs(long_greeks["price"] + short_greeks["price"]) < 1e-10