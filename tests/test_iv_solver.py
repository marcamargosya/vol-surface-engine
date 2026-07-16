"""
Pytest suite for the implied vol solver.

Run with: pytest tests/test_iv_solver.py -v
"""

import pytest

from backend.pricing.black_scholes import price
from backend.pricing.iv_solver import implied_vol


def test_recovers_known_vol_for_call():
    true_sigma = 0.25
    S, K, T, r = 100, 105, 0.5, 0.03

    market_price = price(S, K, T, r, true_sigma, "call")
    recovered = implied_vol(market_price, S, K, T, r, "call")

    assert abs(recovered - true_sigma) < 1e-4


def test_recovers_known_vol_for_put():
    true_sigma = 0.25
    S, K, T, r = 100, 105, 0.5, 0.03

    market_price = price(S, K, T, r, true_sigma, "put")
    recovered = implied_vol(market_price, S, K, T, r, "put")

    assert abs(recovered - true_sigma) < 1e-4


def test_deep_itm_triggers_bisection_fallback():
    # Deep ITM options have tiny vega, which breaks Newton-Raphson -- this
    # case only passes if the bisection fallback actually kicks in.
    S, K, T, r, true_sigma = 200, 100, 0.05, 0.03, 0.2
    market_price = price(S, K, T, r, true_sigma, "call")
    recovered = implied_vol(market_price, S, K, T, r, "call")

    assert abs(recovered - true_sigma) < 1e-3


def test_arbitrage_violating_price_returns_none():
    # A price far above any achievable Black-Scholes value should not
    # silently produce a garbage vol -- it should return None.
    S, K, T, r = 100, 105, 0.5, 0.03
    impossible_price = 1000

    result = implied_vol(impossible_price, S, K, T, r, "call")
    assert result is None


def test_negative_price_returns_none():
    S, K, T, r = 100, 105, 0.5, 0.03
    result = implied_vol(-5, S, K, T, r, "call")
    assert result is None


@pytest.mark.parametrize("true_sigma", [0.05, 0.15, 0.30, 0.60, 1.0])
def test_recovers_vol_across_a_range_of_levels(true_sigma):
    # Sweep across low to very high vol regimes to make sure the solver
    # is robust, not just accurate at one convenient value.
    S, K, T, r = 100, 100, 0.25, 0.02
    market_price = price(S, K, T, r, true_sigma, "call")
    recovered = implied_vol(market_price, S, K, T, r, "call")

    assert abs(recovered - true_sigma) < 1e-3