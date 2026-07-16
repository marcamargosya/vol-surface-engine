"""
Pytest suite for Black-Scholes pricing and Greeks.

Run with: pytest tests/test_black_scholes.py -v
"""

import numpy as np
import pytest

from backend.pricing.black_scholes import price, greeks


def test_call_price_matches_known_reference():
    # Standard textbook reference case: S=100, K=100, T=1, r=0.05, sigma=0.2
    # Known correct answer: ~10.4506
    result = price(100, 100, 1, 0.05, 0.2, "call")
    assert abs(result - 10.4506) < 1e-3


def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    call_price = price(S, K, T, r, sigma, "call")
    put_price = price(S, K, T, r, sigma, "put")

    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8


def test_call_delta_between_zero_and_one():
    d = greeks(100, 100, 1, 0.05, 0.2, "call")["delta"]
    assert 0 < d < 1


def test_put_delta_between_minus_one_and_zero():
    d = greeks(100, 100, 1, 0.05, 0.2, "put")["delta"]
    assert -1 < d < 0


def test_gamma_is_identical_for_call_and_put():
    # Gamma is the same for calls and puts at identical strikes/expiry --
    # a real property of Black-Scholes, not a coincidence.
    call_gamma = greeks(100, 100, 1, 0.05, 0.2, "call")["gamma"]
    put_gamma = greeks(100, 100, 1, 0.05, 0.2, "put")["gamma"]
    assert abs(call_gamma - put_gamma) < 1e-10


def test_vega_is_identical_for_call_and_put():
    call_vega = greeks(100, 100, 1, 0.05, 0.2, "call")["vega"]
    put_vega = greeks(100, 100, 1, 0.05, 0.2, "put")["vega"]
    assert abs(call_vega - put_vega) < 1e-10


def test_deep_itm_call_delta_near_one():
    # A call struck far below spot should behave almost like owning the stock
    d = greeks(200, 50, 0.1, 0.05, 0.2, "call")["delta"]
    assert d > 0.99


def test_deep_otm_call_delta_near_zero():
    # A call struck far above spot is nearly worthless and barely moves with spot
    d = greeks(50, 200, 0.1, 0.05, 0.2, "call")["delta"]
    assert d < 0.01


def test_price_increases_with_volatility():
    # Vega should always be positive -- more uncertainty, more optionality value
    low_vol_price = price(100, 100, 1, 0.05, 0.1, "call")
    high_vol_price = price(100, 100, 1, 0.05, 0.4, "call")
    assert high_vol_price > low_vol_price


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        price(100, 100, 1, 0.05, 0.2, "invalid_type")