"""
Pytest suite for the interpolated vol surface.

Run with: pytest tests/test_interpolate.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backend.pricing.black_scholes import price
from backend.surface.interpolate import build_iv_table, VolSurface, RISK_FREE_RATE


def _make_synthetic_chain(smile_fn, expiries_days, spot=100):
    """
    Builds a synthetic options chain DataFrame from a given smile function
    smile_fn(K) -> IV, across a fixed strike grid and set of expiries.
    """
    strikes = np.array([80, 90, 100, 110, 120])
    rows = []
    for T_days in expiries_days:
        T = T_days / 365
        for K in strikes:
            true_iv = smile_fn(K)
            option_type = "call" if K >= spot else "put"
            mid = price(spot, K, T, RISK_FREE_RATE, true_iv, option_type)
            rows.append({
                "spot": spot, "strike": K, "T": T,
                "option_type": option_type, "mid": mid,
            })
    return pd.DataFrame(rows)


def test_build_iv_table_recovers_all_contracts():
    # A clean synthetic smile should invert without dropping any contracts
    flat_smile = lambda K: 0.20
    chain = _make_synthetic_chain(flat_smile, [30, 60, 90])
    iv_table = build_iv_table(chain)

    assert len(iv_table) == len(chain)


def test_recovers_exact_iv_at_fitted_expiry():
    flat_smile = lambda K: 0.20
    chain = _make_synthetic_chain(flat_smile, [30, 60, 90])
    iv_table = build_iv_table(chain)
    surface = VolSurface(iv_table)

    recovered = surface.get_iv(K=100, T=30 / 365)
    assert abs(recovered - 0.20) < 0.01


def test_interpolates_sensibly_between_expiries():
    flat_smile = lambda K: 0.20
    chain = _make_synthetic_chain(flat_smile, [30, 60, 90])
    iv_table = build_iv_table(chain)
    surface = VolSurface(iv_table)

    # 45 days sits between the 30d and 60d fitted expiries
    mid_iv = surface.get_iv(K=100, T=45 / 365)
    assert 0.15 < mid_iv < 0.25


def test_captures_smile_shape():
    # A smile that's higher away from ATM should be reflected in the fit --
    # OTM strikes should show meaningfully higher IV than ATM.
    smile = lambda K: 0.20 + 0.05 * abs(K - 100) / 20
    chain = _make_synthetic_chain(smile, [30, 60, 90])
    iv_table = build_iv_table(chain)
    surface = VolSurface(iv_table)

    atm_iv = surface.get_iv(K=100, T=30 / 365)
    otm_iv = surface.get_iv(K=120, T=30 / 365)
    assert otm_iv > atm_iv


def test_raises_with_only_one_expiry():
    flat_smile = lambda K: 0.20
    chain = _make_synthetic_chain(flat_smile, [30])  # only one expiry
    iv_table = build_iv_table(chain)

    with pytest.raises(ValueError):
        VolSurface(iv_table)


def test_raises_on_empty_iv_table():
    empty = pd.DataFrame(columns=["strike", "T", "iv"])
    with pytest.raises(ValueError):
        VolSurface(empty)