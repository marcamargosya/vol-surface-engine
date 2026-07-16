"""
Pytest suite for the SABR model formula and calibration.

Run with: pytest tests/test_sabr.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backend.pricing.black_scholes import price
from backend.surface.interpolate import build_iv_table, RISK_FREE_RATE
from backend.surface.sabr import sabr_implied_vol, calibrate_sabr, SABRSurface


def test_calibration_recovers_known_params():
    true_alpha, true_beta, true_rho, true_nu = 0.3, 1.0, -0.3, 0.6
    F, T = 100.0, 0.5

    strikes = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
    true_ivs = np.array([
        sabr_implied_vol(F, K, T, true_alpha, true_beta, true_rho, true_nu)
        for K in strikes
    ])

    alpha, beta, rho, nu = calibrate_sabr(strikes, true_ivs, F, T, beta=true_beta)

    fitted_ivs = np.array([
        sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
    ])
    max_error = np.max(np.abs(fitted_ivs - true_ivs))

    assert max_error < 1e-3


def test_atm_formula_matches_general_formula_in_the_limit():
    # The ATM special case (F == K) should agree with the general formula
    # evaluated at a strike extremely close to F -- confirms there's no
    # discontinuity at the 0/0 boundary the code guards against.
    F, T, alpha, beta, rho, nu = 100.0, 0.5, 0.3, 1.0, -0.3, 0.6

    atm_iv = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
    near_atm_iv = sabr_implied_vol(F, F * 1.0001, T, alpha, beta, rho, nu)

    assert abs(atm_iv - near_atm_iv) < 1e-3


def test_negative_rho_produces_downward_skew():
    # Negative rho is the standard equity smile shape: puts (low strikes)
    # trade at higher IV than calls (high strikes) -- this is what hedging
    # demand for downside protection looks like in the market.
    F, T, alpha, beta, nu = 100.0, 0.5, 0.3, 1.0, 0.6
    rho = -0.5

    low_strike_iv = sabr_implied_vol(F, 80, T, alpha, beta, rho, nu)
    high_strike_iv = sabr_implied_vol(F, 120, T, alpha, beta, rho, nu)

    assert low_strike_iv > high_strike_iv


def _make_synthetic_chain(smile_fn, expiries_days, spot=100):
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


def test_sabr_surface_end_to_end():
    smile = lambda K: 0.20 + 0.05 * abs(K - 100) / 20
    chain = _make_synthetic_chain(smile, [30, 60, 90])
    iv_table = build_iv_table(chain)

    surface = SABRSurface(iv_table, r=RISK_FREE_RATE)
    atm_iv = surface.get_iv(K=100, T=30 / 365)

    # SABR is an approximation, so allow more tolerance than the exact spline fit
    assert abs(atm_iv - 0.20) < 0.02


def test_sabr_surface_raises_with_only_one_expiry():
    smile = lambda K: 0.20
    chain = _make_synthetic_chain(smile, [30])
    iv_table = build_iv_table(chain)

    with pytest.raises(ValueError):
        SABRSurface(iv_table, r=RISK_FREE_RATE)