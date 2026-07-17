"""
Pytest suite: cross-validates analytic (closed-form) Greeks against
finite-difference Greeks. If these ever disagree by more than a tiny
tolerance, it means the analytic formula has a bug -- this is the most
important test in the whole suite for that reason.

Run with: pytest tests/test_finite_diff_greeks.py -v
"""

import pytest

from backend.pricing.black_scholes import greeks as analytic_greeks
from backend.pricing.finite_diff_greeks import fd_greeks


CASES = [
    (100, 100, 1.0, 0.05, 0.20, "call"),
    (100, 100, 1.0, 0.05, 0.20, "put"),
    (120, 100, 0.5, 0.03, 0.30, "call"),
    (80, 100, 0.5, 0.03, 0.30, "call"),
    (100, 100, 0.05, 0.02, 0.15, "call"),
]


@pytest.mark.parametrize("S,K,T,r,sigma,option_type", CASES)
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_analytic_matches_finite_difference(S, K, T, r, sigma, option_type, greek):
    analytic = analytic_greeks(S, K, T, r, sigma, option_type)[greek]
    numerical = fd_greeks(S, K, T, r, sigma, option_type)[greek]

    scale = max(abs(analytic), 1e-6)
    relative_error = abs(analytic - numerical) / scale

    assert relative_error < 1e-3