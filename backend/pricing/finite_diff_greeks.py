"""
Finite-difference ("bump-and-reprice") Greeks.

This module computes every Greek numerically, by nudging each input by a
small epsilon and repricing, then takes the finite-difference derivative.
It exists purely to validate black_scholes.py's closed-form Greeks against
an independent method -- if the analytic formulas had a sign error or a
wrong derivative, the closed-form and finite-difference values would not
agree to high precision. This is exactly how a real pricing library gets
validated before it's trusted in production: never trust a single method
for something this important.

Central differences are used throughout (f(x+h) - f(x-h)) / (2h) rather than
forward differences, since central differences have O(h^2) error instead of
O(h) -- meaningfully more accurate for the same step size.
"""

import numpy as np

from .black_scholes import price


def fd_delta(S, K, T, r, sigma, option_type="call", q=0.0, h=None):
    h = h or S * 1e-4
    return (price(S + h, K, T, r, sigma, option_type, q)
            - price(S - h, K, T, r, sigma, option_type, q)) / (2 * h)


def fd_gamma(S, K, T, r, sigma, option_type="call", q=0.0, h=None):
    h = h or S * 1e-3  # gamma needs a slightly larger step -- it's a 2nd derivative
    return (price(S + h, K, T, r, sigma, option_type, q)
            - 2 * price(S, K, T, r, sigma, option_type, q)
            + price(S - h, K, T, r, sigma, option_type, q)) / (h ** 2)


def fd_vega(S, K, T, r, sigma, option_type="call", q=0.0, h=1e-4):
    return (price(S, K, T, r, sigma + h, option_type, q)
            - price(S, K, T, r, sigma - h, option_type, q)) / (2 * h)


def fd_theta(S, K, T, r, sigma, option_type="call", q=0.0, h=None):
    # Theta is technically d(price)/d(time), which is the negative of
    # d(price)/d(T) since T = time to expiry decreases as time moves forward.
    h = h or T * 1e-4
    return -(price(S, K, T + h, r, sigma, option_type, q)
             - price(S, K, T - h, r, sigma, option_type, q)) / (2 * h)


def fd_rho(S, K, T, r, sigma, option_type="call", q=0.0, h=1e-5):
    return (price(S, K, T, r + h, sigma, option_type, q)
            - price(S, K, T, r - h, sigma, option_type, q)) / (2 * h)


def fd_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Bundle all finite-difference Greeks -- mirrors black_scholes.greeks()."""
    return {
        "price": price(S, K, T, r, sigma, option_type, q),
        "delta": fd_delta(S, K, T, r, sigma, option_type, q),
        "gamma": fd_gamma(S, K, T, r, sigma, option_type, q),
        "vega": fd_vega(S, K, T, r, sigma, option_type, q),
        "theta": fd_theta(S, K, T, r, sigma, option_type, q),
        "rho": fd_rho(S, K, T, r, sigma, option_type, q),
    }


if __name__ == "__main__":
    from .black_scholes import greeks as analytic_greeks

    test_cases = [
        (100, 100, 1.0, 0.05, 0.20, "call"),   # ATM
        (100, 100, 1.0, 0.05, 0.20, "put"),
        (120, 100, 0.5, 0.03, 0.30, "call"),   # ITM call
        (80, 100, 0.5, 0.03, 0.30, "call"),    # OTM call
        (100, 100, 0.05, 0.02, 0.15, "call"),  # near expiry
    ]

    print(f"{'Case':<30} {'Greek':<8} {'Analytic':>12} {'FiniteDiff':>12} {'AbsDiff':>10}")
    max_relative_error = 0.0

    for S, K, T, r, sigma, opt_type in test_cases:
        analytic = analytic_greeks(S, K, T, r, sigma, opt_type)
        numerical = fd_greeks(S, K, T, r, sigma, opt_type)
        label = f"S={S},K={K},T={T},{opt_type}"

        for greek in ["delta", "gamma", "vega", "theta", "rho"]:
            a, n = analytic[greek], numerical[greek]
            diff = abs(a - n)
            scale = max(abs(a), 1e-6)
            rel_error = diff / scale
            max_relative_error = max(max_relative_error, rel_error)
            print(f"{label:<30} {greek:<8} {a:>12.6f} {n:>12.6f} {diff:>10.2e}")

    print(f"\nMax relative error across all Greeks/cases: {max_relative_error:.2e}")
    assert max_relative_error < 1e-3, "Finite-difference Greeks disagree with analytic Greeks!"
    print("Analytic Greeks validated against independent finite-difference method.")