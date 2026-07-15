"""
Black-Scholes European option pricing and Greeks.

Convention:
    S     : spot price
    K     : strike
    T     : time to expiry (years)
    r     : risk-free rate (continuously compounded)
    sigma : volatility (annualized)
    q     : continuous dividend yield (default 0 — scope discipline: no dividends yet,
            kept as a parameter so the surface layer doesn't need a rewrite later)

All formulas derived directly from the BS PDE solution, not looked up as a
one-line library call. norm.cdf/pdf are just the standard normal CDF/PDF —
using them isn't "using a BS library", it's using the definition of N(x).
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be > 0")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        return term1 + term2 + term3
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        return term1 + term2 + term3


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Bundle all Greeks in one call — convenient for the position/risk layer."""
    return {
        "price": price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }


if __name__ == "__main__":
    g = greeks(100, 100, 1, 0.05, 0.2, "call")
    print(g)
    assert abs(g["price"] - 10.4506) < 1e-3, "Call price sanity check failed"

    p = greeks(100, 100, 1, 0.05, 0.2, "put")
    print(p)
    lhs = g["price"] - p["price"]
    rhs = 100 * np.exp(0) - 100 * np.exp(-0.05 * 1)
    assert abs(lhs - rhs) < 1e-8, "Put-call parity failed"

    print("All sanity checks passed.")