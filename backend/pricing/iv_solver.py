"""
Implied volatility solver.

Given a market price, back out the sigma that makes Black-Scholes match it.

Primary method: Newton-Raphson (fast, uses vega as the derivative).
Fallback: bisection (slow but guaranteed to converge) — used when:
    - vega is too small (deep ITM/OTM, near expiry) causing NR to blow up
    - NR overshoots into sigma <= 0
    - NR fails to converge within max_iter

This dual-method approach is exactly what a real risk desk does: Newton-Raphson
for speed on the vast majority of quotes, bisection as a safety net so a single
bad quote doesn't crash the pipeline.
"""

import numpy as np
from .black_scholes import price, vega


def _bisection(target_price, S, K, T, r, option_type, q,
                lo=1e-6, hi=5.0, tol=1e-6, max_iter=100):
    price_lo = price(S, K, T, r, lo, option_type, q) - target_price
    price_hi = price(S, K, T, r, hi, option_type, q) - target_price

    if price_lo * price_hi > 0:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price_mid = price(S, K, T, r, mid, option_type, q) - target_price

        if abs(price_mid) < tol:
            return mid

        if price_lo * price_mid < 0:
            hi = mid
            price_hi = price_mid
        else:
            lo = mid
            price_lo = price_mid

    return (lo + hi) / 2


def implied_vol(target_price, S, K, T, r, option_type="call", q=0.0,
                 initial_guess=0.2, tol=1e-6, max_iter=50):
    """
    Solve for sigma such that BS price(S, K, T, r, sigma) == target_price.

    Returns None if no solution exists (e.g. target_price violates
    no-arbitrage bounds — happens with stale/bad quotes, don't silently
    return garbage in that case).
    """
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    upper_bound = S if option_type == "call" else K
    if target_price < intrinsic - tol or target_price > upper_bound + tol:
        return None

    sigma = initial_guess
    for _ in range(max_iter):
        model_price = price(S, K, T, r, sigma, option_type, q)
        diff = model_price - target_price

        if abs(diff) < tol:
            return sigma

        v = vega(S, K, T, r, sigma, q)
        if v < 1e-8:
            break

        sigma = sigma - diff / v

        if sigma <= 0 or sigma > 5:
            break
    else:
        return _bisection(target_price, S, K, T, r, option_type, q)

    return _bisection(target_price, S, K, T, r, option_type, q)


if __name__ == "__main__":
    true_sigma = 0.25
    S, K, T, r = 100, 105, 0.5, 0.03

    for opt_type in ["call", "put"]:
        market_price = price(S, K, T, r, true_sigma, opt_type)
        recovered = implied_vol(market_price, S, K, T, r, opt_type)
        print(f"{opt_type}: true={true_sigma}, recovered={recovered:.6f}")
        assert abs(recovered - true_sigma) < 1e-4, f"IV solver failed for {opt_type}"

    deep_itm_price = price(200, 100, 0.05, 0.03, 0.2, "call")
    recovered_edge = implied_vol(deep_itm_price, 200, 100, 0.05, 0.03, "call")
    print(f"deep ITM: recovered={recovered_edge:.6f}")
    assert abs(recovered_edge - 0.2) < 1e-3, "Deep ITM edge case failed"

    bad_price = 1000
    result = implied_vol(bad_price, S, K, T, r, "call")
    assert result is None, "Should return None for unachievable price"
    print("Arbitrage-violation case correctly returned None")

    print("All IV solver sanity checks passed.")