"""
Dupire local volatility, extracted from a fitted implied vol surface.

Local vol answers a different question than implied vol: implied vol is
"what constant vol, plugged into Black-Scholes, reproduces this option's
market price." Local vol is "what instantaneous vol must the underlying
actually have, right now, at this exact spot level and time, to be
consistent with the ENTIRE observed vol surface at once." It's the
volatility surface a stochastic model like local-vol Monte Carlo would
actually simulate with.

Uses Gatheral's practitioner formula (in "The Volatility Surface"), written
in terms of total implied variance w(y, T) = sigma_impl(K,T)^2 * T, where
y = log(K / F(T)) is log-moneyness relative to the forward F(T) = S*e^(rT):

    sigma_local(K,T)^2 =
        (dw/dT) / [ 1 - (y/w)(dw/dy) + 1/4*(-1/4 - 1/w + y^2/w^2)(dw/dy)^2
                    + 1/2*(d^2w/dy^2) ]

This is preferred over Dupire's original price-derivative form because it
avoids differentiating the option price twice (numerically unstable) --
differentiating the smoother implied vol surface is far better conditioned.

Derivatives are computed via central finite differences directly on the
fitted surface object (works with either VolSurface or SABRSurface, since
both expose the same get_iv(K, T) interface).

A negative value under the square root means the fitted implied vol surface
has an internal arbitrage (calendar spread or butterfly arbitrage) at that
point -- there is no local vol model consistent with it there. Rather than
silently returning garbage, this is surfaced as NaN, which is itself a
useful diagnostic on the quality of the surface fit.
"""

import numpy as np


def _total_variance(surface, y: float, T: float, S: float, r: float) -> float:
    """w(y, T) = sigma_impl(K, T)^2 * T, where K = F(T) * e^y."""
    F = S * np.exp(r * T)
    K = F * np.exp(y)
    sigma = surface.get_iv(K, T)
    return sigma ** 2 * T


def dupire_local_vol(surface, K: float, T: float, S: float, r: float,
                      dT: float = None, dy: float = 1e-3) -> float:
    """
    Local vol at strike K, time T, via Gatheral's formula.
    Returns NaN if the surface implies a negative local variance there
    (a sign of calendar or butterfly arbitrage in the fitted surface).
    """
    F = S * np.exp(r * T)
    y = np.log(K / F)
    dT = dT or max(T * 0.05, 1 / 365)  # bump T by 5% or at least 1 day

    T_lo = max(T - dT, 1e-4)  # keep T positive
    w = _total_variance(surface, y, T, S, r)
    w_dT_hi = _total_variance(surface, y, T + dT, S, r)
    w_dT_lo = _total_variance(surface, y, T_lo, S, r)
    dw_dT = (w_dT_hi - w_dT_lo) / (T + dT - T_lo)

    w_dy_hi = _total_variance(surface, y + dy, T, S, r)
    w_dy_lo = _total_variance(surface, y - dy, T, S, r)
    dw_dy = (w_dy_hi - w_dy_lo) / (2 * dy)
    d2w_dy2 = (w_dy_hi - 2 * w + w_dy_lo) / (dy ** 2)

    if w < 1e-10:
        return float("nan")  # avoid division by ~0 total variance

    denominator = (
        1
        - (y / w) * dw_dy
        + 0.25 * (-0.25 - 1 / w + (y ** 2) / (w ** 2)) * (dw_dy ** 2)
        + 0.5 * d2w_dy2
    )

    local_var = dw_dT / denominator if denominator > 0 else float("nan")

    if local_var < 0:
        return float("nan")

    return float(np.sqrt(local_var))


def local_vol_grid(surface, strikes: np.ndarray, expiries: np.ndarray,
                    S: float, r: float) -> np.ndarray:
    """Full local vol grid (strike x expiry) for plotting alongside the implied vol surface."""
    return np.array([
        [dupire_local_vol(surface, K, T, S, r) for K in strikes]
        for T in expiries
    ])


if __name__ == "__main__":
    # Sanity check: a FLAT implied vol surface (no smile, no term structure)
    # has a known, exact local vol answer -- it must equal that same flat
    # vol everywhere. This is a real analytic identity, not an approximation:
    # with no skew (dw/dy = 0) and no curvature (d2w/dy2 = 0), Gatheral's
    # formula collapses to local_var = dw/dT = sigma^2 exactly.
    class _FlatSurface:
        def __init__(self, flat_iv):
            self.flat_iv = flat_iv
        def get_iv(self, K, T):
            return self.flat_iv

    flat_iv = 0.25
    S, r = 100, 0.03
    surface = _FlatSurface(flat_iv)

    test_points = [(100, 0.5), (80, 0.5), (120, 1.0), (100, 0.1)]
    print(f"Flat surface (sigma={flat_iv}) -- local vol should recover {flat_iv} everywhere:\n")
    max_error = 0.0
    for K, T in test_points:
        local_vol = dupire_local_vol(surface, K, T, S, r)
        error = abs(local_vol - flat_iv)
        max_error = max(max_error, error)
        print(f"  K={K:>5}, T={T:>4}  ->  local_vol={local_vol:.6f}  (error={error:.2e})")

    assert max_error < 1e-3, "Local vol should exactly recover a flat implied vol surface"
    print(f"\nMax error: {max_error:.2e}")
    print("Dupire local vol correctly collapses to flat vol on a smile-free surface.")