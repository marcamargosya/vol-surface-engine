"""
Interpolated implied vol surface.

Pipeline:
    1. Take raw chain (from chain_fetcher) -> invert mid price to IV per contract
       (using our own iv_solver, not yfinance's built-in IV column)
    2. Build a per-expiry smile: cubic spline across strikes
    3. Interpolate across expiries: linear in total variance (sigma^2 * T),
       not linear in sigma directly -- this is the standard convention because
       total variance is roughly additive over time, so linear interpolation
       there is far more defensible than linear in vol itself.

Scope: European options, no dividends (q=0), flat risk-free rate assumption.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from ..pricing.iv_solver import implied_vol

RISK_FREE_RATE = 0.04  # flat assumption -- scope discipline, no term structure yet


def build_iv_table(chain_df: pd.DataFrame, r: float = RISK_FREE_RATE) -> pd.DataFrame:
    """
    Adds an 'iv' column to the chain by inverting each contract's mid price.
    Drops contracts where the solver returns None (bad/arbitrage-violating quote).
    """
    ivs = []
    for _, row in chain_df.iterrows():
        iv = implied_vol(
            target_price=row["mid"],
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=r,
            option_type=row["option_type"],
        )
        ivs.append(iv)

    out = chain_df.copy()
    out["iv"] = ivs
    return out.dropna(subset=["iv"])


class VolSurface:
    """
    Fitted interpolated vol surface. Call .get_iv(strike, T) to query any
    point on the surface -- lets the risk layer price legs at strikes/expiries
    that weren't directly quoted.
    """

    def __init__(self, iv_table: pd.DataFrame):
        if iv_table.empty:
            raise ValueError("iv_table is empty -- nothing to fit")

        self.expiries = sorted(iv_table["T"].unique())
        if len(self.expiries) < 2:
            raise ValueError("Need at least 2 expiries to interpolate across time")

        self._smiles = {}
        for T in self.expiries:
            slice_df = iv_table[iv_table["T"] == T].sort_values("strike")
            strikes = slice_df["strike"].values
            ivs = slice_df["iv"].values

            strikes_unique, idx = np.unique(strikes, return_index=True)
            if len(strikes_unique) < len(strikes):
                df_tmp = pd.DataFrame({"strike": strikes, "iv": ivs})
                df_tmp = df_tmp.groupby("strike", as_index=False).mean()
                strikes_unique = df_tmp["strike"].values
                ivs = df_tmp["iv"].values

            if len(strikes_unique) < 4:
                continue

            self._smiles[T] = CubicSpline(strikes_unique, ivs, extrapolate=True)

        if not self._smiles:
            raise ValueError("No expiry had enough strikes to fit a smile")

        self._fitted_expiries = sorted(self._smiles.keys())

    def _smile_iv(self, T: float, K: float) -> float:
        return float(self._smiles[T](K))

    def get_iv(self, K: float, T: float) -> float:
        expiries = self._fitted_expiries

        if T <= expiries[0]:
            return self._smile_iv(expiries[0], K)
        if T >= expiries[-1]:
            return self._smile_iv(expiries[-1], K)

        for i in range(len(expiries) - 1):
            T_lo, T_hi = expiries[i], expiries[i + 1]
            if T_lo <= T <= T_hi:
                iv_lo = self._smile_iv(T_lo, K)
                iv_hi = self._smile_iv(T_hi, K)

                var_lo = iv_lo ** 2 * T_lo
                var_hi = iv_hi ** 2 * T_hi

                weight = (T - T_lo) / (T_hi - T_lo)
                var_interp = var_lo + weight * (var_hi - var_lo)

                return float(np.sqrt(var_interp / T))

        raise RuntimeError("Unreachable -- T bracketing failed")

    def grid(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        return np.array([[self.get_iv(K, T) for K in strikes] for T in expiries])


if __name__ == "__main__":
    strikes = np.array([80, 90, 100, 110, 120])
    expiries_days = [30, 60, 90]

    rows = []
    spot = 100
    for T_days in expiries_days:
        T = T_days / 365
        for K in strikes:
            true_iv = 0.2 + 0.05 * abs(K - 100) / 20
            option_type = "call" if K >= spot else "put"
            from ..pricing.black_scholes import price
            mid = price(spot, K, T, RISK_FREE_RATE, true_iv, option_type)
            rows.append({
                "spot": spot, "strike": K, "T": T,
                "option_type": option_type, "mid": mid,
            })

    df = pd.DataFrame(rows)
    iv_table = build_iv_table(df)
    print(f"Recovered IVs for {len(iv_table)}/{len(df)} contracts")

    surface = VolSurface(iv_table)

    recovered = surface.get_iv(K=100, T=30 / 365)
    print(f"ATM 30d IV recovered: {recovered:.4f} (expect ~0.20)")
    assert abs(recovered - 0.20) < 0.01

    mid_expiry_iv = surface.get_iv(K=100, T=45 / 365)
    print(f"ATM 45d (interpolated) IV: {mid_expiry_iv:.4f}")
    assert 0.15 < mid_expiry_iv < 0.25

    print("All vol surface sanity checks passed.")