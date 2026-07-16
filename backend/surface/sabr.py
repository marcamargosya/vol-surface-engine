"""
SABR stochastic vol model, fit per expiry.

SABR params: alpha (vol level), beta (backbone, fixed not fit -- see below),
rho (spot-vol correlation, drives skew), nu (vol-of-vol, drives smile curvature).

beta is fixed rather than fit because alpha and beta are poorly identified
together from a single smile (many (alpha, beta) pairs fit similarly well) --
standard practice is to fix beta from convention (1.0 = lognormal/equity-like,
0.5 = CIR-like/rates-like) and calibrate (alpha, rho, nu) to match the smile.
Using beta=1.0 here since this project targets equity indices.

Uses Hagan et al. (2002) lognormal implied vol approximation -- the standard
closed-form SABR approximation, not the full SDE solution (which has no
simple closed form). This is what's used industry-wide for smile fitting.
"""

import numpy as np
from scipy.optimize import least_squares


def sabr_implied_vol(F: float, K: float, T: float, alpha: float, beta: float,
                      rho: float, nu: float) -> float:
    """
    Hagan's SABR lognormal vol approximation. F = forward price, K = strike.
    """
    if abs(F - K) < 1e-12:
        term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2 / F ** (2 - 2 * beta))
        term2 = (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
        term3 = ((2 - 3 * rho ** 2) / 24) * nu ** 2
        return (alpha / F ** (1 - beta)) * (1 + (term1 + term2 + term3) * T)

    logFK = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)

    z = (nu / alpha) * FK_beta * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    term1 = ((1 - beta) ** 2 / 24) * (logFK ** 2)
    term2 = ((1 - beta) ** 4 / 1920) * (logFK ** 4)
    denom_series = 1 + term1 + term2

    numer_correction_1 = ((1 - beta) ** 2 / 24) * (alpha ** 2 / FK_beta ** 2)
    numer_correction_2 = (rho * beta * nu * alpha) / (4 * FK_beta)
    numer_correction_3 = ((2 - 3 * rho ** 2) / 24) * nu ** 2
    numer_correction = 1 + (numer_correction_1 + numer_correction_2 + numer_correction_3) * T

    prefactor = alpha / (FK_beta * denom_series)
    z_over_xz = z / x_z if abs(z) > 1e-12 else 1.0

    return prefactor * z_over_xz * numer_correction


def calibrate_sabr(strikes: np.ndarray, market_ivs: np.ndarray, F: float, T: float,
                    beta: float = 1.0):
    """
    Fit (alpha, rho, nu) to match market_ivs at given strikes via least squares.
    Returns (alpha, beta, rho, nu).
    """
    def residuals(params):
        alpha, rho, nu = params
        alpha = max(alpha, 1e-4)
        rho = np.clip(rho, -0.999, 0.999)
        nu = max(nu, 1e-4)

        model_ivs = np.array([
            sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
        ])
        return model_ivs - market_ivs

    atm_iv_guess = float(np.interp(F, strikes, market_ivs))
    x0 = [atm_iv_guess * F ** (1 - beta), 0.0, 0.5]

    result = least_squares(
        residuals, x0,
        bounds=([1e-4, -0.999, 1e-4], [5.0, 0.999, 5.0]),
    )

    alpha, rho, nu = result.x
    return alpha, beta, rho, nu


if __name__ == "__main__":
    true_alpha, true_beta, true_rho, true_nu = 0.3, 1.0, -0.3, 0.6
    F, T = 100.0, 0.5

    strikes = np.array([80, 90, 95, 100, 105, 110, 120], dtype=float)
    true_ivs = np.array([
        sabr_implied_vol(F, K, T, true_alpha, true_beta, true_rho, true_nu)
        for K in strikes
    ])
    print("True IVs:", true_ivs.round(4))

    alpha, beta, rho, nu = calibrate_sabr(strikes, true_ivs, F, T, beta=true_beta)
    print(f"Recovered params: alpha={alpha:.4f}, rho={rho:.4f}, nu={nu:.4f}")
    print(f"True params:      alpha={true_alpha:.4f}, rho={true_rho:.4f}, nu={true_nu:.4f}")

    fitted_ivs = np.array([
        sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
    ])
    max_error = np.max(np.abs(fitted_ivs - true_ivs))
    print(f"Max IV fitting error: {max_error:.6f}")
    assert max_error < 1e-3, "SABR calibration failed to recover the smile accurately"

    print("All SABR sanity checks passed.")
class SABRSurface:
    """
    SABR-fitted vol surface -- same query interface as VolSurface (get_iv,
    grid) so the risk/app layers can swap between interpolation and SABR
    without changing any calling code.

    Fits one SABR smile per expiry (on the forward F = S * exp(r*T), since
    SABR is defined in forward terms), then interpolates across expiries the
    same way VolSurface does: linear in total variance, not linear in vol.
    """

    def __init__(self, iv_table, r: float, beta: float = 1.0):
        self.r = r
        self.beta = beta
        self.expiries = sorted(iv_table["T"].unique())
        if len(self.expiries) < 2:
            raise ValueError("Need at least 2 expiries to interpolate across time")

        self._params = {}  # T -> (alpha, beta, rho, nu, F)
        for T in self.expiries:
            slice_df = iv_table[iv_table["T"] == T].sort_values("strike")
            strikes = slice_df["strike"].values
            ivs = slice_df["iv"].values
            spot = slice_df["spot"].iloc[0] if "spot" in slice_df else None

            if len(strikes) < 4:
                continue

            F = spot * np.exp(r * T) if spot is not None else np.median(strikes)
            alpha, beta_out, rho, nu = calibrate_sabr(strikes, ivs, F, T, beta=beta)
            self._params[T] = (alpha, beta_out, rho, nu, F)

        if not self._params:
            raise ValueError("No expiry had enough strikes to fit SABR")

        self._fitted_expiries = sorted(self._params.keys())

    def _smile_iv(self, T: float, K: float) -> float:
        alpha, beta, rho, nu, F = self._params[T]
        return sabr_implied_vol(F, K, T, alpha, beta, rho, nu)

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