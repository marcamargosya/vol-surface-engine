"""
Scenario P&L grid: reprice a position as spot and vol shift.

For each (spot_shift, vol_shift) pair, reprices every leg using the shifted
spot and a vol surface where every point is bumped by vol_shift (a parallel
shift of the whole surface -- simple and standard for a first-pass risk
screen; smile-consistent vol shocks are a possible extension later).

P&L is measured relative to the position's current value (spot_shift=0,
vol_shift=0), so the grid shows profit/loss from today, not absolute price.
"""

import numpy as np

from ..pricing.black_scholes import price


class ShiftedSurface:
    """Wraps a VolSurface and adds a constant vol shift to every query."""
    def __init__(self, base_surface, vol_shift: float):
        self.base_surface = base_surface
        self.vol_shift = vol_shift

    def get_iv(self, K, T):
        return max(self.base_surface.get_iv(K, T) + self.vol_shift, 1e-4)


def position_value(position, S: float, r: float, vol_surface, q: float = 0.0) -> float:
    total = 0.0
    for leg in position.legs:
        sigma = vol_surface.get_iv(leg.strike, leg.T)
        total += leg.quantity * price(S, leg.strike, leg.T, r, sigma, leg.option_type, q)
    return total


def scenario_pnl_grid(position, S0: float, r: float, vol_surface,
                       spot_shifts: np.ndarray, vol_shifts: np.ndarray,
                       q: float = 0.0) -> np.ndarray:
    base_value = position_value(position, S0, r, vol_surface, q)

    pnl = np.zeros((len(vol_shifts), len(spot_shifts)))
    for i, dvol in enumerate(vol_shifts):
        shifted_surface = ShiftedSurface(vol_surface, dvol)
        for j, dspot in enumerate(spot_shifts):
            S_scenario = S0 + dspot
            if S_scenario <= 0:
                pnl[i, j] = np.nan
                continue
            scenario_value = position_value(position, S_scenario, r, shifted_surface, q)
            pnl[i, j] = scenario_value - base_value

    return pnl


if __name__ == "__main__":
    from .position import Position, Leg

    class _FlatSurface:
        def get_iv(self, K, T):
            return 0.20

    surface = _FlatSurface()
    S0, r = 100, 0.03

    straddle = Position([
        Leg(strike=100, T=0.25, option_type="call", quantity=1),
        Leg(strike=100, T=0.25, option_type="put", quantity=1),
    ])

    spot_shifts = np.array([-10, -5, 0, 5, 10])
    vol_shifts = np.array([-0.05, 0.0, 0.05])

    grid = scenario_pnl_grid(straddle, S0, r, surface, spot_shifts, vol_shifts)
    print("P&L grid (rows=vol shifts, cols=spot shifts):")
    print(grid.round(3))

    zero_spot_idx = list(spot_shifts).index(0)
    zero_vol_idx = list(vol_shifts).index(0.0)
    assert abs(grid[zero_vol_idx, zero_spot_idx]) < 1e-8, "P&L at (0,0) should be 0"

    assert grid[zero_vol_idx, 0] > 0, "Large down move should profit a straddle"
    assert grid[zero_vol_idx, -1] > 0, "Large up move should profit a straddle"

    assert grid[-1, zero_spot_idx] > grid[0, zero_spot_idx], \
        "Straddle P&L should increase with vol (long vega)"

    print("All scenario P&L grid sanity checks passed.")