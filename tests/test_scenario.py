"""
Pytest suite for the scenario P&L grid.

Run with: pytest tests/test_scenario.py -v
"""

import numpy as np

from backend.risk.position import Position, Leg
from backend.risk.scenario import scenario_pnl_grid, position_value, ShiftedSurface


class _FlatSurface:
    """Stub surface with constant IV, so tests don't depend on real market data."""
    def get_iv(self, K, T):
        return 0.20


def test_pnl_is_zero_at_zero_shift():
    # P&L is measured relative to today's value, so at spot_shift=0 and
    # vol_shift=0 the P&L must be exactly 0 by construction.
    surface = _FlatSurface()
    S0, r = 100, 0.03

    straddle = Position([
        Leg(strike=100, T=0.25, option_type="call", quantity=1),
        Leg(strike=100, T=0.25, option_type="put", quantity=1),
    ])

    spot_shifts = np.array([-10, -5, 0, 5, 10])
    vol_shifts = np.array([-0.05, 0.0, 0.05])

    grid = scenario_pnl_grid(straddle, S0, r, surface, spot_shifts, vol_shifts)

    zero_spot_idx = list(spot_shifts).index(0)
    zero_vol_idx = list(vol_shifts).index(0.0)

    assert abs(grid[zero_vol_idx, zero_spot_idx]) < 1e-8


def test_straddle_profits_from_large_moves_either_direction():
    surface = _FlatSurface()
    S0, r = 100, 0.03

    straddle = Position([
        Leg(strike=100, T=0.25, option_type="call", quantity=1),
        Leg(strike=100, T=0.25, option_type="put", quantity=1),
    ])

    spot_shifts = np.array([-10, -5, 0, 5, 10])
    vol_shifts = np.array([0.0])

    grid = scenario_pnl_grid(straddle, S0, r, surface, spot_shifts, vol_shifts)

    # Large moves either direction should profit a long straddle
    assert grid[0, 0] > 0   # down 10
    assert grid[0, -1] > 0  # up 10


def test_straddle_pnl_increases_with_vol():
    # A long straddle is long vega -- higher vol should always increase its
    # value (and therefore P&L) at a fixed spot.
    surface = _FlatSurface()
    S0, r = 100, 0.03

    straddle = Position([
        Leg(strike=100, T=0.25, option_type="call", quantity=1),
        Leg(strike=100, T=0.25, option_type="put", quantity=1),
    ])

    spot_shifts = np.array([0])
    vol_shifts = np.array([-0.05, 0.0, 0.05])

    grid = scenario_pnl_grid(straddle, S0, r, surface, spot_shifts, vol_shifts)

    assert grid[0, 0] < grid[1, 0] < grid[2, 0]


def test_negative_spot_scenario_returns_nan():
    # Spot can never go negative -- a large enough down-shift should be
    # marked undefined (NaN), not silently produce a nonsense price.
    surface = _FlatSurface()
    S0, r = 100, 0.03

    call = Position([Leg(strike=100, T=0.25, option_type="call", quantity=1)])

    spot_shifts = np.array([-150])  # would make spot = -50
    vol_shifts = np.array([0.0])

    grid = scenario_pnl_grid(call, S0, r, surface, spot_shifts, vol_shifts)
    assert np.isnan(grid[0, 0])


def test_shifted_surface_floors_at_small_positive_vol():
    # A large negative vol shift shouldn't be able to push IV to zero or
    # negative -- it should floor at a small positive number instead.
    base_surface = _FlatSurface()
    shifted = ShiftedSurface(base_surface, vol_shift=-0.50)  # base is 0.20, so this would go negative

    iv = shifted.get_iv(K=100, T=0.25)
    assert iv > 0


def test_position_value_matches_manual_sum():
    from backend.pricing.black_scholes import price

    surface = _FlatSurface()
    S, r = 100, 0.03

    spread = Position([
        Leg(strike=100, T=0.5, option_type="call", quantity=1),
        Leg(strike=110, T=0.5, option_type="call", quantity=-1),
    ])

    value = position_value(spread, S, r, surface)

    expected = (
        price(S, 100, 0.5, r, 0.20, "call")
        - price(S, 110, 0.5, r, 0.20, "call")
    )
    assert abs(value - expected) < 1e-8