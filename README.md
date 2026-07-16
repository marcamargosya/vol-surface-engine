# Vol Surface & Greeks Risk Engine

A live options risk engine: pulls a real options chain, inverts market prices
to implied vol, fits a volatility surface (interpolation or SABR), lets you
build a multi-leg position, and shows aggregate Greeks plus a scenario P&L
grid as spot and vol move — the kind of screen a vol trading desk actually
uses.

## What it does

1. **Data** — Pulls a live options chain (calls/puts, all strikes/expiries)
   for one underlying via `yfinance`. Only bid/ask/strike/expiry are trusted
   from the source; implied vol is never taken from the API, it's computed
   from scratch (see below).
2. **Pricing & Greeks** — Black-Scholes price, delta, gamma, vega, theta, rho,
   implemented from the closed-form solution (no external pricing library).
3. **Implied vol solver** — Newton-Raphson (fast) with a bisection fallback
   (guaranteed convergence) to invert each contract's mid price into an IV,
   rather than trusting the exchange/vendor's own IV figure.
4. **Vol surface** — Two interchangeable fitting methods:
   - **Interpolation**: cubic spline across strikes per expiry, linear in
     total variance across time.
   - **SABR**: Hagan et al.'s closed-form SABR approximation, calibrated
     per-expiry via least squares (alpha, rho, nu fit; beta fixed at 1.0 for
     an equity-like backbone).
5. **Position risk** — Build a multi-leg position (e.g. a call spread),
   aggregate Greeks across legs, and reprice the whole position across a
   spot x vol scenario grid to see where it makes or loses money.
6. **Visualization** — A Dash app: 3D implied vol surface, a position
   builder, and a P&L heatmap, styled as a dark trading-terminal screen.

## Architecture
backend/
├── pricing/
│   ├── black_scholes.py   # price + Greeks, derived from scratch
│   └── iv_solver.py       # Newton-Raphson + bisection fallback
├── data/
│   └── chain_fetcher.py   # live options chain via yfinance
├── surface/
│   ├── interpolate.py     # cubic spline surface fit
│   └── sabr.py            # SABR model + calibration
├── risk/
│   ├── position.py        # multi-leg Greeks aggregation
│   └── scenario.py        # spot x vol P&L grid
└── app.py                 # Dash risk screen tying it all together

## Scope

European options, one underlying at a time, no dividends, flat risk-free
rate. This is a deliberate choice: a small, fully correct engine is more
defensible than a larger one with hidden gaps (American exercise, dividend
schedules, and multi-underlying support are natural extensions, not
oversights).

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m backend.app
```

Then open `http://127.0.0.1:8050` in your browser. Enter a ticker (e.g.
`SPY`), pick a surface method (Interpolation or SABR), and click
**Load Surface**. Once it loads, build a position in the Position Builder
and click **Compute Risk** to see aggregate Greeks and the scenario P&L
heatmap.

## Testing individual components

Each module has a standalone sanity check runnable on its own:

```bash
python3 -m backend.pricing.black_scholes
python3 -m backend.pricing.iv_solver
python3 -m backend.data.chain_fetcher
python3 -m backend.surface.interpolate
python3 -m backend.surface.sabr
python3 -m backend.risk.position
python3 -m backend.risk.scenario
```