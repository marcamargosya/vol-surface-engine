# Vol Surface & Greeks Risk Engine

Live options risk engine: pulls a real options chain, inverts prices to implied
vol (never trusts the vendor's own IV), fits a vol surface (cubic-spline
interpolation or SABR), and shows aggregate Greeks + scenario P&L for a
multi-leg position as spot and vol move.

Black-Scholes pricing/Greeks and the IV solver (Newton-Raphson + bisection
fallback) are implemented from scratch, not via a library.

European options only, one underlying, no dividends, flat risk-free rate.

## Structure

- `backend/pricing/` — Black-Scholes + Greeks, IV solver
- `backend/data/` — live options chain via yfinance
- `backend/surface/` — cubic-spline interpolation, SABR calibration
- `backend/risk/` — multi-leg Greeks aggregation, spot x vol scenario grid
- `backend/app.py` — Dash risk screen

## Running it

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m backend.app
\`\`\`

Open `http://127.0.0.1:8050`, enter a ticker, pick a surface method, click
Load Surface. Build a position and click Compute Risk for Greeks + P&L.
