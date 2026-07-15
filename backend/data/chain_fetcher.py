"""
Pulls a live options chain for one underlying using yfinance (free, no API key).

Returns a clean pandas DataFrame with one row per contract:
    expiry, strike, option_type, bid, ask, mid, spot, T (time to expiry in years)

Deliberately does NOT compute IV here — that's the pricing layer's job
(inverting mid price -> IV via iv_solver, which is "more defensible" than
trusting yfinance's own impliedVolatility column, per the project brief).
"""

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def fetch_chain(ticker: str, max_expiries: int = 6) -> pd.DataFrame:
    """
    Fetch calls + puts across the first `max_expiries` expiries for `ticker`.
    """
    t = yf.Ticker(ticker)

    spot = t.fast_info["last_price"]
    if spot is None:
        raise ValueError(f"Could not fetch spot price for {ticker}")

    expiries = t.options
    if not expiries:
        raise ValueError(f"No options data available for {ticker}")

    expiries = expiries[:max_expiries]
    now = datetime.now(timezone.utc)

    rows = []
    for expiry_str in expiries:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        T = (expiry_date - now).days / 365.0
        if T <= 0:
            continue

        chain = t.option_chain(expiry_str)

        for option_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                bid, ask = row["bid"], row["ask"]
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue

                rows.append({
                    "expiry": expiry_str,
                    "T": T,
                    "strike": row["strike"],
                    "option_type": option_type,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "spot": spot,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_chain("SPY", max_expiries=3)
    print(df.head(10))
    print(f"\nTotal contracts: {len(df)}")
    print(f"Expiries: {sorted(df['expiry'].unique())}")
    print(f"Spot: {df['spot'].iloc[0]}")