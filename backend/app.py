"""
Dash risk screen: 3D vol surface + multi-leg position P&L heatmap.

Layout:
    - Ticker input -> pulls live chain, fits surface
    - 3D surface plot: strike x expiry x IV
    - Position builder: up to 4 legs (strike, expiry, call/put, quantity)
    - P&L heatmap: spot shift x vol shift, for the built position

Run with: python3 -m backend.app
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, ctx

from .data.chain_fetcher import fetch_chain
from .surface.interpolate import build_iv_table, VolSurface, RISK_FREE_RATE
from .surface.sabr import SABRSurface
from .risk.position import Position, Leg
from .risk.scenario import scenario_pnl_grid

app = Dash(__name__)

# ---- Layout ----------------------------------------------------------------

def leg_row(i):
    return html.Div([
        html.Span(f"LEG {i+1}", className="leg-label"),
        dcc.Input(id=f"strike-{i}", type="number", placeholder="Strike", style={"width": "90px"}),
        dcc.Input(id=f"expiry-days-{i}", type="number", placeholder="Days", style={"width": "90px"}),
        dcc.Dropdown(
            id=f"type-{i}", options=["call", "put"], value="call",
            style={"width": "100px", "display": "inline-block"}, clearable=False,
        ),
        dcc.Input(id=f"qty-{i}", type="number", placeholder="Qty (+/-)", value=0, style={"width": "90px"}),
    ], className="leg-row")


app.layout = html.Div([
    html.H2("VOL SURFACE & GREEKS RISK ENGINE"),

    html.Div([
        dcc.Input(id="ticker-input", type="text", value="SPY", placeholder="TICKER"),
        dcc.Dropdown(
            id="method-dropdown",
            options=[
                {"label": "Interpolation (cubic spline)", "value": "interp"},
                {"label": "SABR", "value": "sabr"},
            ],
            value="interp", clearable=False,
            style={"width": "260px", "display": "inline-block"},
        ),
        html.Button("Load Surface", id="load-button", n_clicks=0),
        html.Span(id="load-status"),
    ], className="ticker-bar"),

    dcc.Graph(id="vol-surface-plot"),

    html.H3("Position Builder"),
    html.Div([leg_row(i) for i in range(4)]),
    html.Button("Compute Risk", id="compute-button", n_clicks=0, style={"marginTop": "8px"}),

    html.Div(id="greeks-summary", style={"marginTop": "20px"}),

    html.H3("Scenario P&L"),
    dcc.Graph(id="pnl-heatmap"),

    dcc.Store(id="surface-store"),
])


# ---- Callbacks ---------------------------------------------------------------

@app.callback(
    Output("vol-surface-plot", "figure"),
    Output("load-status", "children"),
    Output("surface-store", "data"),
    Input("load-button", "n_clicks"),
    State("ticker-input", "value"),
    State("method-dropdown", "value"),
    prevent_initial_call=False,
)
def load_surface(n_clicks, ticker, method):
    try:
        chain = fetch_chain(ticker, max_expiries=8)
        iv_table = build_iv_table(chain)

        if method == "sabr":
            surface = SABRSurface(iv_table, r=RISK_FREE_RATE)
        else:
            surface = VolSurface(iv_table)

        spot = float(chain["spot"].iloc[0])
        strikes = np.linspace(spot * 0.8, spot * 1.2, 30)
        expiries = np.array(surface.expiries)

        iv_grid = surface.grid(strikes, expiries)  # shape (len(expiries), len(strikes))

        method_label = "SABR" if method == "sabr" else "INTERPOLATED"
        fig = go.Figure(data=[go.Surface(
            x=strikes, y=expiries * 365, z=iv_grid,
            colorscale=[[0, "#1a1f2c"], [0.5, "#f0b90b"], [1, "#26d07c"]],
            colorbar=dict(title="IV", tickfont=dict(color="#7d8598"), title_font=dict(color="#7d8598")),
        )])
        fig.update_layout(
            title=dict(text=f"{ticker.upper()} {method_label} VOL SURFACE — SPOT {spot:.2f}",
                       font=dict(family="JetBrains Mono", color="#e6e9ef", size=14)),
            scene=dict(
                xaxis=dict(title="Strike", color="#7d8598", gridcolor="#232838", backgroundcolor="#12161f"),
                yaxis=dict(title="Days to Expiry", color="#7d8598", gridcolor="#232838", backgroundcolor="#12161f"),
                zaxis=dict(title="Implied Vol", color="#7d8598", gridcolor="#232838", backgroundcolor="#12161f"),
            ),
            paper_bgcolor="#12161f",
            font=dict(family="JetBrains Mono", color="#e6e9ef"),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        # Store what downstream callbacks need: iv_table as records + spot + method
        store_data = {"iv_table": iv_table.to_dict("records"), "spot": spot, "method": method}
        return fig, f"Loaded {len(iv_table)} contracts for {ticker.upper()} ({method_label})", store_data

    except Exception as e:
        return go.Figure(), f"Error loading {ticker}: {e}", None


@app.callback(
    Output("greeks-summary", "children"),
    Output("pnl-heatmap", "figure"),
    Input("compute-button", "n_clicks"),
    State("surface-store", "data"),
    [State(f"strike-{i}", "value") for i in range(4)],
    [State(f"expiry-days-{i}", "value") for i in range(4)],
    [State(f"type-{i}", "value") for i in range(4)],
    [State(f"qty-{i}", "value") for i in range(4)],
    prevent_initial_call=True,
)
def compute_risk(n_clicks, store_data, s0, s1, s2, s3, e0, e1, e2, e3,
                  t0, t1, t2, t3, q0, q1, q2, q3):
    if store_data is None:
        return "Load a surface first.", go.Figure()

    import pandas as pd
    iv_table = pd.DataFrame(store_data["iv_table"])
    spot = store_data["spot"]
    method = store_data.get("method", "interp")

    if method == "sabr":
        surface = SABRSurface(iv_table, r=RISK_FREE_RATE)
    else:
        surface = VolSurface(iv_table)

    strikes = [s0, s1, s2, s3]
    days = [e0, e1, e2, e3]
    types = [t0, t1, t2, t3]
    qtys = [q0, q1, q2, q3]

    legs = []
    for strike, day, opt_type, qty in zip(strikes, days, types, qtys):
        if strike and day and qty:
            legs.append(Leg(strike=strike, T=day / 365, option_type=opt_type, quantity=qty))

    if not legs:
        return "Enter at least one leg with strike, expiry, and non-zero quantity.", go.Figure()

    position = Position(legs)
    greeks = position.aggregate_greeks(spot, RISK_FREE_RATE, surface)

    price_color = "#26d07c" if greeks["price"] >= 0 else "#ff5c5c"
    delta_color = "#26d07c" if greeks["delta"] >= 0 else "#ff5c5c"

    summary = html.Div([
        html.P([
            html.Span("PRICE  ", style={"color": "#7d8598"}),
            html.Span(f"{greeks['price']:.4f}", style={"color": price_color, "fontWeight": "600"}),
            html.Span("     DELTA  ", style={"color": "#7d8598"}),
            html.Span(f"{greeks['delta']:.4f}", style={"color": delta_color, "fontWeight": "600"}),
            html.Span("     GAMMA  ", style={"color": "#7d8598"}),
            html.Span(f"{greeks['gamma']:.4f}"),
        ]),
        html.P([
            html.Span("VEGA   ", style={"color": "#7d8598"}),
            html.Span(f"{greeks['vega']:.4f}"),
            html.Span("     THETA  ", style={"color": "#7d8598"}),
            html.Span(f"{greeks['theta']:.4f}", style={"color": "#ff5c5c" if greeks['theta'] < 0 else "#26d07c"}),
            html.Span("     RHO    ", style={"color": "#7d8598"}),
            html.Span(f"{greeks['rho']:.4f}"),
        ]),
    ])

    spot_shifts = np.linspace(-spot * 0.15, spot * 0.15, 25)
    vol_shifts = np.linspace(-0.10, 0.10, 15)

    pnl = scenario_pnl_grid(position, spot, RISK_FREE_RATE, surface, spot_shifts, vol_shifts)

    heatmap = go.Figure(data=go.Heatmap(
        x=spot + spot_shifts, y=vol_shifts,
        z=pnl, colorscale=[[0, "#ff5c5c"], [0.5, "#12161f"], [1, "#26d07c"]], zmid=0,
        colorbar=dict(title="P&L", tickfont=dict(color="#7d8598"), title_font=dict(color="#7d8598")),
    ))
    heatmap.update_layout(
        title=dict(text="SCENARIO P&L — SPOT VS VOL SHIFT",
                   font=dict(family="JetBrains Mono", color="#e6e9ef", size=14)),
        xaxis=dict(title="Spot", color="#7d8598", gridcolor="#232838"),
        yaxis=dict(title="Vol shift", color="#7d8598", gridcolor="#232838"),
        paper_bgcolor="#12161f",
        plot_bgcolor="#12161f",
        font=dict(family="JetBrains Mono", color="#e6e9ef"),
        height=500,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return summary, heatmap


if __name__ == "__main__":
    app.run(debug=True)