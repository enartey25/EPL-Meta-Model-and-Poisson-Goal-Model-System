"""Plotly charts in the Ticket Wallet palette (navy ground, stock ink, carbon and jet red series)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui import CARBON, CARBON_COPY, CARBON_ON_NAVY, INK, MUTED_ON_NAVY, NAVY, RED, STOCK

GRID = "rgba(242,240,234,0.14)"
FONT = "Barlow, system-ui, sans-serif"
NUM = "IBM Plex Mono, ui-monospace, monospace"


def _layout(fig, title=None, height=360, legend=True, **kw):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=STOCK, size=13),
        title=dict(text=title, font=dict(family=FONT, size=15, color=STOCK), x=0, xanchor="left") if title else None,
        margin=dict(l=8, r=8, t=44 if title else 12, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=MUTED_ON_NAVY)),
        hoverlabel=dict(bgcolor=INK, bordercolor=STOCK, font=dict(family=NUM, color=STOCK)),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED_ON_NAVY))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED_ON_NAVY))
    return fig


def score_grid(matrix: np.ndarray, home: str, away: str, show: int = 6):
    m = matrix[:show, :show] * 100
    fig = go.Figure(go.Heatmap(
        z=m, x=[str(i) for i in range(show)], y=[str(i) for i in range(show)],
        colorscale=[[0, "#0F2048"], [0.5, "#6E57A8"], [1, CARBON_COPY]], showscale=False,
        text=[[f"{v:.1f}" for v in row] for row in m], texttemplate="%{text}",
        textfont=dict(family=NUM, size=13),
        hovertemplate=f"{home} %{{y}} – %{{x}} {away}<br>%{{z:.1f}}%<extra></extra>", xgap=1, ygap=1))
    _layout(fig, None, height=380, legend=False)
    fig.update_xaxes(title=dict(text=f"{away} goals", font=dict(color=MUTED_ON_NAVY)), side="top", showgrid=False)
    fig.update_yaxes(title=dict(text=f"{home} goals", font=dict(color=MUTED_ON_NAVY)), autorange="reversed", showgrid=False)
    return fig


def cumulative_edge(rec: pd.DataFrame):
    """Running total of (baseline log loss - model log loss) per match: rising = model ahead."""
    y = rec["FTR"].to_numpy().astype(int)
    idx = np.arange(len(y))
    p_model = rec[["P_Draw", "P_Away", "P_Home"]].to_numpy()[idx, y]
    p_null = rec[["Null_Draw", "Null_Away", "Null_Home"]].to_numpy()[idx, y]
    edge = np.cumsum(np.log(p_model) - np.log(p_null))
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED_ON_NAVY, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=np.arange(1, len(edge) + 1), y=edge, mode="lines", name="Model vs baseline",
                             line=dict(color=CARBON_ON_NAVY, width=2.5),
                             hovertemplate="after %{x} matches: %{y:+.2f} nats<extra></extra>"))
    _layout(fig, "Cumulative log-loss edge over the naive baseline (rising = model ahead)", legend=False)
    fig.update_xaxes(title="Matches played")
    fig.update_yaxes(title="Edge (nats)")
    return fig


def calibration(rec: pd.DataFrame, bins: int = 8):
    probs = rec[["P_Draw", "P_Away", "P_Home"]].to_numpy().ravel()
    y = rec["FTR"].to_numpy().astype(int)
    hits = np.column_stack([(y == c) for c in (0, 1, 2)]).ravel().astype(float)
    edges = np.linspace(0, max(0.8, probs.max() + 1e-9), bins + 1)
    which = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"b": which, "p": probs, "h": hits}).groupby("b").agg(p=("p", "mean"), h=("h", "mean"), n=("h", "size"))
    df = df[df["n"] >= 8]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.8], y=[0, 0.8], mode="lines", name="Perfect calibration",
                             line=dict(color=MUTED_ON_NAVY, width=1, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["p"], y=df["h"], mode="lines+markers", name="Model",
                             line=dict(color=CARBON_ON_NAVY, width=2), marker=dict(size=np.sqrt(df["n"]) * 1.6, color=CARBON_ON_NAVY,
                                                                              line=dict(color=NAVY, width=1)),
                             customdata=df["n"], hovertemplate="forecast %{x:.0%} → happened %{y:.0%} (%{customdata} outcomes)<extra></extra>"))
    _layout(fig, "Calibration: forecast probability vs how often it happened")
    fig.update_xaxes(title="Forecast probability", tickformat=".0%", range=[0, 0.8])
    fig.update_yaxes(title="Observed frequency", tickformat=".0%", range=[0, 0.8])
    return fig


def round_losses(rec: pd.DataFrame):
    y = rec["FTR"].to_numpy().astype(int)
    idx = np.arange(len(y))
    r = rec.assign(m=-np.log(rec[["P_Draw", "P_Away", "P_Home"]].to_numpy()[idx, y]),
                   b=-np.log(rec[["Null_Draw", "Null_Away", "Null_Home"]].to_numpy()[idx, y]))
    g = r.groupby("round").agg(m=("m", "mean"), b=("b", "mean"), n=("m", "size")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["round"], y=g["m"], name="Model", marker=dict(color=[CARBON_ON_NAVY if a <= b else RED for a, b in zip(g["m"], g["b"])]),
                         customdata=np.column_stack([g["b"], g["n"]]),
                         hovertemplate="Gameweek %{x}: model %{y:.3f}, baseline %{customdata[0]:.3f} (%{customdata[1]} matches)<extra></extra>"))
    fig.add_trace(go.Scatter(x=g["round"], y=g["b"], mode="markers", name="Baseline",
                             marker=dict(symbol="line-ew", size=18, line=dict(color=STOCK, width=2))))
    _layout(fig, "Log loss by gameweek (red bar = worse than the baseline that week)")
    fig.update_xaxes(title="Gameweek", dtick=1 if len(g) <= 20 else 2)
    fig.update_yaxes(title="Log loss (lower is better)")
    return fig


def elo_ladder(elo: dict, teams: list[str]):
    d = pd.Series({t: elo[t] for t in teams if t in elo}).sort_values()
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker=dict(color=CARBON_ON_NAVY),
                           text=[f"{v:.0f}" for v in d.values], textposition="outside",
                           textfont=dict(family=NUM, color=STOCK), hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
    _layout(fig, "Current Elo ratings", height=max(420, 26 * len(d)), legend=False)
    fig.add_vline(x=1500, line=dict(color=MUTED_ON_NAVY, width=1, dash="dot"))
    fig.update_xaxes(range=[min(1300, d.min() - 30), d.max() + 60], title="Elo")
    return fig


def goals_by_season(df: pd.DataFrame):
    g = df.groupby("Season").agg(hg=("FTHG", "mean"), ag=("FTAG", "mean"), hx=("home_xG", "mean"), ax=("away_xG", "mean")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["Season"], y=g["hg"], name="Home goals", marker_color=STOCK))
    fig.add_trace(go.Bar(x=g["Season"], y=g["ag"], name="Away goals", marker_color=CARBON_ON_NAVY))
    fig.add_trace(go.Scatter(x=g["Season"], y=g["hx"], name="Home xG", mode="markers",
                             marker=dict(symbol="line-ew", size=22, line=dict(color=RED, width=2))))
    fig.add_trace(go.Scatter(x=g["Season"], y=g["ax"], name="Away xG", mode="markers",
                             marker=dict(symbol="line-ew", size=22, line=dict(color=RED, width=2, ))))
    _layout(fig, "Goals and xG per match, by season", barmode="group")
    return fig


def result_split(df: pd.DataFrame):
    g = df.groupby("Season")["FTR"].value_counts(normalize=True).unstack().reindex(columns=["H", "D", "A"]).fillna(0)
    fig = go.Figure()
    for k, name, col in (("H", "Home win", STOCK), ("D", "Draw", MUTED_ON_NAVY), ("A", "Away win", CARBON_ON_NAVY)):
        fig.add_trace(go.Bar(y=g.index, x=g[k], name=name, orientation="h", marker_color=col,
                             text=[f"{v:.0%}" for v in g[k]], textposition="inside", insidetextanchor="middle",
                             textfont=dict(family=NUM, color=INK), hovertemplate=f"%{{y}} {name}: %{{x:.1%}}<extra></extra>"))
    _layout(fig, "Result split by season", barmode="stack", height=60 + 52 * len(g))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return fig


def xg_form(df: pd.DataFrame, team: str):
    h = df[df["HomeTeam"] == team][["Date", "home_xG", "away_xG"]].rename(columns={"home_xG": "f", "away_xG": "a"})
    a = df[df["AwayTeam"] == team][["Date", "away_xG", "home_xG"]].rename(columns={"away_xG": "f", "home_xG": "a"})
    c = pd.concat([h, a]).sort_values("Date")
    if c.empty:
        return None
    c["rf"], c["ra"] = c["f"].rolling(5, min_periods=1).mean(), c["a"].rolling(5, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["Date"], y=c["rf"], name="xG for (5-match)", line=dict(color=CARBON_ON_NAVY, width=2.5)))
    fig.add_trace(go.Scatter(x=c["Date"], y=c["ra"], name="xG against (5-match)", line=dict(color=RED, width=2.5)))
    _layout(fig, f"{team}: rolling xG form")
    fig.update_yaxes(title="xG per match")
    return fig
