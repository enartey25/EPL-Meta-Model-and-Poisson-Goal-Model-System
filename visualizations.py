import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

# ─── Palette ─────────────────────────────────────────────────────────────────
TEAL   = "#00d4aa"
VIOLET = "#7c3aed"
AMBER  = "#f59e0b"
ROSE   = "#f43f5e"
BG     = "#0d1117"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

OUTCOME_COLORS = {"Home Win": TEAL, "Draw": AMBER, "Away Win": ROSE}

_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=44, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


# ─── 1. Result Distribution Donut ────────────────────────────────────────────
def plot_result_distribution(df: pd.DataFrame, season: str = "All Seasons") -> go.Figure:
    if season and season != "All Seasons":
        df = df[df["Season"] == season]
    counts = df["FTR"].value_counts()
    labels_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    labels = [labels_map.get(k, k) for k in counts.index]
    values = counts.values.tolist()
    colors = [OUTCOME_COLORS.get(l, VIOLET) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color=BG, width=3)),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:10px'>Matches</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=TEXT), xref="paper", yref="paper",
    )
    fig.update_layout(
        title=dict(text="Match Result Distribution", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig


# ─── 2. ELO Rankings ─────────────────────────────────────────────────────────
def plot_elo_rankings(elo_dict: dict) -> go.Figure:
    df_elo = pd.DataFrame(list(elo_dict.items()), columns=["Team", "ELO"]).sort_values("ELO")
    n = len(df_elo)
    colors = [TEAL if v >= 1550 else VIOLET if v >= 1500 else MUTED for v in df_elo["ELO"]]

    fig = go.Figure(go.Bar(
        y=df_elo["Team"],
        x=df_elo["ELO"],
        orientation="h",
        marker=dict(color=colors, line=dict(color=BG, width=0.5)),
        text=df_elo["ELO"].round(0).astype(int),
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color=TEXT, size=12, family="Inter"),
        hovertemplate="<b>%{y}</b><br>ELO: %{x:.1f}<extra></extra>",
        width=0.75,
    ))

    min_elo = df_elo["ELO"].min()
    max_elo = df_elo["ELO"].max()

    fig.update_xaxes(
        range=[min_elo - 80, max_elo + 30],
        gridcolor=BORDER,
        color=MUTED,
        tickfont=dict(size=11),
        title="ELO Rating",
    )
    fig.update_yaxes(
        color=TEXT,
        tickfont=dict(size=12),
        automargin=True,
    )
    fig.add_vline(x=1500, line_dash="dash", line_color=MUTED, opacity=0.35)
    fig.update_layout(
        title=dict(text="Current ELO Rankings", font=dict(size=15, color=TEXT), x=0.5),
        height=max(500, n * 30),   # 30px per team, minimum 500
        margin=dict(l=120, r=30, t=44, b=30),
        **{k: v for k, v in _THEME.items() if k != "margin"},
    )
    return fig


# ─── 3. xG Form Trend ────────────────────────────────────────────────────────
def plot_xg_trend(df_xg: pd.DataFrame, team: str) -> go.Figure:
    if df_xg.empty or "home_team" not in df_xg.columns:
        return go.Figure()

    home = df_xg[df_xg["home_team"] == team][["date", "home_xG", "away_xG"]].rename(
        columns={"home_xG": "xG_For", "away_xG": "xG_Against"})
    away = df_xg[df_xg["away_team"] == team][["date", "away_xG", "home_xG"]].rename(
        columns={"away_xG": "xG_For", "home_xG": "xG_Against"})
    combined = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
    if combined.empty:
        return go.Figure()

    combined["Roll_For"]     = combined["xG_For"].rolling(5, min_periods=1).mean()
    combined["Roll_Against"] = combined["xG_Against"].rolling(5, min_periods=1).mean()

    fig = go.Figure()
    for col, color, name in [
        ("xG_For",      TEAL, "xG For (match)"),
        ("xG_Against",  ROSE, "xG Against (match)"),
    ]:
        fig.add_trace(go.Scatter(
            x=combined["date"], y=combined[col],
            mode="markers", name=name,
            marker=dict(color=color, size=5, opacity=0.4),
        ))
    for col, color, name in [
        ("Roll_For",     TEAL, "5-match avg (For)"),
        ("Roll_Against", ROSE, "5-match avg (Against)"),
    ]:
        fig.add_trace(go.Scatter(
            x=combined["date"], y=combined[col],
            mode="lines", name=name,
            line=dict(color=color, width=2.5),
        ))

    fig.update_xaxes(gridcolor=BORDER, color=MUTED, title="Date")
    fig.update_yaxes(gridcolor=BORDER, color=MUTED, title="xG")
    fig.update_layout(
        title=dict(text=f"{team} – xG Form Trend", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig


# ─── 4. Calibrated Scoreline Heatmap ─────────────────────────────────────────
def plot_score_heatmap(matrix: np.ndarray, home_team: str, away_team: str) -> go.Figure:
    n = matrix.shape[0]
    z = (matrix * 100).round(2)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(g) for g in range(n)],
        y=[str(g) for g in range(n)],
        colorscale=[[0, "#0d1117"], [0.4, VIOLET], [1.0, TEAL]],
        hovertemplate=f"<b>{home_team} %{{y}} – %{{x}} {away_team}</b><br>Prob: %{{z:.2f}}%<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color=MUTED), title=dict(text="Prob %", font=dict(color=MUTED))),
    ))
    for i in range(n):
        for j in range(n):
            fig.add_annotation(
                x=str(j), y=str(i),
                text=f"{z[i,j]:.1f}%",
                showarrow=False,
                font=dict(size=9, color=TEXT if z[i, j] > 2 else MUTED),
            )
    fig.update_xaxes(title=f"{away_team} Goals", color=MUTED, gridcolor=BORDER)
    fig.update_yaxes(title=f"{home_team} Goals", color=MUTED, gridcolor=BORDER)
    fig.update_layout(
        title=dict(text="Calibrated Scoreline Heatmap", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig


# ─── 5. Outcome Probability Bars ──────────────────────────────────────────────
def plot_outcome_bars(proba: dict, home_team: str, away_team: str) -> go.Figure:
    ordered = ["Home Win", "Draw", "Away Win"]
    vals    = [proba.get(k, 0) * 100 for k in ordered]
    colors  = [OUTCOME_COLORS[k] for k in ordered]
    labels  = [f"{home_team} Win", "Draw", f"{away_team} Win"]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker=dict(color=colors, line=dict(color=BG, width=1)),
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=14, color=TEXT, family="Inter"),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        width=0.5,
    ))
    fig.update_yaxes(range=[0, max(vals) * 1.3], gridcolor=BORDER, color=MUTED, ticksuffix="%")
    fig.update_xaxes(color=MUTED)
    fig.update_layout(
        title=dict(text="Calibrated Outcome Probabilities", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig


# ─── 6. Poisson Goal Distribution ─────────────────────────────────────────────
def plot_poisson_distribution(h_pmf: np.ndarray, a_pmf: np.ndarray,
                               home_team: str, away_team: str) -> go.Figure:
    goals = np.arange(len(h_pmf))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=goals, y=h_pmf * 100, name=home_team,
        marker_color=TEAL, opacity=0.85,
        hovertemplate=f"<b>{home_team}</b> scores %{{x}}: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=goals, y=a_pmf * 100, name=away_team,
        marker_color=ROSE, opacity=0.85,
        hovertemplate=f"<b>{away_team}</b> scores %{{x}}: %{{y:.1f}}%<extra></extra>",
    ))
    fig.update_xaxes(title="Goals Scored", tickvals=list(goals), color=MUTED, gridcolor=BORDER)
    fig.update_yaxes(title="Probability (%)", gridcolor=BORDER, color=MUTED, ticksuffix="%")
    fig.update_layout(
        barmode="group",
        title=dict(text="Poisson Goal Distribution", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig


# ─── 7. Average Goals per Season ──────────────────────────────────────────────
def plot_goals_per_season(df: pd.DataFrame) -> go.Figure:
    if df.empty or "FTHG" not in df.columns:
        return go.Figure()
    g = df.groupby("Season").agg(
        HomeGoals=("FTHG", "mean"), AwayGoals=("FTAG", "mean"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["Season"], y=g["HomeGoals"], name="Avg Home Goals",
                         marker_color=TEAL, opacity=0.85))
    fig.add_trace(go.Bar(x=g["Season"], y=g["AwayGoals"], name="Avg Away Goals",
                         marker_color=ROSE, opacity=0.85))
    fig.update_xaxes(color=MUTED, gridcolor=BORDER)
    fig.update_yaxes(color=MUTED, gridcolor=BORDER, title="Avg Goals per Match")
    fig.update_layout(
        barmode="group",
        title=dict(text="Average Goals per Season", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig


# ─── 8. Head-to-Head Results ──────────────────────────────────────────────────
def plot_h2h(df: pd.DataFrame, team1: str, team2: str) -> go.Figure:
    mask = (
        ((df["HomeTeam"] == team1) & (df["AwayTeam"] == team2)) |
        ((df["HomeTeam"] == team2) & (df["AwayTeam"] == team1))
    )
    h2h = df[mask].copy()
    if h2h.empty:
        fig = go.Figure()
        fig.add_annotation(text="No head-to-head matches found", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=MUTED, size=14),
                           xref="paper", yref="paper")
        fig.update_layout(**_THEME)
        return fig

    t1_w, draws, t2_w = 0, 0, 0
    for _, row in h2h.iterrows():
        if row["HomeTeam"] == team1:
            if row["FTR"] == "H":   t1_w  += 1
            elif row["FTR"] == "D": draws  += 1
            else:                   t2_w  += 1
        else:
            if row["FTR"] == "A":   t1_w  += 1
            elif row["FTR"] == "D": draws  += 1
            else:                   t2_w  += 1

    fig = go.Figure(go.Bar(
        x=[f"{team1} Wins", "Draws", f"{team2} Wins"],
        y=[t1_w, draws, t2_w],
        marker=dict(color=[TEAL, AMBER, ROSE]),
        text=[t1_w, draws, t2_w], textposition="outside",
        textfont=dict(color=TEXT, size=13), width=0.45,
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_yaxes(gridcolor=BORDER, color=MUTED)
    fig.update_xaxes(color=MUTED)
    fig.update_layout(
        title=dict(text=f"Head-to-Head: {team1} vs {team2}", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig


# ─── 9. Team Performance Radar ────────────────────────────────────────────────
def plot_team_radar(df: pd.DataFrame, teams: list, season: str = "All Seasons") -> go.Figure:
    if season != "All Seasons":
        df = df[df["Season"] == season]
    if df.empty or "FTHG" not in df.columns:
        return go.Figure()

    categories = ["Goals Scored", "Goals Conceded", "Shot Accuracy", "Corners", "Cards"]
    colors = [TEAL, ROSE, AMBER, VIOLET]

    def _stats(team):
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]
        n = len(home) + len(away)
        if n == 0: return [0] * 5
        gf = (home["FTHG"].sum() + away["FTAG"].sum()) / n
        ga = (home["FTAG"].sum() + away["FTHG"].sum()) / n
        if "HS" in df.columns:
            sa = ((home.get("HST", pd.Series(dtype=float)).sum() +
                   away.get("AST", pd.Series(dtype=float)).sum()) /
                  max(home.get("HS", pd.Series(dtype=float)).sum() +
                      away.get("AS", pd.Series(dtype=float)).sum(), 1))
        else: sa = 0.33
        corners = ((home.get("HC", pd.Series(dtype=float)).sum() +
                    away.get("AC", pd.Series(dtype=float)).sum()) / n) if "HC" in df.columns else 5.0
        cards = ((home.get("HY", pd.Series(dtype=float)).sum() +
                  away.get("AY", pd.Series(dtype=float)).sum()) / n) if "HY" in df.columns else 1.5
        return [gf, ga, sa * 3, corners / 3, cards]

    fig = go.Figure()
    for i, team in enumerate(teams[:4]):
        vals = _stats(team)
        v_c  = vals + [vals[0]]
        c_c  = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=v_c, theta=c_c, name=team, fill="toself",
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=colors[i % len(colors)], opacity=0.2,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, color=MUTED, gridcolor=BORDER),
            angularaxis=dict(color=MUTED, gridcolor=BORDER),
        ),
        title=dict(text="Team Performance Radar", font=dict(size=15, color=TEXT), x=0.5),
        **_THEME,
    )
    return fig
