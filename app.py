import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    box-sizing: border-box;
}

.stApp { background: #080b10; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
    background: #080b10;
    border-right: 1px solid #1c2128;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p { color: #6e7681; font-size: 12px; }

[data-testid="stMetric"] {
    background: #0e1117;
    border: 1px solid #1c2128;
    border-top: 2px solid #00d4aa;
    border-radius: 0;
    padding: 18px 20px;
}
[data-testid="stMetricLabel"] {
    color: #6e7681 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

.stButton > button {
    background: #e6edf3;
    color: #080b10;
    border: none;
    border-radius: 0;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 13px;
    padding: 14px 28px;
    width: 100%;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: background 0.15s, color 0.15s;
}
.stButton > button:hover { background: #00d4aa; color: #080b10; }

.stSelectbox > div > div {
    background: #0e1117 !important;
    border: 1px solid #1c2128 !important;
    border-radius: 0 !important;
    color: #e6edf3 !important;
    font-family: 'Outfit', sans-serif !important;
}

.stDataFrame { border: 1px solid #1c2128; border-radius: 0; overflow: hidden; }
.stAlert { border-radius: 0; }
hr { border-color: #1c2128; margin: 28px 0; }

/* ── Hero ── */
.hero-eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 4px;
    color: #00d4aa;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: #e6edf3;
    line-height: 1.0;
    margin-bottom: 10px;
    letter-spacing: -1px;
}
.hero-title em { color: #00d4aa; font-style: normal; }
.hero-sub {
    color: #6e7681;
    font-size: 0.95rem;
    font-weight: 400;
    margin-bottom: 32px;
    line-height: 1.6;
    max-width: 680px;
}

/* ── Section labels ── */
.sec-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    color: #6e7681;
    text-transform: uppercase;
    padding-bottom: 10px;
    border-bottom: 1px solid #1c2128;
    margin: 32px 0 16px;
}

/* ── Chart wrapper ── */
.chart-wrap {
    background: #0e1117;
    border: 1px solid #1c2128;
    border-radius: 0;
    padding: 16px 12px 8px;
    margin-bottom: 16px;
}

/* ── Versus board ── */
.versus-board {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0;
    border: 1px solid #1c2128;
    margin: 20px 0 28px;
    background: #0e1117;
}
.versus-team {
    padding: 28px 24px;
    text-align: center;
}
.versus-team-name {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.versus-team-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6e7681;
}
.versus-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    border-left: 1px solid #1c2128;
    border-right: 1px solid #1c2128;
    padding: 0 28px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: #1c2128;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Prediction result block ── */
.result-block {
    border: 1px solid #1c2128;
    border-left-width: 4px;
    padding: 24px 28px;
    margin: 0 0 24px;
    background: #0e1117;
}
.result-verdict {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #6e7681;
    margin-bottom: 8px;
}
.result-outcome {
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -1px;
    margin-bottom: 4px;
}
.result-confidence {
    font-size: 0.85rem;
    color: #6e7681;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Scoreline badge ── */
.scoreline-wrap {
    border: 1px solid #1c2128;
    background: #0e1117;
    padding: 24px;
    margin: 16px 0 20px;
    text-align: center;
}
.scoreline-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #6e7681;
    margin-bottom: 12px;
}
.scoreline-value {
    font-size: 3.5rem;
    font-weight: 900;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 8px;
    line-height: 1;
}
.scoreline-meta {
    font-size: 0.8rem;
    color: #6e7681;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Notice / disclaimer ── */
.notice-box {
    background: #0e1117;
    border-left: 3px solid #f59e0b;
    padding: 12px 16px;
    margin-bottom: 24px;
    font-size: 12px;
    color: #6e7681;
    line-height: 1.7;
}
.notice-box b { color: #f59e0b; }
.notice-box-red {
    background: #0e1117;
    border: 1px solid #f43f5e33;
    border-left: 3px solid #f43f5e;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 12px;
    color: #6e7681;
    line-height: 1.7;
}
.notice-box-red b { color: #f43f5e; }

/* ── Sidebar info ── */
.sidebar-disclaimer {
    font-size: 10px;
    color: #6e7681;
    line-height: 1.7;
    background: #0e1117;
    border: 1px solid #f59e0b33;
    border-left: 2px solid #f59e0b;
    padding: 10px 12px;
}
.sidebar-disclaimer b { color: #f59e0b; }

</style>
""", unsafe_allow_html=True)

# ─── Team Color Map ────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Arsenal":         {"primary": "#EF0107", "text": "#ffffff"},
    "Aston Villa":     {"primary": "#95BFE5", "text": "#670E36"},
    "Bournemouth":     {"primary": "#DA291C", "text": "#ffffff"},
    "Brentford":       {"primary": "#e30613", "text": "#ffffff"},
    "Brighton":        {"primary": "#0057B8", "text": "#ffffff"},
    "Burnley":         {"primary": "#6C1D45", "text": "#ffffff"},
    "Chelsea":         {"primary": "#034694", "text": "#ffffff"},
    "Crystal Palace":  {"primary": "#1B458F", "text": "#ffffff"},
    "Everton":         {"primary": "#003399", "text": "#ffffff"},
    "Fulham":          {"primary": "#CC0000", "text": "#ffffff"},
    "Ipswich":         {"primary": "#3a64a3", "text": "#ffffff"},
    "Leeds":           {"primary": "#FFCD00", "text": "#1D428A"},
    "Leicester":       {"primary": "#003090", "text": "#ffffff"},
    "Liverpool":       {"primary": "#C8102E", "text": "#ffffff"},
    "Luton":           {"primary": "#F78F1E", "text": "#002060"},
    "Man City":        {"primary": "#6CABDD", "text": "#1c2c5b"},
    "Man United":      {"primary": "#DA291C", "text": "#ffffff"},
    "Newcastle":       {"primary": "#241F20", "text": "#ffffff"},
    "Nott'm Forest":   {"primary": "#DD0000", "text": "#ffffff"},
    "Sheffield United":{"primary": "#EE2737", "text": "#ffffff"},
    "Southampton":     {"primary": "#D71920", "text": "#ffffff"},
    "Sunderland":      {"primary": "#EB172B", "text": "#ffffff"},
    "Tottenham":       {"primary": "#132257", "text": "#ffffff"},
    "West Ham":        {"primary": "#7A263A", "text": "#1BB1E7"},
    "Wolves":          {"primary": "#FDB913", "text": "#231f20"},
}

def team_color(team: str) -> str:
    return TEAM_COLORS.get(team, {}).get("primary", "#00d4aa")

def team_text(team: str) -> str:
    return TEAM_COLORS.get(team, {}).get("text", "#080b10")

# ─── Imports ──────────────────────────────────────────────────────────────────
from data_loader import load_full_dataset, fetch_xg_data, compute_league_table
from feature_engine import get_teams, get_elo_state
from predictor import predict_match, OUTCOME_COLORS
from visualizations import (
    plot_result_distribution, plot_elo_rankings, plot_xg_trend,
    plot_score_heatmap, plot_outcome_bars, plot_poisson_distribution,
    plot_goals_per_season, plot_h2h, plot_team_radar,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 24px 0 12px;'>
        <div style='font-size:9px; font-weight:700; letter-spacing:4px; color:#00d4aa; text-transform:uppercase; margin-bottom:6px;'>EPL</div>
        <div style='font-size:1.3rem; font-weight:900; color:#e6edf3; letter-spacing:-0.5px;'>Match Predictor</div>
        <div style='font-size:10px; color:#6e7681; margin-top:4px;'>Stacked Ensemble · Calibrated Poisson</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["Home", "Match Predictor", "Season Analytics", "Head to Head", "League Table"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:10px; color:#6e7681; line-height:2;'>
    <span style='font-size:9px; font-weight:700; letter-spacing:3px; color:#e6edf3; text-transform:uppercase;'>Model</span><br>
    XGBoost + Random Forest<br>
    &rarr; Logistic Regression meta<br><br>
    <span style='font-size:9px; font-weight:700; letter-spacing:3px; color:#e6edf3; text-transform:uppercase;'>Poisson</span><br>
    Zone-scaled, leakage-free<br>
    Goals + xG blend<br><br>
    <span style='font-size:9px; font-weight:700; letter-spacing:3px; color:#e6edf3; text-transform:uppercase;'>Sources</span><br>
    football-data.co.uk<br>
    Understat (xG)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-disclaimer'>
        <b>Disclaimer</b><br>
        Academic project only. Predictions are
        <b>not</b> intended for sports betting or
        gambling. The author accepts no
        responsibility for financial losses.
    </div>
    """, unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data():
    df = load_full_dataset()
    df_xg = fetch_xg_data()
    return df, df_xg

with st.spinner("Loading EPL data..."):
    df_full, df_xg = get_data()

TEAMS = get_teams()
ELO_STATE = get_elo_state()
SEASONS = ["All Seasons"] + (
    sorted(df_full["Season"].unique().tolist()) if not df_full.empty else []
)

# ═══════════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "Home":
    st.markdown("<div class='hero-eyebrow'>Premier League Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>EPL <em>Match</em> Predictor</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-sub'>
        Stacked ensemble model (XGBoost + Random Forest &rarr; Logistic Regression),
        calibrated via leakage-free Poisson goal modelling using blended Goals + xG.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class='notice-box'>
        <b>Academic Disclaimer</b> &mdash;
        This application is a <b>student project</b> developed solely for educational
        and research purposes. All predictions are statistical estimates and are
        <b>not</b> intended to be used for sports betting, gambling, or any
        financial decision-making. The author accepts no responsibility for losses.
    </div>
    """, unsafe_allow_html=True)

    if not df_full.empty:
        total = len(df_full)
        hw = (df_full["FTR"] == "H").sum()
        dr = (df_full["FTR"] == "D").sum()
        aw = (df_full["FTR"] == "A").sum()
        avg_goals = (df_full["FTHG"].sum() + df_full["FTAG"].sum()) / total if total else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Matches", f"{total:,}")
        c2.metric("Home Wins", f"{hw:,}", f"{hw/total*100:.1f}%")
        c3.metric("Draws", f"{dr:,}", f"{dr/total*100:.1f}%")
        c4.metric("Away Wins", f"{aw:,}", f"{aw/total*100:.1f}%")
        c5.metric("Avg Goals / Match", f"{avg_goals:.2f}")

        st.markdown("---")

        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(plot_result_distribution(df_full, "All Seasons"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(plot_elo_rankings(ELO_STATE), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(plot_goals_per_season(df_full), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Could not load match data. Check your internet connection.")

    if ELO_STATE:
        st.markdown("<div class='sec-label'>Current ELO Standings</div>", unsafe_allow_html=True)
        elo_df = pd.DataFrame(
            sorted(ELO_STATE.items(), key=lambda x: -x[1]),
            columns=["Team", "ELO Rating"]
        )
        elo_df.insert(0, "Rank", range(1, len(elo_df) + 1))
        elo_df["ELO Rating"] = elo_df["ELO Rating"].round(1)
        st.dataframe(elo_df, use_container_width=True, hide_index=True, height=380)

# ═══════════════════════════════════════════════════════════════════════════════
#  MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Match Predictor":
    st.markdown("<div class='hero-eyebrow'>Premier League Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'><em>Match</em> Predictor</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='notice-box-red'>
        <b>Important Notice</b> &mdash;
        Predictions are for <b>academic and entertainment purposes only</b>.
        They are <b>not</b> a basis for sports betting or wagering of any kind.
        The author accepts <b>no liability</b> for financial losses incurred.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home_team = st.selectbox(
            "Home Team", TEAMS,
            index=TEAMS.index("Arsenal") if "Arsenal" in TEAMS else 0,
            key="home",
        )
    with col2:
        st.markdown("<div style='text-align:center; padding-top:28px; font-size:0.75rem; font-weight:700; letter-spacing:4px; color:#1c2128; font-family:JetBrains Mono,monospace;'>VS</div>", unsafe_allow_html=True)
    with col3:
        away_options = [t for t in TEAMS if t != home_team]
        away_team = st.selectbox(
            "Away Team", away_options,
            index=away_options.index("Liverpool") if "Liverpool" in away_options else 0,
            key="away",
        )

    # Versus board
    hc = team_color(home_team)
    ac = team_color(away_team)
    st.markdown(f"""
    <div class='versus-board'>
        <div class='versus-team' style='border-top: 3px solid {hc};'>
            <div class='versus-team-label'>Home</div>
            <div class='versus-team-name' style='color:{hc};'>{home_team}</div>
        </div>
        <div class='versus-divider'>VS</div>
        <div class='versus-team' style='border-top: 3px solid {ac};'>
            <div class='versus-team-label'>Away</div>
            <div class='versus-team-name' style='color:{ac};'>{away_team}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("Run Prediction")

    if predict_btn:
        with st.spinner("Running stacked model + calibrated Poisson..."):
            try:
                result    = predict_match(home_team, away_team)
                proba     = result["proba"]
                matrix    = result["matrix"]
                predicted = result["predicted_outcome"]
                stk       = result["stacked_proba"]
                hc        = team_color(home_team)
                ac        = team_color(away_team)

                # ── Result color by outcome ───────────────────────────────────
                outcome_color = hc if predicted == "Home Win" else (ac if predicted == "Away Win" else "#f59e0b")
                outcome_label = f"{home_team} Win" if predicted == "Home Win" else (f"{away_team} Win" if predicted == "Away Win" else "Draw")

                # ── SECTION 1: Meta Model result block ────────────────────────
                st.markdown("<div class='sec-label'>Stacking Meta Model &mdash; XGBoost + Random Forest &rarr; Logistic Regression</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class='result-block' style='border-left-color:{outcome_color};'>
                    <div class='result-verdict'>Predicted Outcome</div>
                    <div class='result-outcome' style='color:{outcome_color};'>{outcome_label}</div>
                    <div class='result-confidence'>{stk[predicted]*100:.1f}% confidence &nbsp;&middot;&nbsp; {home_team} vs {away_team}</div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric(f"{home_team} Win", f"{stk['Home Win']*100:.1f}%")
                m2.metric("Draw", f"{stk['Draw']*100:.1f}%")
                m3.metric(f"{away_team} Win", f"{stk['Away Win']*100:.1f}%")

                st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
                st.plotly_chart(
                    plot_outcome_bars(stk, home_team, away_team, home_color=hc, away_color=ac),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

                # ── SECTION 2: Calibrated Poisson ─────────────────────────────
                st.markdown(f"<div class='sec-label'>Calibrated Poisson &mdash; &lambda;&sub;h;={result['lambda_h']:.2f}, &lambda;&sub;a;={result['lambda_a']:.2f}</div>", unsafe_allow_html=True)

                mls = result["most_likely_score"]
                st.markdown(f"""
                <div class='scoreline-wrap'>
                    <div class='scoreline-label'>Most Likely Scoreline</div>
                    <div class='scoreline-value'>{mls[0]}&ndash;{mls[1]}</div>
                    <div class='scoreline-meta'>
                        {home_team} &nbsp;&middot;&nbsp; {away_team} &nbsp;&middot;&nbsp;
                        p = {result['matrix'][mls[0], mls[1]]*100:.1f}%
                    </div>
                    <div style='font-size:0.7rem; color:#6e7681; margin-top:10px;'>
                        The most likely scoreline may differ from the predicted outcome &mdash;
                        outcome probabilities aggregate across all matching scorelines.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                p1, p2, p3 = st.columns(3)
                p1.metric(f"{home_team} Win (Poisson)", f"{proba['Home Win']*100:.1f}%")
                p2.metric("Draw (Poisson)", f"{proba['Draw']*100:.1f}%")
                p3.metric(f"{away_team} Win (Poisson)", f"{proba['Away Win']*100:.1f}%")

                st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
                st.plotly_chart(
                    plot_score_heatmap(matrix, home_team, away_team, home_color=hc, away_color=ac),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
                st.plotly_chart(
                    plot_poisson_distribution(result["h_pmf8"], result["a_pmf8"], home_team, away_team, home_color=hc, away_color=ac),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

                # ── Model inputs ──────────────────────────────────────────────
                st.markdown("<div class='sec-label'>Model Inputs</div>", unsafe_allow_html=True)
                feat_data = {
                    "Metric": [
                        f"{home_team} Home Avg Scored (lh)",
                        f"{away_team} Away Avg Scored (la)",
                        "Meta Model: Home Win", "Meta Model: Draw", "Meta Model: Away Win",
                        "Calibrated Poisson: Home Win", "Calibrated Poisson: Draw", "Calibrated Poisson: Away Win",
                    ],
                    "Value": [
                        f"{result['lambda_h']:.3f}", f"{result['lambda_a']:.3f}",
                        f"{stk['Home Win']*100:.1f}%", f"{stk['Draw']*100:.1f}%", f"{stk['Away Win']*100:.1f}%",
                        f"{proba['Home Win']*100:.1f}%", f"{proba['Draw']*100:.1f}%", f"{proba['Away Win']*100:.1f}%",
                    ],
                }
                st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

            except Exception:
                st.error("Prediction could not be generated. Please try a different team selection or refresh the page.")

# ═══════════════════════════════════════════════════════════════════════════════
#  SEASON ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Season Analytics":
    st.markdown("<div class='hero-eyebrow'>Data &middot; Trends &middot; Form</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'><em>Season</em> Analytics</div>", unsafe_allow_html=True)

    col_s, col_t = st.columns([1, 2])
    with col_s:
        sel_season = st.selectbox("Season", SEASONS, key="season_sel")
    with col_t:
        all_teams_sel = st.multiselect(
            "Compare Teams (radar chart)",
            TEAMS, default=TEAMS[:4] if len(TEAMS) >= 4 else TEAMS,
            max_selections=4,
        )

    if df_full.empty:
        st.warning("No match data available.")
    else:
        df_s = df_full if sel_season == "All Seasons" else df_full[df_full["Season"] == sel_season]

        total_s = len(df_s)
        if total_s > 0:
            hw_s = (df_s["FTR"] == "H").sum()
            dr_s = (df_s["FTR"] == "D").sum()
            aw_s = (df_s["FTR"] == "A").sum()
            avg_g = (df_s["FTHG"].sum() + df_s["FTAG"].sum()) / total_s

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Matches", f"{total_s}")
            k2.metric("Home Win %", f"{hw_s/total_s*100:.1f}%")
            k3.metric("Draw %", f"{dr_s/total_s*100:.1f}%")
            k4.metric("Avg Goals / Match", f"{avg_g:.2f}")

        st.markdown("---")

        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(plot_result_distribution(df_s, "All Seasons"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(plot_goals_per_season(df_full), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if all_teams_sel:
            st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
            st.plotly_chart(plot_team_radar(df_s, all_teams_sel, "All Seasons"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-label'>Team xG Form Trend</div>", unsafe_allow_html=True)
        xg_team = st.selectbox("Select Team", TEAMS, key="xg_team_sel")
        if not df_xg.empty:
            st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
            st.plotly_chart(
                plot_xg_trend(df_xg, xg_team, team_color=team_color(xg_team)),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("xG data not available.")

# ═══════════════════════════════════════════════════════════════════════════════
#  HEAD TO HEAD
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Head to Head":
    st.markdown("<div class='hero-eyebrow'>Historical Record</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Head <em>to</em> Head</div>", unsafe_allow_html=True)

    hc1, hc2 = st.columns(2)
    with hc1:
        h2h_team1 = st.selectbox("Team 1", TEAMS, index=0, key="h2h_t1")
    with hc2:
        h2h_opts = [t for t in TEAMS if t != h2h_team1]
        h2h_team2 = st.selectbox("Team 2", h2h_opts, index=min(1, len(h2h_opts)-1), key="h2h_t2")

    if df_full.empty:
        st.warning("No match data available.")
    else:
        mask = (
            ((df_full["HomeTeam"] == h2h_team1) & (df_full["AwayTeam"] == h2h_team2)) |
            ((df_full["HomeTeam"] == h2h_team2) & (df_full["AwayTeam"] == h2h_team1))
        )
        h2h_df = df_full[mask].copy()
        n_matches = len(h2h_df)

        st.markdown(
            f"<div style='font-size:12px; color:#6e7681; margin-bottom:20px; font-family:JetBrains Mono,monospace;'>{n_matches} matches &middot; 2022/23 &ndash; 2025/26</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(
            plot_h2h(df_full, h2h_team1, h2h_team2,
                     color1=team_color(h2h_team1), color2=team_color(h2h_team2)),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if not h2h_df.empty:
            st.markdown("<div class='sec-label'>Match History</div>", unsafe_allow_html=True)
            display_cols = ["Date", "Season", "HomeTeam", "FTHG", "FTAG", "AwayTeam", "FTR"]
            display_cols = [c for c in display_cols if c in h2h_df.columns]
            h2h_display = h2h_df[display_cols].sort_values("Date", ascending=False).copy()
            h2h_display["Date"] = h2h_display["Date"].dt.strftime("%Y-%m-%d")
            h2h_display["FTR"] = h2h_display["FTR"].map({"H": "Home Win", "D": "Draw", "A": "Away Win"})
            h2h_display.columns = ["Date", "Season", "Home Team", "HG", "AG", "Away Team", "Result"][:len(display_cols)]
            st.dataframe(h2h_display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LEAGUE TABLE
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "League Table":
    st.markdown("<div class='hero-eyebrow'>Standing &middot; Points</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'><em>League</em> Table</div>", unsafe_allow_html=True)

    lt_season = st.selectbox(
        "Season",
        [s for s in SEASONS if s != "All Seasons"] + ["All Seasons"],
        key="lt_season",
    )

    if df_full.empty:
        st.warning("No match data available.")
    else:
        table = compute_league_table(df_full, lt_season)

        if not table.empty:
            def style_table(row):
                styles = [""] * len(row)
                pos = row["#"]
                if pos <= 4:
                    styles = [
                        "background-color: #00d4aa18; color: #00d4aa" if i == 0 else ""
                        for i in range(len(row))
                    ]
                elif pos <= 6:
                    styles = [
                        "background-color: #3b82f618;" if i == 0 else ""
                        for i in range(len(row))
                    ]
                elif pos >= len(table) - 2:
                    styles = [
                        "background-color: #f43f5e18; color: #f43f5e" if i == 0 else ""
                        for i in range(len(row))
                    ]
                return styles

            st.dataframe(
                table.style.apply(style_table, axis=1),
                use_container_width=True,
                hide_index=True,
                height=720,
            )
            st.markdown("""
            <div style='font-size:11px; color:#6e7681; margin-top:10px; font-family:JetBrains Mono,monospace; letter-spacing:1px;'>
                <span style='color:#00d4aa;'>&#9632;</span> Champions League &nbsp;
                <span style='color:#3b82f6;'>&#9632;</span> Europa League &nbsp;
                <span style='color:#f43f5e;'>&#9632;</span> Relegation Zone
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No data for selected season.")
