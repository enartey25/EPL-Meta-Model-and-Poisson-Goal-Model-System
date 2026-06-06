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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p { color: #8b949e; font-size: 13px; }

[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 15px;
    padding: 12px 28px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}

.stDataFrame { border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }
.stAlert { border-radius: 10px; }
hr { border-color: #30363d; }

.pred-card {
    background: linear-gradient(135deg, #161b22, #0d1117);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 12px 0 24px;
    text-align: center;
}
.outcome-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 20px;
    letter-spacing: 0.5px;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 8px;
}
.hero-subtitle { color: #8b949e; font-size: 1rem; margin-bottom: 28px; }
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 32px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #30363d;
}
.chart-wrap {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 8px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

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
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:1.15rem; font-weight:700; color:#e6edf3; letter-spacing:1px;'>EPL PREDICTOR</div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:4px;'>Stacked Ensemble + Calibrated Poisson</div>
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
    <div style='font-size:11px; color:#8b949e; line-height:1.8;'>
    <b style='color:#00d4aa;'>Model Stack</b><br>
    XGBoost + Random Forest<br>
    &rarr; Logistic Regression meta<br><br>
    <b style='color:#00d4aa;'>Poisson Calibration</b><br>
    Zone-scaled by stacking probs<br>
    Leakage-free (Goals + xG blend)<br><br>
    <b style='color:#00d4aa;'>Data Sources</b><br>
    football-data.co.uk<br>
    Understat (xG)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:10px; color:#8b949e; line-height:1.6;
                background:#161b22; border:1px solid #f59e0b44;
                border-radius:8px; padding:10px 12px;'>
        <b style='color:#f59e0b;'>Disclaimer</b><br>
        This is an academic project built for
        educational and research purposes only.
        Predictions are <b>not</b> intended for
        sports betting or any form of gambling.
        The author accepts no responsibility for
        financial losses arising from the use of
        this tool.
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
    st.markdown("<div class='hero-title'>EPL Match Predictor</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-subtitle'>
        Stacked ensemble model (XGBoost + Random Forest &rarr; Logistic Regression),
        calibrated via leakage-free Poisson goal modelling using blended Goals + xG.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#161b22; border-left:3px solid #f59e0b;
                border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:24px;
                font-size:12px; color:#8b949e; line-height:1.7;'>
        <b style='color:#f59e0b;'>Academic Disclaimer</b> &mdash;
        This application is a <b>student project</b> developed solely for educational
        and research purposes. All predictions are statistical estimates and are
        <b>not</b> intended to be used for sports betting, gambling, or any
        financial decision-making. The author will not be held responsible for
        any financial losses or damages resulting from reliance on this tool.
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
        st.markdown("<div class='section-header'>Current ELO Standings</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='hero-title'>Match Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Select two teams to generate a calibrated match prediction.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#161b22; border:1px solid #f43f5e44;
                border-radius:8px; padding:12px 16px; margin-bottom:20px;
                font-size:12px; color:#8b949e; line-height:1.7;'>
        <b style='color:#f43f5e;'>Important Notice</b> &mdash;
        Predictions generated by this tool are for <b>academic and entertainment
        purposes only</b>. They are <b>not</b> a basis for sports betting or any
        form of wagering. No financial advice is given or implied. The author
        accepts <b>no liability</b> for any financial losses incurred.
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
        st.markdown("<div style='text-align:center; padding-top:28px; font-size:1.4rem; color:#30363d; font-weight:700;'>vs</div>", unsafe_allow_html=True)
    with col3:
        away_options = [t for t in TEAMS if t != home_team]
        away_team = st.selectbox(
            "Away Team", away_options,
            index=away_options.index("Liverpool") if "Liverpool" in away_options else 0,
            key="away",
        )

    st.markdown("")
    predict_btn = st.button("Generate Prediction")

    if predict_btn:
        with st.spinner("Running stacked model + calibrated Poisson..."):
            try:
                result = predict_match(home_team, away_team)
                proba     = result["proba"]
                matrix    = result["matrix"]
                predicted = result["predicted_outcome"]
                confidence = result["confidence"]
                stk       = result["stacked_proba"]

                badge_colors = {"Home Win": "#00d4aa", "Draw": "#f59e0b", "Away Win": "#f43f5e"}
                badge_col = badge_colors.get(predicted, "#7c3aed")

                # ── SECTION 1: Stacking Meta Model ───────────────────────────
                st.markdown("""
                <div style='margin: 8px 0 4px;'>
                    <div style='font-size:0.7rem; color:#00d4aa; letter-spacing:3px; font-weight:600; text-transform:uppercase;'>
                        Stacking Meta Model
                    </div>
                    <div style='font-size:1rem; color:#8b949e; margin-top:2px;'>
                        XGBoost + Random Forest &rarr; Logistic Regression &mdash; outcome probabilities
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='pred-card' style='border-color:{badge_col}55;'>
                    <div style='font-size:0.75rem; color:#8b949e; margin-bottom:10px; letter-spacing:2px;'>META MODEL PREDICTION</div>
                    <div class='outcome-badge' style='background:{badge_col}22; color:{badge_col}; border:1px solid {badge_col}66;'>
                        {predicted.upper()} &nbsp;&middot;&nbsp; {stk[predicted]*100:.1f}% confidence
                    </div>
                    <div style='margin-top:18px; font-size:1.1rem; color:#e6edf3; font-weight:600;'>
                        {home_team} &nbsp;<span style='color:#30363d;'>vs</span>&nbsp; {away_team}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric(f"{home_team} Win", f"{stk['Home Win']*100:.1f}%")
                m2.metric("Draw", f"{stk['Draw']*100:.1f}%")
                m3.metric(f"{away_team} Win", f"{stk['Away Win']*100:.1f}%")

                st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
                st.plotly_chart(plot_outcome_bars(stk, home_team, away_team), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

                # ── SECTION 2: Calibrated Poisson Distribution ────────────────
                st.markdown(f"""
                <div style='margin: 8px 0 4px;'>
                    <div style='font-size:0.7rem; color:#7c3aed; letter-spacing:3px; font-weight:600; text-transform:uppercase;'>
                        Calibrated Poisson Distribution
                    </div>
                    <div style='font-size:1rem; color:#8b949e; margin-top:2px;'>
                        Raw Poisson matrix (&#955;&#8320;={result["lambda_h"]:.2f}, &#955;&#8321;={result["lambda_a"]:.2f})
                        zone-scaled by meta model probs, then normalised &mdash; scoreline breakdown
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Most likely score badge
                mls = result["most_likely_score"]
                st.markdown(f"""
                <div style='background:#161b22; border:1px solid #7c3aed44; border-radius:12px;
                            padding:16px 24px; margin:12px 0 20px; text-align:center;'>
                    <div style='font-size:0.75rem; color:#8b949e; letter-spacing:2px; margin-bottom:6px;'>MOST LIKELY SCORELINE (POISSON)</div>
                    <div style='font-size:2rem; font-weight:800; color:#e6edf3; letter-spacing:4px;'>
                        {mls[0]} &ndash; {mls[1]}
                    </div>
                    <div style='font-size:0.8rem; color:#8b949e; margin-top:4px;'>
                        {home_team} &nbsp;&middot;&nbsp; {away_team} &nbsp;&middot;&nbsp;
                        Probability: <b style='color:#7c3aed;'>{result["matrix"][mls[0], mls[1]]*100:.1f}%</b>
                    </div>
                    <div style='font-size:0.75rem; color:#8b949e; margin-top:8px; font-style:italic;'>
                        Note: the most likely single scoreline can differ from the meta model's predicted outcome
                        because outcome probabilities sum across all matching scorelines.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                p1, p2, p3 = st.columns(3)
                p1.metric(f"{home_team} Win (Poisson)", f"{proba['Home Win']*100:.1f}%")
                p2.metric("Draw (Poisson)", f"{proba['Draw']*100:.1f}%")
                p3.metric(f"{away_team} Win (Poisson)", f"{proba['Away Win']*100:.1f}%")

                # Calibrated heatmap
                st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
                st.plotly_chart(plot_score_heatmap(matrix, home_team, away_team), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Poisson goal distribution
                st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
                st.plotly_chart(
                    plot_poisson_distribution(result["h_pmf8"], result["a_pmf8"], home_team, away_team),
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

                # ── Model inputs table ────────────────────────────────────────
                st.markdown("<div class='section-header'>Model Inputs</div>", unsafe_allow_html=True)
                feat_data = {
                    "Metric": [
                        f"{home_team} Home Avg Scored (lambda h)",
                        f"{away_team} Away Avg Scored (lambda a)",
                        "Meta Model: Home Win",
                        "Meta Model: Draw",
                        "Meta Model: Away Win",
                        "Calibrated Poisson: Home Win",
                        "Calibrated Poisson: Draw",
                        "Calibrated Poisson: Away Win",
                    ],
                    "Value": [
                        f"{result['lambda_h']:.3f}",
                        f"{result['lambda_a']:.3f}",
                        f"{stk['Home Win']*100:.1f}%",
                        f"{stk['Draw']*100:.1f}%",
                        f"{stk['Away Win']*100:.1f}%",
                        f"{proba['Home Win']*100:.1f}%",
                        f"{proba['Draw']*100:.1f}%",
                        f"{proba['Away Win']*100:.1f}%",
                    ],
                }
                st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

            except Exception:
                st.error("Prediction could not be generated. Please try a different team selection or refresh the page.")

# ═══════════════════════════════════════════════════════════════════════════════
#  SEASON ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Season Analytics":
    st.markdown("<div class='hero-title'>Season Analytics</div>", unsafe_allow_html=True)

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

        st.markdown("<div class='section-header'>Team xG Form Trend</div>", unsafe_allow_html=True)
        xg_team = st.selectbox("Select Team", TEAMS, key="xg_team_sel")
        if not df_xg.empty:
            st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
            st.plotly_chart(plot_xg_trend(df_xg, xg_team), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("xG data not available.")

# ═══════════════════════════════════════════════════════════════════════════════
#  HEAD TO HEAD
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "Head to Head":
    st.markdown("<div class='hero-title'>Head to Head</div>", unsafe_allow_html=True)

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
            f"<div style='color:#8b949e; margin-bottom:16px;'>{n_matches} matches found (2022/23 – 2025/26)</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(plot_h2h(df_full, h2h_team1, h2h_team2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if not h2h_df.empty:
            st.markdown("<div class='section-header'>Match History</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='hero-title'>League Table</div>", unsafe_allow_html=True)

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
                        "background-color: #7c3aed18;" if i == 0 else ""
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
            <div style='font-size:12px; color:#8b949e; margin-top:8px;'>
                <span style='color:#00d4aa;'>&#9632;</span> Champions League &nbsp;
                <span style='color:#7c3aed;'>&#9632;</span> Europa League &nbsp;
                <span style='color:#f43f5e;'>&#9632;</span> Relegation Zone
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No data for selected season.")
