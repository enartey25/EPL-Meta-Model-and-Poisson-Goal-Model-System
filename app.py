"""EPL Forecast: Streamlit app over the weekly pipeline's model and data files."""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EPL Forecast", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

import charts  # noqa: E402
import ui  # noqa: E402
from data_loader import (compute_league_table, load_forecast_log, load_full_dataset, load_metrics,  # noqa: E402
                         load_track_record, load_upcoming)
from predictor import is_cold_start, predict_match, teams  # noqa: E402

UK = "Europe/London"
ui.inject_css()

matches = load_full_dataset()
metrics = load_metrics()
record = load_track_record()
upcoming = load_upcoming()
CURRENT = metrics.get("current_season", matches["Season"].max() if not matches.empty else "")


def kickoff_label(ts) -> str:
    t = pd.Timestamp(ts).tz_convert(UK)
    return t.strftime("%a %d %b · %H:%M")


def fmt_ll(x) -> str:
    return f"{x:.3f}"


def current_teams() -> list[str]:
    cur = matches[matches["Season"] == CURRENT]
    names = set(cur["HomeTeam"]) | set(cur["AwayTeam"])
    if not upcoming.empty:
        names |= set(upcoming["HomeTeam"]) | set(upcoming["AwayTeam"])
    return sorted(names) or teams()


def settled_with_logs(season: str) -> pd.DataFrame:
    """Walk-forward record for a season, with forecasts logged before kickoff
    taking precedence where the pipeline issued one."""
    rec = record[record["season"] == season].copy()
    rec["logged"] = False
    log = load_forecast_log()
    if not rec.empty and not log.empty and "fotmob_id" in rec.columns:
        log = log.drop_duplicates("fotmob_id", keep="first").set_index("fotmob_id")
        hit = rec["fotmob_id"].isin(log.index)
        for col in ("P_Draw", "P_Away", "P_Home"):
            rec.loc[hit, col] = rec.loc[hit, "fotmob_id"].map(log[col]).values
        rec.loc[hit, "logged"] = True
    return rec


def outcome_letter(ftr: int) -> str:
    return {2: "H", 1: "A", 0: "D"}[int(ftr)]


# ======================================================================= pages
def fixtures_page():
    if upcoming.empty:
        st.markdown('<h1 class="pg">No fixtures scheduled</h1><p class="pgsub">The next gameweek\'s coupons are issued '
                    'when the pipeline next runs after fixtures are published.</p>', unsafe_allow_html=True)
    else:
        rnd = upcoming["round"].iloc[0]
        first, last = upcoming["kickoff_utc"].min().tz_convert(UK), upcoming["kickoff_utc"].max().tz_convert(UK)
        span = first.strftime("%d %b") if first.date() == last.date() else f"{first.strftime('%d')}–{last.strftime('%d %b %Y')}"
        issued = pd.Timestamp(metrics.get("generated_at", pd.Timestamp.now(tz="UTC"))).tz_convert(UK).strftime("%d %b %Y")
        st.markdown(f'<h1 class="pg">Gameweek {int(rnd)}</h1><p class="pgsub">{span} · issued {issued} · '
                    f'home / draw / away probabilities in %, and the single most likely score</p>', unsafe_allow_html=True)

    left, right = st.columns([3.3, 1], gap="large")
    with left:
        if not upcoming.empty:
            cps = "".join(ui.coupon(r.HomeTeam, r.AwayTeam, r.P_Home, r.P_Draw, r.P_Away,
                                    kickoff_label(r.kickoff_utc), f"No. {int(r.fotmob_id)}",
                                    "Likely", r.Most_Likely_Score, f"{r.P_Most_Likely_Score * 100:.0f}%")
                          for r in upcoming.itertuples())
            st.markdown(f'<div class="wallet">{cps}</div>', unsafe_allow_html=True)
    with right:
        tr = metrics.get("track_record", {}).get(CURRENT)
        if tr:
            st.markdown(ui.ledger([
                ("Model log loss", fmt_ll(tr["model"]["log_loss"]), f"{CURRENT}, {tr['model']['n']} matches", True),
                ("Baseline", fmt_ll(tr["null_baseline"]["log_loss"]), "historical result rates", False),
                ("Favourite won", f"{tr['favourite_hit_rate'] * 100:.0f}%", "of settled coupons", False),
            ]), unsafe_allow_html=True)
            st.markdown('<p class="note">Lower log loss is better. Coupons are scored walk-forward: models frozen '
                        'at the start of the season, features updated before each match.</p>', unsafe_allow_html=True)

    rec = settled_with_logs(CURRENT)
    if not rec.empty:
        last_round = rec["round"].max()
        last = rec[rec["round"] == last_round].sort_values("Date")
        hits = int((last[["P_Draw", "P_Away", "P_Home"]].to_numpy().argmax(1) == last["FTR"].to_numpy()).sum())
        st.markdown(f'<h2 class="sec">Gameweek {int(last_round)}, settled: {hits} of {len(last)} favourites won</h2>',
                    unsafe_allow_html=True)
        cps = []
        for r in last.itertuples():
            fav = int(np.argmax([r.P_Draw, r.P_Away, r.P_Home]))
            cps.append(ui.coupon(r.HomeTeam, r.AwayTeam, r.P_Home, r.P_Draw, r.P_Away,
                                 pd.Timestamp(r.Date).strftime("%a %d %b"), "Logged pre-match" if r.logged else "Walk-forward",
                                 "Final", f"{int(r.FTHG)}–{int(r.FTAG)}", actual=outcome_letter(r.FTR),
                                 stamp="hit" if fav == int(r.FTR) else "miss"))
        st.markdown(f'<div class="wallet">{"".join(cps)}</div>', unsafe_allow_html=True)
    ui.footer(metrics)


def predictor_page():
    st.markdown('<h1 class="pg">Match predictor</h1><p class="pgsub">Issue a forecast for any pairing, with the '
                'model\'s current ratings and form as of the latest results.</p>', unsafe_allow_html=True)
    opts = current_teams()
    others = [t for t in teams() if t not in opts]
    options = opts + others
    d_home, d_away = (upcoming["HomeTeam"].iloc[0], upcoming["AwayTeam"].iloc[0]) if not upcoming.empty else (options[0], options[1])
    with st.form("issue"):
        c1, c2, c3 = st.columns([2, 2, 1.2], vertical_alignment="bottom")
        home = c1.selectbox("Home club", options, index=options.index(d_home) if d_home in options else 0)
        away = c2.selectbox("Away club", options, index=options.index(d_away) if d_away in options else 1)
        go = c3.form_submit_button("Issue forecast", use_container_width=True)
    if home == away:
        st.error("Choose two different clubs: a club can't play itself.")
        return
    if not go and "issued" not in st.session_state:
        st.markdown('<p class="prose">Pick a home and an away club, then issue the forecast.</p>', unsafe_allow_html=True)
        ui.footer(metrics)
        return
    st.session_state["issued"] = True
    try:
        res = predict_match(home, away)
    except Exception as e:  # unknown club or broken assets
        st.error(f"The forecast couldn't be issued: {e}")
        return
    p = res["proba"]
    fav = max(p, key=p.get)
    fields = "".join(f'<div class="fd{" fav" if k == fav else ""}">{lbl}<b>{p[k] * 100:.0f}</b></div>'
                     for k, lbl in (("Home Win", "Home win"), ("Draw", "Draw"), ("Away Win", "Away win")))
    hs, as_ = res["most_likely_score"]
    mk = res["markets"]
    carbon = "".join(f'<div class="fd">{ui.esc(k)}<b>{v * 100:.0f}%</b></div>' for k, v in mk.items())
    carbon += (f'<div class="fd">Expected goals<b>{res["lambda_h"]:.2f}–{res["lambda_a"]:.2f}</b></div>')
    issued = pd.Timestamp.now(tz=UK).strftime("%d %b %Y · %H:%M")
    st.markdown(
        f'<div class="issue" role="group" aria-label="Forecast {ui.esc(home)} v {ui.esc(away)}">'
        f'<div class="is-hd"><span>Forecast · {CURRENT}</span><span>Issued {issued}</span></div>'
        f'<div class="route"><div class="c">{ui.code(res["home_team"])}<small>{ui.esc(res["home_team"])} · home</small></div>'
        f'<div class="vs">v</div><div class="c r">{ui.code(res["away_team"])}<small>{ui.esc(res["away_team"])} · away</small></div></div>'
        f'<div class="is-fields">{fields}<div class="fd">Likely score<b>{hs}–{as_}</b></div></div>'
        f'<div class="carbon">{carbon}</div></div>', unsafe_allow_html=True)
    cold = [t for t in (home, away) if is_cold_start(t)]
    if cold:
        st.markdown(f'<p class="note"><b>Prior in use:</b> {ui.esc(" and ".join(cold))} has no top-flight matches in the '
                    f'model\'s history yet, so it starts from 1420 Elo and league-average xG.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        st.markdown('<h2 class="sec">Score grid (%)</h2>', unsafe_allow_html=True)
        st.plotly_chart(charts.score_grid(res["matrix"], res["home_team"], res["away_team"]),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        f = res["features"]
        st.markdown('<h2 class="sec">Inputs</h2>', unsafe_allow_html=True)
        rows = [("Pre-match Elo", f"{f['home_elo']:.0f}", f"{f['away_elo']:.0f}"),
                ("xG for, last 5", f"{f['home_xg_for']:.2f}", f"{f['away_xg_for']:.2f}"),
                ("xG against, last 5", f"{f['home_xg_against']:.2f}", f"{f['away_xg_against']:.2f}"),
                ("Poisson rate (raw)", f"{res['raw_lambda_h']:.2f}", f"{res['raw_lambda_a']:.2f}")]
        body = "".join(f'<tr><td class="l team">{a}</td><td>{b}</td><td>{c}</td></tr>' for a, b, c in rows)
        st.markdown(f'<div class="tw"><table class="rt"><tr><th class="l"></th><th>{ui.esc(res["home_team"])}</th>'
                    f'<th>{ui.esc(res["away_team"])}</th></tr>{body}</table></div>'
                    '<p class="prose" style="margin-top:14px">The win/draw/loss figures come from a logistic-regression '
                    'stack of a Random Forest (Elo, xG form and volatility) and an opponent-adjusted Poisson goal model. '
                    'The score grid uses Poisson goal rates solved to match those probabilities.</p>',
                    unsafe_allow_html=True)
    ui.footer(metrics)


def record_page():
    st.markdown('<h1 class="pg">Track record</h1><p class="pgsub">Every match scored against a naive baseline '
                '(historical result rates), misses included. Lower log loss and Brier score are better.</p>',
                unsafe_allow_html=True)
    seasons = sorted(record["season"].unique(), reverse=True) if not record.empty else []
    if not seasons:
        st.info("No settled forecasts yet.")
        return
    tabs = st.tabs(seasons)
    for tab, season in zip(tabs, seasons):
        with tab:
            rec = settled_with_logs(season).sort_values("Date")
            tr = metrics.get("track_record", {}).get(season, {})
            if tr:
                m, b = tr["model"], tr["null_baseline"]
                st.markdown(ui.ledger([
                    ("Log loss", fmt_ll(m["log_loss"]), f"baseline {fmt_ll(b['log_loss'])}", m["log_loss"] < b["log_loss"]),
                    ("Brier score", fmt_ll(m["brier"]), f"baseline {fmt_ll(b['brier'])}", m["brier"] < b["brier"]),
                    ("Favourite won", f"{tr['favourite_hit_rate'] * 100:.0f}%", "share of matches", False),
                    ("Matches", f"{m['n']}", f"{int(rec['logged'].sum())} logged pre-match", False),
                ]), unsafe_allow_html=True)
            st.markdown(f'<p class="note">{season}: <b>walk-forward</b> forecasts. Models were trained only on '
                        f'seasons before {season}; ratings, form and goal rates were updated before each match, '
                        f'so no forecast saw its own result. Forecasts the weekly pipeline issued before kickoff '
                        f'are marked <b>logged pre-match</b>.</p>', unsafe_allow_html=True)
            st.plotly_chart(charts.cumulative_edge(rec), use_container_width=True, config={"displayModeBar": False})
            c1, c2 = st.columns(2, gap="large")
            c1.plotly_chart(charts.round_losses(rec), use_container_width=True, config={"displayModeBar": False})
            c2.plotly_chart(charts.calibration(rec), use_container_width=True, config={"displayModeBar": False})

            st.markdown('<h2 class="sec">Every settled forecast</h2>', unsafe_allow_html=True)
            show = rec.sort_values("Date", ascending=False)
            rows = []
            for r in show.itertuples():
                pr = [r.P_Draw, r.P_Away, r.P_Home]
                fav = int(np.argmax(pr))
                ok = fav == int(r.FTR)
                mark = '<span class="zone ucl"></span>Hit' if ok else '<span class="zone rel"></span><s>Miss</s>'
                rows.append(f'<tr><td class="l">{pd.Timestamp(r.Date).strftime("%d %b")}</td><td>{int(r.round) if pd.notna(r.round) else ""}</td>'
                            f'<td class="l team">{ui.esc(r.HomeTeam)} {int(r.FTHG)}–{int(r.FTAG)} {ui.esc(r.AwayTeam)}</td>'
                            f'<td>{r.P_Home * 100:.0f}</td><td>{r.P_Draw * 100:.0f}</td><td>{r.P_Away * 100:.0f}</td>'
                            f'<td>{pr[int(r.FTR)] * 100:.0f}</td><td class="l">{mark}</td></tr>')
            st.markdown('<div class="tw" style="max-height:520px;overflow-y:auto"><table class="rt"><tr><th class="l">Date</th>'
                        '<th>GW</th><th class="l">Result</th><th>Home %</th><th>Draw %</th><th>Away %</th>'
                        '<th>% on result</th><th class="l">Favourite</th></tr>' + "".join(rows) + '</table></div>',
                        unsafe_allow_html=True)

    h = metrics.get("holdout")
    if h:
        st.markdown(f'<h2 class="sec">Holdout test: {h["test_season"]}</h2>', unsafe_allow_html=True)
        names = [("Naive baseline", "null_baseline"), ("Random Forest alone", "random_forest"),
                 ("Poisson model alone", "poisson"), ("Stack (deployed design)", "stack")]
        body = "".join(f'<tr><td class="l team">{n}</td><td>{h[k]["log_loss"]:.4f}</td><td>{h[k]["brier"]:.4f}</td></tr>'
                       for n, k in names)
        st.markdown(f'<div class="tw"><table class="rt"><tr><th class="l">Model</th><th>Log loss</th><th>Brier</th></tr>{body}'
                    f'</table></div><p class="prose" style="margin-top:14px">Development protocol: base models trained on '
                    f'{", ".join(h["train_seasons"])}, meta-learner fitted on {h["validation_season"]}, every modelling '
                    f'choice made on {h["validation_season"]}, and {h["test_season"]} ({h["stack"]["n"]} matches) scored '
                    f'once. The stack beats both of its inputs.</p>', unsafe_allow_html=True)
    ui.footer(metrics)


def analytics_page():
    st.markdown('<h1 class="pg">Analytics</h1><p class="pgsub">League trends, ratings and form from every match since '
                '2022/23.</p>', unsafe_allow_html=True)
    if matches.empty:
        st.info("Match data is not available yet.")
        return
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(charts.result_split(matches), use_container_width=True, config={"displayModeBar": False})
    c2.plotly_chart(charts.goals_by_season(matches), use_container_width=True, config={"displayModeBar": False})

    cur = current_teams()
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.plotly_chart(charts.elo_ladder(metrics.get("elo", {}), cur), use_container_width=True, config={"displayModeBar": False})
    with c2:
        team = st.selectbox("Club form", cur, index=cur.index("Arsenal") if "Arsenal" in cur else 0)
        fig = charts.xg_form(matches, team)
        if fig is None:
            st.info(f"{team} has no matches in the data yet.")
        else:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<h2 class="sec">Head to head</h2>', unsafe_allow_html=True)
    allt = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    a1, a2 = st.columns(2)
    t1 = a1.selectbox("Club", allt, index=allt.index("Arsenal") if "Arsenal" in allt else 0, key="h1")
    rest = [t for t in allt if t != t1]
    t2 = a2.selectbox("Opponent", rest, index=rest.index("Chelsea") if "Chelsea" in rest else 0, key="h2")
    hh = matches[((matches.HomeTeam == t1) & (matches.AwayTeam == t2)) | ((matches.HomeTeam == t2) & (matches.AwayTeam == t1))]
    if hh.empty:
        st.markdown(f'<p class="prose">{ui.esc(t1)} and {ui.esc(t2)} have not met in the Premier League since 2022/23.</p>',
                    unsafe_allow_html=True)
    else:
        w1 = int(((hh.HomeTeam == t1) & (hh.FTR == "H")).sum() + ((hh.AwayTeam == t1) & (hh.FTR == "A")).sum())
        dr = int((hh.FTR == "D").sum())
        st.markdown(ui.ledger([(f"{t1} wins", str(w1), "", False), ("Draws", str(dr), "", False),
                               (f"{t2} wins", str(len(hh) - w1 - dr), "", False)]), unsafe_allow_html=True)
        body = "".join(f'<tr><td class="l">{r.Date.strftime("%d %b %Y")}</td><td class="l team">{ui.esc(r.HomeTeam)} '
                       f'{int(r.FTHG)}–{int(r.FTAG)} {ui.esc(r.AwayTeam)}</td><td>{r.home_xG:.2f}–{r.away_xG:.2f}</td></tr>'
                       for r in hh.sort_values("Date", ascending=False).itertuples())
        st.markdown(f'<div class="tw" style="margin-top:14px"><table class="rt"><tr><th class="l">Date</th>'
                    f'<th class="l">Result</th><th>xG</th></tr>{body}</table></div>', unsafe_allow_html=True)
    ui.footer(metrics)


def table_page():
    seasons = sorted(matches["Season"].unique(), reverse=True) if not matches.empty else []
    st.markdown('<h1 class="pg">League table</h1><p class="pgsub">Computed from results, with expected goals for and '
                'against.</p>', unsafe_allow_html=True)
    if not seasons:
        st.info("Match data is not available yet.")
        return
    season = st.selectbox("Season", seasons, index=0)
    t = compute_league_table(matches, season)
    n = len(t)
    rows = []
    for r in t.itertuples():
        z = '<span class="zone ucl" title="Champions League places"></span>' if r._1 <= 4 else (
            '<span class="zone rel" title="Relegation places"></span>' if r._1 > n - 3 else '<span class="zone"></span>')
        rows.append(f'<tr><td class="l">{z}{r._1}</td><td class="l team">{ui.esc(r.Team)}</td><td>{r.P}</td><td>{r.W}</td>'
                    f'<td>{r.D}</td><td>{r.L}</td><td>{r.GF}</td><td>{r.GA}</td><td>{r.GD:+d}</td>'
                    f'<td>{r.xGF:.1f}</td><td>{r.xGA:.1f}</td><td><b>{r.Pts}</b></td></tr>')
    st.markdown('<div class="tw"><table class="rt"><tr><th class="l">#</th><th class="l">Club</th><th>P</th><th>W</th>'
                '<th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>xGF</th><th>xGA</th><th>Pts</th></tr>'
                + "".join(rows) + '</table></div>'
                '<p class="note"><span class="zone ucl"></span>Champions League places &nbsp; '
                '<span class="zone rel"></span>Relegation places</p>', unsafe_allow_html=True)
    ui.footer(metrics)


# ======================================================================= shell
PAGES = [st.Page(fixtures_page, title="Fixtures", url_path="fixtures", default=True),
         st.Page(predictor_page, title="Predictor", url_path="predictor"),
         st.Page(record_page, title="Track record", url_path="record"),
         st.Page(analytics_page, title="Analytics", url_path="analytics"),
         st.Page(table_page, title="Table", url_path="table")]
nav = st.navigation(PAGES, position="hidden")
ui.top_bar()
for col, page in zip(st.columns(len(PAGES)), PAGES):
    col.page_link(page, label=page.title)
nav.run()
