"""Ticket Wallet visual system: theme CSS, club codes and coupon components.

Carrier navy ground, coupon-stock cards, jet-red heads, carbon-purple numerals.
Rules are one device pixel; no rounded corners. Hit and miss are carried by
line form (solid stamp vs dashed, struck stamp) as well as colour.
"""
import html

import streamlit as st

NAVY, NAVY_DEEP, NAVY_RAISED = "#0D1B3D", "#08122B", "#14244A"
STOCK, STOCK_WHITE, INK = "#F2F0EA", "#FFFFFF", "#0D1B3D"
RED, RED_HEAD = "#D7262D", "#C41E26"
CARBON, CARBON_COPY, CARBON_ON_NAVY = "#4B2E83", "#DCD3F0", "#C3B4EE"
MUTED_ON_NAVY = "#B9BDC9"
RULE_ON_NAVY = "rgba(242,240,234,0.22)"

CLUB_CODES = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Burnley": "BUR", "Chelsea": "CHE", "Coventry": "COV",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL", "Hull": "HUL",
    "Ipswich": "IPS", "Leeds": "LEE", "Leicester": "LEI", "Liverpool": "LIV", "Luton": "LUT",
    "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW", "Nott'm Forest": "NFO",
    "Sheffield United": "SHU", "Southampton": "SOU", "Sunderland": "SUN", "Tottenham": "TOT",
    "West Ham": "WHU", "Wolves": "WOL", "Middlesbrough": "MID", "Norwich": "NOR",
    "Watford": "WAT", "West Brom": "WBA", "Stoke": "STK", "Swansea": "SWA",
}


def code(team: str) -> str:
    return CLUB_CODES.get(team, team[:3].upper())


def esc(s) -> str:
    return html.escape(str(s))


def pct(p: float) -> str:
    return f"{p * 100:.0f}"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --navy:{NAVY}; --navy-deep:{NAVY_DEEP}; --navy-raised:{NAVY_RAISED};
  --stock:{STOCK}; --ink:{INK}; --red:{RED}; --red-head:{RED_HEAD};
  --carbon:{CARBON}; --carbon-copy:{CARBON_COPY}; --carbon-navy:{CARBON_ON_NAVY};
  --muted:{MUTED_ON_NAVY}; --rule:{RULE_ON_NAVY};
  --display:'Barlow Condensed', 'Arial Narrow', sans-serif;
  --body:'Barlow', system-ui, sans-serif;
  --num:'IBM Plex Mono', ui-monospace, monospace;
}}
html, body, .stApp, [class*="st-"] {{ font-family: var(--body); }}
.stApp {{ background: var(--navy); color: var(--stock); }}
header[data-testid="stHeader"] {{ display: none; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding: 0 40px 64px; max-width: 1440px; }}
::selection {{ background: var(--red); color: var(--stock); }}
* {{ scrollbar-color: rgba(242,240,234,.35) var(--navy-deep); }}
a {{ color: var(--stock); text-underline-offset: 3px; }}
:focus-visible {{ outline: 2px solid var(--stock) !important; outline-offset: 2px; }}
* {{ border-radius: 0 !important; }}
p, li {{ max-width: 72ch; line-height: 1.6; }}

/* ---------- top bar ---------- */
.bar {{ display:flex; align-items:center; gap:28px; height:60px; margin:0 -40px 0; padding:0 40px;
        background:var(--navy-deep); }}
.wordmark {{ font-family:var(--display); font-weight:800; font-size:24px; letter-spacing:.1em;
             text-transform:uppercase; color:var(--stock); white-space:nowrap; }}
.wordmark b {{ color:var(--red); }}
.notbet {{ margin-left:auto; font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
           border:1px solid var(--rule); padding:6px 10px; white-space:nowrap; }}
div[data-testid="stHorizontalBlock"]:has(a[data-testid="stPageLink-NavLink"]) {{
   margin:0 -40px 8px; padding:0 34px; background:var(--navy-deep); border-bottom:1px solid var(--rule);
   gap:2px; flex-wrap:wrap; }}
div[data-testid="stHorizontalBlock"]:has(a[data-testid="stPageLink-NavLink"]) > div {{ flex:0 0 auto !important; width:auto !important; min-width:0 !important; }}
a[data-testid="stPageLink-NavLink"] {{ background:transparent !important; padding:8px 12px; }}
a[data-testid="stPageLink-NavLink"] p {{ font-family:var(--display); font-size:15px; font-weight:600;
   letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
a[data-testid="stPageLink-NavLink"]:hover p {{ color:var(--stock); }}
a[data-testid="stPageLink-NavLink"][aria-current="page"] {{ box-shadow: inset 0 -2px 0 var(--red); }}
a[data-testid="stPageLink-NavLink"][aria-current="page"] p {{ color:var(--stock); }}

/* ---------- headings ---------- */
h1.pg {{ font-family:var(--display); font-weight:800; font-size:48px; line-height:1; letter-spacing:.02em;
         text-transform:uppercase; color:var(--stock); margin:36px 0 6px; }}
.pgsub {{ color:var(--muted); font-size:15px; margin:0 0 24px; }}
h2.sec {{ font-family:var(--display); font-weight:700; font-size:24px; letter-spacing:.08em; text-transform:uppercase;
          color:var(--stock); margin:40px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--rule); }}
.prose {{ color:var(--muted); font-size:15px; line-height:1.65; max-width:72ch; }}
.prose b {{ color:var(--stock); font-weight:600; }}

/* ---------- coupons ---------- */
.wallet {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(330px, 1fr)); gap:18px; }}
.cp {{ background:var(--stock); color:var(--ink); display:grid; grid-template-columns:1fr 96px;
       grid-template-rows:34px 1fr; min-height:212px; }}
.cp-hd {{ grid-column:1/-1; background:var(--red-head); color:var(--stock); display:flex; justify-content:space-between;
          align-items:center; padding:0 14px; font-family:var(--display); font-weight:700; font-size:14px;
          letter-spacing:.12em; text-transform:uppercase; }}
.cp-hd span:last-child {{ font-family:var(--num); font-weight:500; font-size:12px; letter-spacing:.04em; }}
.cp-bd {{ padding:14px 14px 12px; display:flex; flex-direction:column; justify-content:space-between; gap:12px; position:relative; }}
.cp-teams {{ font-family:var(--display); font-weight:800; font-size:24px; line-height:1.08; text-transform:uppercase; }}
.cp-teams small {{ display:block; font-family:var(--body); font-weight:500; font-size:13px; text-transform:none;
                   letter-spacing:0; color:#3c4660; margin-top:2px; }}
.fields {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; }}
.fd {{ border-top:1px solid var(--ink); padding-top:4px; font-family:var(--display); font-weight:600; font-size:12px;
       letter-spacing:.12em; text-transform:uppercase; color:#3c4660; }}
.fd b {{ display:block; font-family:var(--num); font-weight:500; font-size:26px; letter-spacing:0; color:var(--carbon);
         font-variant-numeric:tabular-nums; line-height:1.15; }}
.fd.fav b {{ font-weight:600; }}
.fd.fav {{ color:var(--ink); }}
.fd.actual {{ outline:1px solid var(--ink); outline-offset:3px; }}
.stub {{ border-left:1px dashed rgba(13,27,61,.45); display:flex; flex-direction:column; align-items:center;
         justify-content:center; gap:2px; font-family:var(--display); font-weight:600; font-size:12px;
         letter-spacing:.12em; text-transform:uppercase; color:#3c4660; text-align:center; padding:8px; }}
.stub b {{ font-family:var(--num); font-weight:600; font-size:26px; letter-spacing:0; color:var(--carbon); }}
.stub i {{ font-style:normal; font-family:var(--num); font-size:12px; letter-spacing:0; }}
.stamp {{ position:absolute; right:10px; top:12px; transform:rotate(-9deg); font-family:var(--display); font-weight:800;
          font-size:20px; letter-spacing:.16em; padding:2px 10px; text-transform:uppercase;
          animation: press .5s cubic-bezier(.16,1,.3,1) both; }}
.stamp.hit {{ color:var(--ink); border:2px solid var(--ink); }}
.stamp.miss {{ color:var(--red); border:2px dashed var(--red); text-decoration:line-through; text-decoration-thickness:2px; }}
@keyframes press {{ from {{ transform:rotate(-9deg) scale(1.35); opacity:0; }} to {{ transform:rotate(-9deg) scale(1); opacity:1; }} }}
@media (prefers-reduced-motion: reduce) {{ .stamp, .issue {{ animation:none !important; }} }}

/* ---------- ledger (ruled fare boxes on navy) ---------- */
.ledger {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); border:1px solid var(--rule); }}
.lg {{ padding:12px 14px; border-right:1px solid var(--rule); }}
.lg:last-child {{ border-right:0; }}
.lg .l {{ font-family:var(--display); font-weight:600; font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
.lg .v {{ font-family:var(--num); font-size:28px; color:var(--stock); font-variant-numeric:tabular-nums; margin-top:4px; }}
.lg .n {{ font-size:13px; color:var(--muted); margin-top:2px; }}
.lg .v.good {{ color:var(--carbon-navy); }}

/* ---------- issued forecast (predictor) ---------- */
.issue {{ background:var(--stock); color:var(--ink); margin-top:8px; animation: print .7s cubic-bezier(.16,1,.3,1) both; }}
@keyframes print {{ from {{ clip-path: inset(0 0 100% 0); }} to {{ clip-path: inset(0 0 0 0); }} }}
.is-hd {{ background:var(--red-head); color:var(--stock); display:flex; justify-content:space-between; padding:9px 18px;
          font-family:var(--display); font-weight:700; font-size:15px; letter-spacing:.14em; text-transform:uppercase; }}
.route {{ display:grid; grid-template-columns:1fr auto 1fr; align-items:end; padding:18px 22px 10px; gap:18px; }}
.route .c {{ font-family:var(--display); font-weight:800; font-size:clamp(56px, 8vw, 96px); line-height:.9; letter-spacing:.04em; }}
.route .c small {{ display:block; font-family:var(--body); font-weight:500; font-size:14px; letter-spacing:0; color:#3c4660; margin-top:6px; }}
.route .c.r {{ text-align:right; }}
.route .vs {{ font-family:var(--display); font-weight:700; font-size:18px; letter-spacing:.2em; color:var(--red); padding-bottom:22px; }}
.is-fields {{ display:grid; grid-template-columns:repeat(4, 1fr); border-top:1px solid var(--ink); }}
.is-fields .fd {{ border-top:0; border-right:1px solid rgba(13,27,61,.3); padding:10px 18px 14px; }}
.is-fields .fd:last-child {{ border-right:0; }}
.is-fields .fd b {{ font-size:34px; }}
.carbon {{ background:var(--carbon-copy); color:var(--carbon); padding:14px 18px; display:grid;
           grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); gap:12px; border-top:1px dashed rgba(75,46,131,.5); }}
.carbon .fd {{ border-top:1px solid var(--carbon); color:var(--carbon); }}
.carbon .fd b {{ font-size:24px; }}
.note {{ font-size:13px; color:var(--muted); border:1px solid var(--rule); padding:10px 12px; margin-top:14px; max-width:none; }}
.note b {{ color:var(--stock); }}

/* ---------- ruled tables ---------- */
table.rt {{ width:100%; border-collapse:collapse; font-size:15px; }}
table.rt th {{ font-family:var(--display); font-weight:600; font-size:13px; letter-spacing:.12em; text-transform:uppercase;
               color:var(--muted); text-align:right; padding:8px 10px; border-bottom:1px solid var(--stock); }}
table.rt td {{ padding:9px 10px; border-bottom:1px solid var(--rule); text-align:right; font-family:var(--num);
               font-variant-numeric:tabular-nums; color:var(--stock); }}
table.rt th.l, table.rt td.l {{ text-align:left; }}
table.rt td.team {{ font-family:var(--body); font-weight:600; }}
table.rt tr:hover td {{ background:var(--navy-raised); }}
.zone {{ display:inline-block; width:8px; height:8px; margin-right:8px; vertical-align:middle; }}
.zone.ucl {{ background:var(--carbon-navy); }} .zone.rel {{ border:1px dashed var(--red); }}
.tw {{ overflow-x:auto; }}

/* ---------- Streamlit controls in the wallet vocabulary ---------- */
.stButton > button, .stFormSubmitButton > button {{ background:var(--red); color:var(--stock); border:1px solid var(--red);
   font-family:var(--display); font-weight:700; font-size:16px; letter-spacing:.16em; text-transform:uppercase;
   padding:10px 28px; min-height:48px; }}
.stButton > button:hover {{ background:#B81E25; border-color:#B81E25; color:var(--stock); }}
.stButton > button:active {{ background:#9E1A20; }}
[data-baseweb="select"] > div {{ background:var(--navy-raised) !important; border:1px solid var(--rule) !important; }}
[data-testid="stWidgetLabel"] p {{ font-family:var(--display); font-weight:600; font-size:13px !important;
   letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{ font-family:var(--display); font-weight:600; letter-spacing:.12em; text-transform:uppercase;
   border:1px solid var(--rule); border-bottom:0; padding:6px 16px; background:transparent; }}
.stTabs [aria-selected="true"] {{ background:var(--stock); color:var(--ink); }}
.stTabs [data-baseweb="tab-highlight"] {{ display:none; }}
[data-testid="stPlotlyChart"] {{ border:1px solid var(--rule); padding:6px; }}
.foot {{ margin-top:56px; padding-top:14px; border-top:1px solid var(--rule); font-size:13px; color:var(--muted); }}

@media (max-width: 760px) {{
  .block-container {{ padding:0 16px 48px; }}
  .bar {{ margin:0 -16px; padding:0 16px; height:56px; }}
  .notbet {{ display:none; }}
  div[data-testid="stHorizontalBlock"]:has(a[data-testid="stPageLink-NavLink"]) {{ margin:0 -16px 8px; padding:0 10px; }}
  h1.pg {{ font-size:36px; margin-top:20px; }}
  .wallet {{ grid-template-columns:1fr; }}
  .route {{ padding:14px; }} .is-fields {{ grid-template-columns:repeat(2, 1fr); }}
}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def coupon(home, away, p_home, p_draw, p_away, head_left, head_right, stub_label, stub_value,
           stub_sub="", actual=None, stamp=None, sub=None) -> str:
    """One fixture coupon. actual: 'H'|'D'|'A' outlines the field that happened;
    stamp: 'hit'|'miss' presses the settled stamp."""
    probs = {"H": p_home, "D": p_draw, "A": p_away}
    fav = max(probs, key=probs.get)
    fields = "".join(
        f'<div class="fd{" fav" if k == fav else ""}{" actual" if k == actual else ""}">{lbl}<b>{pct(probs[k])}</b></div>'
        for k, lbl in (("H", "Home"), ("D", "Draw"), ("A", "Away")))
    stamp_html = f'<div class="stamp {stamp}">{"Hit" if stamp == "hit" else "Miss"}</div>' if stamp else ""
    sub_html = f"<small>{esc(sub)}</small>" if sub else ""
    return (f'<div class="cp" role="group" aria-label="{esc(home)} v {esc(away)}">'
            f'<div class="cp-hd"><span>{esc(head_left)}</span><span>{esc(head_right)}</span></div>'
            f'<div class="cp-bd">{stamp_html}<div class="cp-teams">{esc(home)}<br>v {esc(away)}{sub_html}</div>'
            f'<div class="fields">{fields}</div></div>'
            f'<div class="stub">{esc(stub_label)}<b>{esc(stub_value)}</b><i>{esc(stub_sub)}</i></div></div>')


def ledger(items) -> str:
    """items: list of (label, value, note, good: bool)."""
    cells = "".join(f'<div class="lg"><div class="l">{esc(l)}</div><div class="v{" good" if g else ""}">{esc(v)}</div>'
                    f'<div class="n">{esc(n)}</div></div>' for l, v, n, g in items)
    return f'<div class="ledger">{cells}</div>'


def top_bar():
    st.markdown('<div class="bar"><div class="wordmark">EPL <b>Forecast</b></div>'
                '<div class="notbet">Academic project · not for betting</div></div>', unsafe_allow_html=True)


def footer(metrics: dict):
    through = metrics.get("data_through", "unknown")
    st.markdown(
        f'<div class="foot">Results to {esc(through)} · data from football-data.co.uk and FotMob · '
        f'refreshed weekly by the pipeline. <b>Academic and research project:</b> forecasts are statistical '
        f'estimates and are not intended for betting or any financial decision.</div>', unsafe_allow_html=True)
