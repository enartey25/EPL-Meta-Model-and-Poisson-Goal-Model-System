---
version: 1
slug: "app-py"
primary_target: "app.py"
related_targets: []
---

# Surface: EPL Forecast app (app.py, all pages)

Mode: Operate. Audience: football fans checking the next gameweek, and portfolio reviewers checking method and record (equal weight). Job: read a forecast fast; verify that forecasts are honest. Constraints: dark theme, prominent academic not-for-betting notice, Streamlit standard controls, no hand-typed numbers.

## Direction contract

THESIS: Every forecast is a coupon issued before kickoff and stamped after the result, nothing disappears. Refuses the category default of a KPI-card dashboard with probability bars.

OWN-WORLD: Carrier navy ground (#0D1B3D) with coupon-stock cards (#F2F0EA), jet red (#D7262D) coupon heads and primary action, carbon purple (#4B2E83) numerals on stock, carbon-copy lilac panels for derived figures. Condensed airline grotesk in tracked caps for heads and tab labels, monospaced carbon numerals for every probability. Hairline-ruled fields, tabbed caps labels, dashed tear lines, rubber-stamp states.

STORY: The visitor sees this gameweek's coupons, reads home/draw/away and the likely score at a glance, then sees last gameweek's coupons stamped HIT or MISS and the season record against a naive baseline, and trusts it because the misses are shown too.

FIRST VIEWPORT: Navy top bar with wordmark, nav (Fixtures, Predictor, Track record, Analytics, Table) and the not-for-betting pill. Left: "Gameweek N" heading in large condensed caps with issue date; a three-column grid of fixture coupons (red head with kickoff and coupon number; club names; three ruled carbon fields Home/Draw/Away; torn stub with likely score). Right rail: the season ledger (model log loss vs baseline, matches, favourite-won rate). Primary action lives on the Predictor page: Issue forecast (jet red).

FORM: Ticket Wallet, a dealt challenger (catalog vernacular-ephemera-jet-age-ticket-wallet), chosen by the user over the assigned Floodlit Pitch (my list position 3) and my pick Ceefax (position 1). seed:592adf7d7ced4aca. Raises kept from declined challengers: one-device-pixel rules, no rounded corners; hit/miss carried by line form (solid stamp vs struck/dashed) as well as colour; one dominant area per screen.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
