---
name: EPL Forecast
description: Premier League forecasts issued as coupons before kickoff and stamped after the result.
colors:
  carrier-navy: "#0D1B3D"
  carrier-navy-deep: "#08122B"
  carrier-navy-raised: "#14244A"
  coupon-stock: "#F2F0EA"
  stock-ink: "#0D1B3D"
  stock-ink-muted: "#3C4660"
  jet-red: "#D7262D"
  jet-red-press: "#B81E25"
  jet-red-pressed: "#9E1A20"
  coupon-head-red: "#C41E26"
  carbon-purple: "#4B2E83"
  carbon-copy-lilac: "#DCD3F0"
  carbon-on-navy: "#C3B4EE"
  muted-on-navy: "#AEB9D6"
  rule-on-navy: "rgba(242,240,234,0.22)"
typography:
  display:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "48px"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.02em"
  headline:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "24px"
    fontWeight: 700
    letterSpacing: "0.08em"
  title:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "24px"
    fontWeight: 800
    lineHeight: 1.08
  body:
    fontFamily: "Barlow, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    letterSpacing: "0.12em"
  numeral:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "26px"
    fontWeight: 500
    lineHeight: 1.15
    letterSpacing: "0em"
    fontFeature: "tnum"
rounded:
  none: "0px"
spacing:
  hair: "4px"
  xs: "10px"
  sm: "14px"
  md: "18px"
  lg: "24px"
  xl: "40px"
components:
  button-primary:
    backgroundColor: "{colors.jet-red}"
    textColor: "{colors.coupon-stock}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "10px 28px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.jet-red-press}"
    textColor: "{colors.coupon-stock}"
  button-primary-active:
    backgroundColor: "{colors.jet-red-pressed}"
    textColor: "{colors.coupon-stock}"
  coupon:
    backgroundColor: "{colors.coupon-stock}"
    textColor: "{colors.stock-ink}"
    rounded: "{rounded.none}"
  coupon-head:
    backgroundColor: "{colors.coupon-head-red}"
    textColor: "{colors.coupon-stock}"
    padding: "6px 14px"
    height: "34px"
  carbon-slip:
    backgroundColor: "{colors.carbon-copy-lilac}"
    textColor: "{colors.carbon-purple}"
    rounded: "{rounded.none}"
  carbon-slip-head:
    backgroundColor: "{colors.carbon-purple}"
    textColor: "{colors.coupon-stock}"
    padding: "9px 14px"
  select-field:
    backgroundColor: "{colors.carrier-navy-raised}"
    textColor: "{colors.coupon-stock}"
    rounded: "{rounded.none}"
  tab-selected:
    backgroundColor: "{colors.coupon-stock}"
    textColor: "{colors.stock-ink}"
    padding: "6px 16px"
  nav-link:
    textColor: "{colors.muted-on-navy}"
    padding: "0 10px"
    height: "40px"
  nav-link-current:
    textColor: "{colors.coupon-stock}"
---

# Design System: EPL Forecast

## Overview

**Creative North Star: "The Ticket Wallet"**

Every forecast is printed matter: a coupon of off-white stock issued on a carrier-navy ground, torn along a dashed stub, and later rubber-stamped with how it went. The system borrows the vernacular of jet-age airline tickets and fare slips (condensed tracked caps, ruled fields, monospaced carbon numerals, a lilac carbon copy for derived figures) and applies it with the discipline of a data product. The navy is the wallet; stock is the only bright surface; red marks the heads of issued documents and the single action that issues one.

Density is moderate and gridded. Coupons sit in an auto-filling wallet grid beside a narrower right-hand rail; tables and charts are hairline-ruled on navy, never boxed in cards. Nothing is rounded and nothing floats: depth comes from paper colour against the ground, not from shadows. Settled outcomes are always shown, hits and misses alike, and the miss is drawn in a different line form, not only a different colour.

The shipped build departs from its direction notes in one recorded place: coupon and forecast heads use a deeper head red (coupon-head-red) rather than jet-red, because stock-coloured caps on that red read more firmly at 12-15px, and the MISS stamp uses the same head red for contrast on stock. Jet-red remains the primary action, the wordmark accent, the current-page underline and the "worse than baseline" series in charts.

**Key Characteristics:**
- Carrier navy ground with coupon-stock documents; zero corner radius everywhere.
- Condensed tracked caps for heads, labels, tabs and nav; monospaced tabular numerals for every probability and score.
- One-pixel hairline rules and dashed tear lines carry structure.
- Rubber stamps settle a coupon: solid boxed HIT, dashed struck MISS.
- Derived and aggregate figures live on a lilac carbon-copy slip.

## Colors

A two-paper palette, navy wallet and off-white stock, with one red family for issuance and one purple family for carbon numerals and derived figures.

### Primary
- **Jet Red** (jet-red): the primary action button, the wordmark accent, the current nav underline (2px inset), the "v" between club codes on an issued forecast, and chart series that mean "worse than baseline" or "against". Pressed states step down to Jet Red Press and Jet Red Pressed.
- **Coupon Head Red** (coupon-head-red): the head band of every coupon and issued forecast, and the MISS stamp. It is the document's printed header, not an action.

### Secondary
- **Carbon Purple** (carbon-purple): every numeral printed on stock (probabilities, likely score, ledger values on the slip) and the slip's head band.
- **Carbon Copy Lilac** (carbon-copy-lilac): the background of the season-ledger slip and the derived-markets strip under an issued forecast; also the top stop of the score-grid heatmap.
- **Carbon on Navy** (carbon-on-navy): the purple carried onto navy: model series in charts, "better than baseline" values, the Champions League zone marker.

### Neutral
- **Carrier Navy** (carrier-navy): the page ground. Also used as stock-ink, the text colour on stock, and as the punch colour of the stub notches.
- **Carrier Navy Deep** (carrier-navy-deep): the top bar and nav band.
- **Carrier Navy Raised** (carrier-navy-raised): select fields and table row hover.
- **Coupon Stock** (coupon-stock): coupon and forecast paper, primary text on navy, the selected tab.
- **Stock Ink Muted** (stock-ink-muted): field labels, stub labels and club full names on stock.
- **Muted on Navy** (muted-on-navy): subtitles, prose, ledger labels, table headers, inactive nav, chart ticks.
- **Rule on Navy** (rule-on-navy): every hairline on navy: section underlines, table rows, ledger cells, the not-for-betting pill border, chart frames. Charts use a fainter 14% stock for gridlines.

### Named Rules
**The Two Papers Rule.** Navy is the ground and stock is the paper. Any surface a reader should pick up and read as one forecast is stock; everything else stays on navy with hairlines.

**The Red Issues Rule.** Red marks issuance only: document heads, the action that issues a forecast, the current page, and adverse results. It is never decoration or a generic highlight.

**The Carbon Numerals Rule.** Numbers printed on stock are carbon purple; numbers on navy are stock (or carbon-on-navy when they mean "model ahead").

## Typography

**Display Font:** Barlow Condensed (with Arial Narrow, sans-serif)
**Body Font:** Barlow (with system-ui, sans-serif)
**Label/Mono Font:** IBM Plex Mono (with ui-monospace, monospace) for all numerals

**Character:** A narrow airline grotesk set in tracked capitals does the talking for every head and label; Barlow keeps prose quiet; IBM Plex Mono prints the figures like a carbon impact, tabular and aligned.

### Hierarchy
- **Display** (800, 48px, 1.0, 0.02em, uppercase; 36px under 760px): one page heading, e.g. the gameweek.
- **Route codes** (800, clamp(56px, 8vw, 96px), 0.9, 0.04em): three-letter club codes on an issued forecast only.
- **Headline** (700, 24px, 0.08em, uppercase, hairline underneath): section heads.
- **Title** (800, 24px, 1.08, uppercase; 20px when the coupon is under 320px wide): team names on a coupon.
- **Head band** (700, 14-15px, 0.12-0.14em, uppercase): coupon, forecast and slip heads.
- **Body** (400, 15px, 1.6-1.65, max 72ch): prose, subtitles, notes (13px).
- **Label** (600, 12-13px, 0.12em, uppercase): field labels, stub labels, ledger labels, widget labels, table headers. Nav links and tabs use the same face at 15px.
- **Numeral** (IBM Plex Mono 500, 26px on coupons, 34px on issued fields, 28-32px in ledgers, tabular): every probability, score and metric. The favourite field steps to 600.

### Named Rules
**The Tracked Caps Rule.** Anything that names a field, a document or a place to go is condensed, uppercase and tracked 0.08-0.16em. Running prose is never set in caps.

**The Mono Figures Rule.** Every probability, score, log loss and count is IBM Plex Mono with tabular figures, so columns of coupons align by eye.

## Layout

Pages run in a single container up to 1440px with 40px side padding (16px under 760px). The top bar (52px, navy-deep) and the nav band beneath it bleed to the container edges. The fixtures page splits 3.3 : 1 between the coupon wallet and a right rail holding the carbon slip and notes; one dominant area per screen.

The wallet is an auto-fill grid of coupons, minimum 280px each, 18px gaps, collapsing to one column under 760px. Coupons themselves are a two-column grid: body plus a fixed 108px stub, under a 34px head. Ledgers auto-fit cells of at least 150px. The issued forecast uses a three-column route (code, v, code), then a four-column field row (two under 760px), then an auto-fit carbon strip.

Spacing steps between 4px (label to rule), 10-14px (inside fields and cells), 18px (grid gaps, forecast padding), 24px (after subtitles), and 36-40px (above page and section heads). The footer sits 56px below content on a hairline.

## Elevation & Depth

The system is flat. There are no drop shadows anywhere. Depth is the contrast between stock paper and the navy ground, plus the deeper navy of the top bar. Where `box-shadow` appears it is used only to draw hairlines (ledger cell rules, the 2px current-nav underline), never to lift a surface.

### Named Rules
**The Paper Not Plastic Rule.** Surfaces never cast shadows or glow. A thing is "above" the ground only by being paper-coloured.

## Shapes

Zero radius on every element, enforced globally. Structure is drawn with 1px rules: solid hairlines on navy at 22% stock, solid ink rules above fields on stock, and dashed lines where paper would tear (the coupon stub, the top of a carbon slip or strip). The stub is punched with two small navy diamonds (11px squares rotated 45°) at its top and bottom where it meets the body. Stamps are the only rotated shape (-9°).

### Named Rules
**The One Pixel Rule.** Rules are one pixel. The only heavier lines are the HIT stamp (3px), the MISS stamp (2px dashed) and the outlined "what happened" field (1px box).

**The Line Form Rule.** Outcome is carried by line form as well as colour: solid box for HIT, dashed and struck through for MISS; in tables, a filled carbon square for hit and a dashed red square with struck text for miss.

## Components

### Buttons
Blunt and printed, one per screen.
- **Shape:** square corners (0px), 1px border in the fill colour, minimum 48px tall.
- **Primary:** jet red fill, stock text in condensed tracked caps (700, 16px, 0.16em), 10px 28px padding. Used for "Issue forecast".
- **Hover / Focus:** steps to jet-red-press; active steps to jet-red-pressed. Keyboard focus draws a 2px stock outline offset 2px (global).

### Fixture Coupon (signature)
The unit of the whole product.
- **Head:** coupon-head-red band, 34px, kickoff in tracked caps left, reference or provenance in mono right.
- **Body:** team names as title caps, full names in body type below when shown; three fields (Home / Draw / Away) each with a 1px ink rule above, a caps label, and a carbon mono numeral. The favourite's numeral is heavier; the field that happened is boxed in a 1px ink outline.
- **Stub:** 108px, separated by a dashed tear line with punched notches; a caps label ("Likely" or "Final"), a carbon numeral score, and a mono footnote.
- **Stamp:** when settled, a -9° stamp presses into the stub top: HIT in ink with a 3px solid border, MISS in head red with a 2px dashed border and struck text. It animates in once (0.5s, scale 1.35 to 1, expo-out) and is static under reduced motion.

### Carbon Slip
Derived figures, filed beside the coupons.
- **Head:** carbon purple band with stock caps.
- **Body:** lilac, a dashed carbon tear line under the head, single-column ledger cells divided by faint carbon rules; values in carbon mono at 32px, "better than baseline" values at 600 weight.

### Issued Forecast
The predictor's result: a full-width stock document under a head-red band. Club codes print large either side of a jet-red "v"; a four-field row (Home, Draw, Away, Likely score) sits on an ink rule with internal 1px dividers; a lilac carbon strip below holds goal markets and expected goals. It reveals top to bottom once (clip-path, 0.7s, expo-out), static under reduced motion.

### Ledger (on navy)
Ruled fare boxes: an auto-fit grid of cells bounded by navy hairlines, caps label, stock mono value (28px), muted note.

### Ruled Tables
Full-width, collapsed. Headers in muted caps with a stock rule beneath; rows on navy hairlines; figures right-aligned in mono, club names in Barlow 600, left-aligned. Row hover fills navy-raised. Zone markers are 8px squares: filled carbon-on-navy for top places, dashed red for relegation.

### Inputs / Fields
- **Style:** navy-raised fill, 1px navy hairline, square corners. Labels above in muted condensed caps (13px).
- **Focus:** the global 2px stock outline.

### Navigation
A navy-deep band under the top bar: condensed caps links (600, 15px, 0.12em), muted at rest, stock on hover. The current page turns stock with a 2px jet-red underline. The band wraps under 760px. The top bar carries the wordmark (800, 24px, 0.1em caps, second word in jet red) and, right-aligned, the not-for-betting pill: 12px muted caps in a 1px hairline box.

### Tabs
Folder tabs: condensed caps, 1px hairline on three sides, transparent at rest; the selected tab is a stock tab with ink text. No sliding highlight.

### Charts
Transparent on navy inside a 1px hairline frame with 6px padding. Titles are condensed caps at 16px, ticks muted, gridlines 14% stock, hover labels ink with a stock border and mono text. Series: model in carbon-on-navy, adverse in jet red, reference lines dotted muted, baselines as stock tick marks. The score grid heatmap runs dark navy to mid violet to lilac with 1px gaps.

## Do's and Don'ts

### Do:
- **Do** print every forecast or settled result as a stock coupon with a head-red band, ruled fields and a dashed stub.
- **Do** set every probability and score in IBM Plex Mono with tabular figures; carbon purple on stock, stock on navy.
- **Do** keep corners square (0px) and rules at one pixel; use dashed lines only where paper tears or for a miss.
- **Do** show misses alongside hits, distinguished by line form (dashed, struck) as well as by red.
- **Do** keep the academic not-for-betting pill visible in the top bar on every page.
- **Do** put derived and aggregate figures on the lilac carbon slip or strip, not on a new card style.
- **Do** honour reduced motion: stamps and the forecast print appear without animation.

### Don't:
- **Don't** round any corner or add drop shadows; depth is stock against navy only.
- **Don't** use jet red as decoration or a generic highlight; it marks issuance, the current page, the one primary action, and adverse results.
- **Don't** introduce a second primary button on a screen.
- **Don't** convey hit or miss by colour alone.
- **Don't** set prose in caps or numerals in the display or body face.
- **Don't** box charts or tables in stock cards; they live on navy behind hairlines.
