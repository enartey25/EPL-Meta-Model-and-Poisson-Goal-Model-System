# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two audiences, weighted equally (confirmed):

- **Football fans** checking forecasts for upcoming Premier League fixtures. Their job: see who is likely to win, the likely scoreline, and goal markets for a match, quickly.
- **Portfolio reviewers** (recruiters, lecturers, peers) judging the author's machine-learning work. Their job: understand the method, see honest evidence that it works, and see that it runs live.

## Product Purpose

A live English Premier League forecasting system. For any fixture it gives win/draw/loss probabilities from a stacked model (Random Forest and an opponent-adjusted Poisson goal model, combined by a logistic-regression meta-learner). It also gives an exact-score grid and goal markets from Poisson rates implied by those probabilities. A scheduled pipeline refreshes the data, ratings and models every week. Success means fans get a clear forecast, and reviewers come away convinced the forecasts are honest and well calibrated.

## Positioning

Every claim is backed by a proper scoring rule and an out-of-sample record. The site publishes its own track record, misses included, against a naive baseline, instead of cherry-picked hits.

## Operating Context

- Streamlit app (`app.py`) deployed from the GitHub repository's `main` branch.
- A GitHub Actions workflow runs every Wednesday at 04:00 UTC. It fetches results (football-data.co.uk) and match xG (FotMob), rebuilds features and models, and commits the assets.
- Seasons tracked: 2022/23 onward; the current season is 2026/27.

## Capabilities and Constraints

- Pages: Home, Match Predictor, Season Analytics, Head to Head, League Table. The redesign may restructure them; keeping all five was not required.
- Forecasts are probabilities; outcomes are labeled Home Win / Draw / Away Win.
- The track record is computed from real results only. Walk-forward forecasts (models trained before the season, features updated pre-match) must be labeled as such. Forecasts logged before kickoff are labeled as logged.
- Newly promoted clubs start from priors (1420 Elo, league-average xG); the UI says so when a prior is in use.

## Brand Commitments

- **Academic disclaimer must stay prominent**: the forecasts are for academic and research purposes and are not intended for betting (confirmed).
- **Dark theme** (confirmed).

## Evidence on Hand

- 2025/26 holdout season (370 matches): stacked model log loss 1.0183 vs a 1.0866 naive baseline; Brier 0.6123 vs 0.6578.
- 2025/26 matchday 38 backtest: the model scored worse than the baseline (1.230 vs 1.107). It must not be hidden.
- The 2026/27 live record is computed by the pipeline from results; no numbers may be typed in by hand.
- No testimonials, users, press, or betting-return figures exist; none may be invented.

## Product Principles

1. Probabilities, not picks: show uncertainty as it is.
2. The record is the proof: publish hits and misses together, against a baseline.
3. Fan question first, method one step away.
4. Nothing hard-coded that the pipeline can compute.
