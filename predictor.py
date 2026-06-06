import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

from feature_engine import load_assets, build_inference_row

# label_map from pkl: {0: 'Draw', 1: 'AwayWin', 2: 'HomeWin'}
OUTCOME_COLORS = {
    "Home Win": "#00d4aa",
    "Draw":     "#f59e0b",
    "Away Win": "#f43f5e",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_stacked_proba(home_team: str, away_team: str) -> np.ndarray:
    """
    Run XGB + RF base models → stack → meta LogReg.
    Returns shape-(3,) array ordered [Draw, AwayWin, HomeWin]
    matching label_map {0:'Draw', 1:'AwayWin', 2:'HomeWin'}.
    """
    assets = load_assets()
    xgb_model  = assets["best_xgb_model"]
    rf_model   = assets["best_rf_model"]
    meta_model = assets["stacked_meta_model"]
    final_features = assets["final_features"]

    row = build_inference_row(home_team, away_team)
    X   = row[final_features].values

    xgb_p = xgb_model.predict_proba(X)   # (1, 3)
    rf_p  = rf_model.predict_proba(X)    # (1, 3)

    meta_X       = np.hstack([xgb_p, rf_p])          # (1, 6)
    final_proba  = meta_model.predict_proba(meta_X)[0] # (3,)
    final_classes = meta_model.classes_               # class indices in model order

    # Build [Draw, AwayWin, HomeWin] in label_map order (0,1,2)
    label_map = assets["label_map"]  # {0:'Draw', 1:'AwayWin', 2:'HomeWin'}
    stacked = np.zeros(3)
    for i, cls_idx in enumerate(final_classes):
        stacked[int(cls_idx)] = float(final_proba[i])

    # Normalise
    stacked /= stacked.sum()
    return stacked   # [Draw_prob, AwayWin_prob, HomeWin_prob]


def _get_avg_scored(home_team: str, away_team: str) -> tuple[float, float]:
    """
    Get Home_Avg_Scored for home_team and Away_Avg_Scored for away_team
    from df_poisson — each team's own most recent expanding average,
    exactly as computed by get_leakage_free_strengths():

      Home_Avg_Scored = expanding mean of (FTHG + home_xG)/2 for that
                        team's home matches (shift=1, leakage-free).
      Away_Avg_Scored = expanding mean of (FTAG + away_xG)/2 for that
                        team's away matches (shift=1, leakage-free).
    """
    assets = load_assets()
    df = assets.get("df_poisson")
    team_map = assets["team_map"]
    h_id = team_map.get(home_team)
    a_id = team_map.get(away_team)
    DEFAULT_XG = 1.35

    if df is None:
        return DEFAULT_XG, DEFAULT_XG

    # Home team: most recent Home_Avg_Scored from any home match
    if h_id is not None:
        home_rows = df[df["HomeTeam"] == h_id].sort_values("Date")
        h_exp = float(home_rows["Home_Avg_Scored"].iloc[-1]) if not home_rows.empty else DEFAULT_XG
    else:
        h_exp = DEFAULT_XG

    # Away team: most recent Away_Avg_Scored from any away match
    if a_id is not None:
        away_rows = df[df["AwayTeam"] == a_id].sort_values("Date")
        a_exp = float(away_rows["Away_Avg_Scored"].iloc[-1]) if not away_rows.empty else DEFAULT_XG
    else:
        a_exp = DEFAULT_XG

    return h_exp, a_exp



# ─── Calibrated Poisson ──────────────────────────────────────────────────────

def calibrated_poisson_matrix(
    stacked_proba: np.ndarray,
    h_exp: float,
    a_exp: float,
    max_goals: int = 5,
) -> np.ndarray:
    """
    Build a raw Poisson score matrix then calibrate each zone (H/D/A)
    using the stacking model probabilities as scaling targets.

    stacked_proba must be [Draw_prob, AwayWin_prob, HomeWin_prob].
    Returns a normalised (max_goals+1 x max_goals+1) matrix.
    """
    epsilon = 1e-9
    goals = np.arange(max_goals + 1)

    # Raw Poisson matrix
    h_pmf = stats.poisson.pmf(goals, max(h_exp, 0.01))
    a_pmf = stats.poisson.pmf(goals, max(a_exp, 0.01))
    raw   = np.outer(h_pmf, a_pmf)

    # Raw zone sums
    raw_draw     = np.sum(np.diag(raw))
    raw_home_win = np.sum(np.tril(raw, k=-1))
    raw_away_win = np.sum(np.triu(raw, k=1))
    total_raw    = raw_draw + raw_home_win + raw_away_win

    raw_norm = np.array([
        raw_draw     / (total_raw + epsilon),  # Draw
        raw_away_win / (total_raw + epsilon),  # AwayWin
        raw_home_win / (total_raw + epsilon),  # HomeWin
    ])

    # Scaling factors: stacking / raw_poisson
    scaling = stacked_proba / (raw_norm + epsilon)  # [Draw, Away, Home]

    # Apply zone scaling
    calibrated = np.zeros_like(raw)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h == a:
                calibrated[h, a] = raw[h, a] * scaling[0]   # Draw
            elif h > a:
                calibrated[h, a] = raw[h, a] * scaling[2]   # Home Win
            else:
                calibrated[h, a] = raw[h, a] * scaling[1]   # Away Win

    # Normalise
    total_cal = calibrated.sum()
    if total_cal > 0:
        calibrated /= total_cal
    else:
        calibrated = raw / raw.sum()

    return calibrated


def extract_outcome_probs(matrix: np.ndarray) -> dict:
    """Extract H/D/A probabilities from a score matrix."""
    home_win = float(np.sum(np.tril(matrix, k=-1)))
    draw     = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, k=1)))
    total    = home_win + draw + away_win
    if total > 0:
        home_win /= total; draw /= total; away_win /= total
    return {"Home Win": home_win, "Draw": draw, "Away Win": away_win}


# ─── Main predict function ────────────────────────────────────────────────────

def predict_match(home_team: str, away_team: str) -> dict:
    """
    Full inference pipeline:
      1. Stacked model (XGB + RF → LogReg) → stacked_proba [D, AW, HW]
      2. Look up Home/Away_Avg_Scored lambdas from df_poisson
      3. Calibrated Poisson matrix
      4. Return outcome probs, most likely score, Poisson display data
    """
    # Step 1: stacked model probabilities
    stacked = _get_stacked_proba(home_team, away_team)

    # Step 2: expected goals (lambdas)
    h_exp, a_exp = _get_avg_scored(home_team, away_team)

    # Step 3: calibrated Poisson
    matrix = calibrated_poisson_matrix(stacked, h_exp, a_exp, max_goals=5)

    # Step 4: outcome probabilities from calibrated matrix
    proba = extract_outcome_probs(matrix)

    predicted  = max(proba, key=proba.get)
    confidence = proba[predicted]

    # Most likely score
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely = (int(idx[0]), int(idx[1]))

    # Also build uncapped Poisson for distribution display (max_goals=8)
    goals8 = np.arange(9)
    h_pmf8 = stats.poisson.pmf(goals8, max(h_exp, 0.01))
    a_pmf8 = stats.poisson.pmf(goals8, max(a_exp, 0.01))

    return {
        "proba":             proba,
        "stacked_proba":     {"Draw": stacked[0], "Away Win": stacked[1], "Home Win": stacked[2]},
        "predicted_outcome": predicted,
        "confidence":        confidence,
        "matrix":            matrix,
        "most_likely_score": most_likely,
        "lambda_h":          h_exp,
        "lambda_a":          a_exp,
        "h_pmf8":            h_pmf8,
        "a_pmf8":            a_pmf8,
        "home_team":         home_team,
        "away_team":         away_team,
    }
