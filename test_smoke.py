import warnings
warnings.filterwarnings("ignore")
from data_loader import load_full_dataset, fetch_xg_data, compute_league_table
from feature_engine import get_teams, get_elo_state, build_inference_row
from predictor import predict_match
from visualizations import plot_result_distribution, plot_elo_rankings

print("All imports OK")
teams = get_teams()
print(f"Teams loaded: {len(teams)}")
elo = get_elo_state()
print(f"ELO state: {len(elo)} teams")
row = build_inference_row("Arsenal", "Liverpool")
print(f"Inference row shape: {row.shape}")
result = predict_match("Arsenal", "Liverpool")
pred = result["predicted_outcome"]
conf = result["confidence"]
print(f"Prediction: {pred} ({conf*100:.1f}%)")
print(f"Proba: {result['proba']}")
score = result["most_likely_score"]
print(f"Most likely score: {score}")
print(f"lambda_h={result['lambda_h']:.3f}  lambda_a={result['lambda_a']:.3f}")
print("ALL TESTS PASSED")
