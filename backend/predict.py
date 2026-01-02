from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Load trained model
model = joblib.load(MODELS / "logreg_baseline.joblib")

# Load feature tables
team_season = pd.read_csv(DATA / "regular_team_season_features.csv")
seeds_path = DATA / "tourney_seeds.csv"
seeds = pd.read_csv(seeds_path) if seeds_path.exists() else None


#Helper functions
def get_team_features(season: int, team_id: int):
    row = team_season.query(
        "Season == @season and TeamID == @team_id"
    )
    if row.empty:
        raise ValueError(f"No data for TeamID={team_id} in Season={season}")
    return row.iloc[0]


def get_seed(season: int, team_id: int):
    if seeds is None:
        return None
    row = seeds.query(
        "Season == @season and TeamID == @team_id"
    )
    if row.empty:
        return None
    return int(row.iloc[0]["SeedNum"])

def build_matchup_features(season: int, teamA_id: int, teamB_id: int):
    A = get_team_features(season, teamA_id)
    B = get_team_features(season, teamB_id)

    features = {
        "win_pct_diff": A["win_pct"] - B["win_pct"],
        "avg_margin_diff": A["avg_margin"] - B["avg_margin"],
        "avg_pf_diff": A["avg_pf"] - B["avg_pf"],
        "avg_pa_diff": A["avg_pa"] - B["avg_pa"],
        "std_margin_diff": A["std_margin"] - B["std_margin"],
    }

    seedA = get_seed(season, teamA_id)
    seedB = get_seed(season, teamB_id)
    if seedA is not None and seedB is not None:
        # lower seed number = better
        features["seed_diff"] = seedB - seedA

    return pd.DataFrame([features])

def predict_matchup(season: int, teamA_id: int, teamB_id: int):
    X = build_matchup_features(season, teamA_id, teamB_id)
    proba_A = model.predict_proba(X)[0, 1]
    return {
        "teamA_win_prob": round(proba_A, 4),
        "teamB_win_prob": round(1 - proba_A, 4),
    }

if __name__ == "__main__":
    SEASON = 2023
    TEAM_A = 1345 #Purdue
    TEAM_B = 1266 #Marquette

    result = predict_matchup(SEASON, TEAM_A, TEAM_B)
    print(result)


