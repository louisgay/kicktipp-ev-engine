"""Backtesting framework for evaluating prediction strategies.

Trains a model on data up to a cutoff date, then evaluates predictions
on a target tournament (e.g., World Cup 2022) without any data leakage.
Compares the model's optimal pool predictions against baselines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.base import ScorePredictor
from src.models.dixon_coles import DixonColesModel
from src.models.poisson import DoublePoissonModel
from src.scoring.kicktipp import expected_points, optimal_prediction, points

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


# -- Baseline strategies ----------------------------------------------

def baseline_favorite_1_0(home_team: str, away_team: str,
                          elo_home: float, elo_away: float) -> tuple[int, int]:
    """Predict 1-0 for the Elo favourite, 0-1 if away is stronger."""
    if elo_home >= elo_away:
        return (1, 0)
    return (0, 1)


def baseline_always_1_1(home_team: str, away_team: str,
                        elo_home: float, elo_away: float) -> tuple[int, int]:
    """Always predict 1-1."""
    return (1, 1)


def baseline_always_1_0(home_team: str, away_team: str,
                        elo_home: float, elo_away: float) -> tuple[int, int]:
    """Always predict 1-0 (home win)."""
    return (1, 0)


BASELINES = {
    "favorite_1_0": baseline_favorite_1_0,
    "always_1_1": baseline_always_1_1,
    "always_1_0": baseline_always_1_0,
}


# -- Calibration ------------------------------------------------------

def compute_calibration(predictions: list[dict], n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration: do predicted probabilities match observed frequencies?

    For each match, we look at the predicted probability of the actual outcome.
    We bin these probabilities and compare predicted vs observed.
    """
    probs = []
    hits = []
    for pred in predictions:
        matrix = pred["prob_matrix"]
        ah, aa = pred["actual"]
        max_g = matrix.shape[0] - 1
        if ah <= max_g and aa <= max_g:
            p_actual = matrix[ah, aa]
        else:
            p_actual = 0.0
        probs.append(p_actual)
        hits.append(1)  # The actual outcome always "happened"

    # Also check tendency calibration
    # P(home win), P(draw), P(away win) vs actual
    tendency_data = []
    for pred in predictions:
        matrix = pred["prob_matrix"]
        ah, aa = pred["actual"]

        p_home_win = np.tril(matrix, k=-1).sum()  # i > j
        p_draw = np.trace(matrix)
        p_away_win = np.triu(matrix, k=1).sum()   # i < j

        if ah > aa:
            actual_tendency = "home"
        elif ah == aa:
            actual_tendency = "draw"
        else:
            actual_tendency = "away"

        tendency_data.append({
            "p_home_win": p_home_win,
            "p_draw": p_draw,
            "p_away_win": p_away_win,
            "actual": actual_tendency,
        })

    # Binned calibration for exact-score probabilities
    probs = np.array(probs)
    bin_edges = np.linspace(0, probs.max() + 0.01, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if mask.sum() > 0:
            rows.append({
                "bin_low": bin_edges[i],
                "bin_high": bin_edges[i+1],
                "mean_predicted": probs[mask].mean(),
                "n_matches": int(mask.sum()),
            })

    # Tendency accuracy
    tdf = pd.DataFrame(tendency_data)
    tendency_acc = {
        "home_win_accuracy": (
            ((tdf["actual"] == "home") &
             (tdf["p_home_win"] > tdf[["p_draw", "p_away_win"]].max(axis=1)))
            .sum() / (tdf["actual"] == "home").sum()
            if (tdf["actual"] == "home").sum() > 0 else 0
        ),
        "draw_accuracy": (
            ((tdf["actual"] == "draw") &
             (tdf["p_draw"] > tdf[["p_home_win", "p_away_win"]].max(axis=1)))
            .sum() / (tdf["actual"] == "draw").sum()
            if (tdf["actual"] == "draw").sum() > 0 else 0
        ),
        "away_win_accuracy": (
            ((tdf["actual"] == "away") &
             (tdf["p_away_win"] > tdf[["p_home_win", "p_draw"]].max(axis=1)))
            .sum() / (tdf["actual"] == "away").sum()
            if (tdf["actual"] == "away").sum() > 0 else 0
        ),
    }

    return pd.DataFrame(rows), tendency_acc


# -- Main backtest ----------------------------------------------------

def run_backtest(
    results_df: pd.DataFrame,
    elo_df: pd.DataFrame | None = None,
    model_class: type[ScorePredictor] = DixonColesModel,
    model_kwargs: dict | None = None,
    cutoff_date: str | None = None,
    tournament: str | None = None,
    tournament_year: int | None = None,
) -> dict:
    """Run a full backtest.

    1. Split data at cutoff_date.
    2. Train model on pre-cutoff data.
    3. For each match in the target tournament after cutoff:
       - Generate probability matrix
       - Select optimal pool prediction
       - Score against actual result
    4. Compare with baselines.

    Returns dict with results, predictions, and comparison tables.
    """
    cfg = load_config()
    if cutoff_date is None:
        cutoff_date = cfg["backtest"]["cutoff_date"]
    if tournament is None:
        tournament = cfg["backtest"]["tournament"]
    if tournament_year is None:
        tournament_year = cfg["backtest"]["tournament_year"]
    if model_kwargs is None:
        model_kwargs = {}

    cutoff = pd.Timestamp(cutoff_date)
    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["date"])

    # Split
    train_start = pd.Timestamp(cfg["model"]["train_start"])
    train = results_df[(results_df["date"] >= train_start) &
                       (results_df["date"] < cutoff)].copy()

    # Find test matches (target tournament in the right year)
    test = results_df[
        (results_df["date"] >= cutoff) &
        (results_df["tournament"].str.contains(tournament, case=False, na=False)) &
        (results_df["date"].dt.year == tournament_year)
    ].copy()

    logger.info("Training on %d matches (%s to %s)",
                len(train), train["date"].min().date(), train["date"].max().date())
    logger.info("Testing on %d matches (%s)",
                len(test), tournament)

    if len(test) == 0:
        logger.error("No test matches found for %s %d!", tournament, tournament_year)
        return {"error": "No test matches found"}

    # Fit model
    model = model_class(**model_kwargs)
    model.fit(train)

    # Prepare Elo lookup for baselines
    elo_lookup: dict[str, float] = {}
    if elo_df is not None:
        elo_df["date"] = pd.to_datetime(elo_df["date"])
        # Get latest pre-cutoff Elo for each team
        pre_cutoff_elo = elo_df[elo_df["date"] < cutoff]
        if len(pre_cutoff_elo) > 0:
            last = pre_cutoff_elo.iloc[-1]
            # Build lookup from both home and away columns
            for _, row in pre_cutoff_elo.tail(500).iterrows():
                elo_lookup[row["home_team"]] = row["elo_home_after"]
                elo_lookup[row["away_team"]] = row["elo_away_after"]

    # Evaluate each test match
    predictions = []
    model_points_total = 0
    baseline_points = {name: 0 for name in BASELINES}

    for _, match in test.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        actual = (int(match["home_score"]), int(match["away_score"]))
        neutral = bool(match.get("neutral", True))  # WC matches are neutral

        # Check if teams are in the model
        if home not in model.team_idx_ or away not in model.team_idx_:
            logger.warning("Skipping %s vs %s: team not in training data",
                           home, away)
            continue

        # Model prediction
        prob_matrix = model.predict_score_matrix(home, away, neutral=neutral)
        best_pred, best_ev = optimal_prediction(prob_matrix)
        pts = points(best_pred, actual)
        model_points_total += pts

        # Baselines
        elo_h = elo_lookup.get(home, 1500.0)
        elo_a = elo_lookup.get(away, 1500.0)
        for name, fn in BASELINES.items():
            b_pred = fn(home, away, elo_h, elo_a)
            baseline_points[name] += points(b_pred, actual)

        predictions.append({
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "actual": actual,
            "predicted": best_pred,
            "expected_points": best_ev,
            "actual_points": pts,
            "prob_matrix": prob_matrix,
        })

        logger.debug("%s vs %s: actual=%s, pred=%s, pts=%d (EV=%.2f)",
                     home, away, actual, best_pred, pts, best_ev)

    # Summary
    n_matches = len(predictions)
    calibration_df, tendency_acc = compute_calibration(predictions)

    summary = {
        "n_matches": n_matches,
        "model_total_points": model_points_total,
        "model_avg_points": model_points_total / n_matches if n_matches > 0 else 0,
        "baselines": {
            name: {"total": pts, "avg": pts / n_matches if n_matches > 0 else 0}
            for name, pts in baseline_points.items()
        },
        "tendency_accuracy": tendency_acc,
    }

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS: %s %d (%d matches)",
                tournament, tournament_year, n_matches)
    logger.info("=" * 60)
    logger.info("Model (optimal EV):  %d pts (%.2f avg)",
                model_points_total,
                model_points_total / n_matches if n_matches else 0)
    for name, pts in baseline_points.items():
        logger.info("Baseline %-15s: %d pts (%.2f avg)",
                    name, pts, pts / n_matches if n_matches else 0)
    logger.info("Tendency accuracy: %s", tendency_acc)

    return {
        "summary": summary,
        "predictions": predictions,
        "calibration": calibration_df,
    }


def predictions_to_dataframe(predictions: list[dict]) -> pd.DataFrame:
    """Convert predictions list to a readable DataFrame."""
    rows = []
    for p in predictions:
        rows.append({
            "date": p["date"],
            "match": f"{p['home_team']} vs {p['away_team']}",
            "actual_score": f"{p['actual'][0]}-{p['actual'][1]}",
            "predicted_score": f"{p['predicted'][0]}-{p['predicted'][1]}",
            "points": p["actual_points"],
            "expected_points": round(p["expected_points"], 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config()
    processed = _ROOT / cfg["paths"]["processed_data"]

    results = pd.read_csv(processed / "results_clean.csv", parse_dates=["date"])
    elo = pd.read_csv(processed / "elo_history.csv", parse_dates=["date"])

    result = run_backtest(results, elo)
    if "error" not in result:
        df = predictions_to_dataframe(result["predictions"])
        print("\n" + df.to_string(index=False))
        print(f"\nTotal: {result['summary']['model_total_points']} points")
