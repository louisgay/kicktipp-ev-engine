"""Comparative backtest: all strategies on a common test set.

Compares on kicktipp points, RPS, log-loss, calibration, and
breakdown (exact / goal-diff / tendency / wrong).

Strategies:
1. Dixon-Coles Elo (Phase 1)
2. Odds-reconstructed (normalisation)
3. Odds-reconstructed (Shin)
4. Ensemble (odds + Dixon-Coles)
5. Baselines: favorite_1_0, always_1_1
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.evaluate import (
    BASELINES,
    compute_calibration,
    load_config,
    predictions_to_dataframe,
)
from src.models.base import ScorePredictor
from src.models.dixon_coles import DixonColesModel
from src.models.odds_model import EnsembleModel, OddsModel
from src.scoring.kicktipp import optimal_prediction, points

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


def _rps_single(prob_matrix: np.ndarray, actual: tuple[int, int]) -> float:
    """Ranked Probability Score for a 3-outcome event (H/D/A).

    RPS = (1/2) * sum_{k=1}^{2} (CDF_pred(k) - CDF_actual(k))^2

    Lower is better. Measures calibration across the ordered outcomes.
    """
    # Extract 1X2 probabilities
    p_h = np.tril(prob_matrix, k=-1).sum()
    p_d = np.trace(prob_matrix)
    p_a = np.triu(prob_matrix, k=1).sum()
    pred_cdf = np.cumsum([p_h, p_d, p_a])

    # Actual outcome
    ah, aa = actual
    if ah > aa:
        actual_vec = [1, 0, 0]
    elif ah == aa:
        actual_vec = [0, 1, 0]
    else:
        actual_vec = [0, 0, 1]
    actual_cdf = np.cumsum(actual_vec)

    rps = 0.5 * np.sum((pred_cdf - actual_cdf) ** 2)
    return float(rps)


def _logloss_single(prob_matrix: np.ndarray, actual: tuple[int, int]) -> float:
    """Log-loss for the exact score outcome."""
    ah, aa = actual
    max_g = prob_matrix.shape[0] - 1
    if ah <= max_g and aa <= max_g:
        p = prob_matrix[ah, aa]
    else:
        p = 1e-10
    return -np.log(max(p, 1e-10))


def _points_breakdown(pred: tuple[int, int], actual: tuple[int, int]) -> str:
    """Categorise result: exact / goal_diff / tendency / wrong."""
    pts = points(pred, actual)
    if pts == 4:
        return "exact"
    elif pts == 3:
        return "goal_diff"
    elif pts == 2:
        return "tendency"
    else:
        return "wrong"


def run_comparative_backtest(
    test_matches: pd.DataFrame,
    strategies: dict[str, ScorePredictor | None],
    elo_lookup: dict[str, float] | None = None,
) -> dict:
    """Run all strategies on the same test set.

    Parameters
    ----------
    test_matches : DataFrame with date, home_team, away_team,
                   home_score, away_score, neutral, tournament
    strategies : dict mapping strategy_name -> fitted ScorePredictor
                 (None for baselines, handled internally)
    elo_lookup : optional dict team -> Elo for baselines

    Returns
    -------
    Dict with 'results_df' (per-match details), 'summary_df' (aggregates),
    and 'calibration' per strategy.
    """
    if elo_lookup is None:
        elo_lookup = {}

    all_results = []

    for _, match in test_matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        actual = (int(match["home_score"]), int(match["away_score"]))
        neutral = bool(match.get("neutral", True))
        match_date = match["date"]

        row = {
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "actual": f"{actual[0]}-{actual[1]}",
        }

        # Model strategies
        for name, model in strategies.items():
            if model is None:
                continue
            try:
                if home not in model.team_idx_ or away not in model.team_idx_:
                    # Skip if team not known
                    row[f"{name}_pred"] = None
                    row[f"{name}_pts"] = None
                    row[f"{name}_ev"] = None
                    row[f"{name}_rps"] = None
                    row[f"{name}_logloss"] = None
                    row[f"{name}_category"] = None
                    continue

                matrix = model.predict_score_matrix(
                    home, away, match_date, neutral
                )
                pred, ev = optimal_prediction(matrix)
                pts = points(pred, actual)
                rps = _rps_single(matrix, actual)
                ll = _logloss_single(matrix, actual)
                cat = _points_breakdown(pred, actual)

                row[f"{name}_pred"] = f"{pred[0]}-{pred[1]}"
                row[f"{name}_pts"] = pts
                row[f"{name}_ev"] = round(ev, 3)
                row[f"{name}_rps"] = round(rps, 4)
                row[f"{name}_logloss"] = round(ll, 4)
                row[f"{name}_category"] = cat
                row[f"{name}_matrix"] = matrix  # for calibration

            except Exception as e:
                logger.warning("Strategy %s failed on %s vs %s: %s",
                               name, home, away, e)
                row[f"{name}_pred"] = None
                row[f"{name}_pts"] = None

        # Baselines
        elo_h = elo_lookup.get(home, 1500.0)
        elo_a = elo_lookup.get(away, 1500.0)
        for bl_name, bl_fn in BASELINES.items():
            bl_pred = bl_fn(home, away, elo_h, elo_a)
            bl_pts = points(bl_pred, actual)
            row[f"bl_{bl_name}_pred"] = f"{bl_pred[0]}-{bl_pred[1]}"
            row[f"bl_{bl_name}_pts"] = bl_pts
            row[f"bl_{bl_name}_category"] = _points_breakdown(bl_pred, actual)

        all_results.append(row)

    # Build summary
    strategy_names = [n for n in strategies if strategies[n] is not None]
    baseline_names = [f"bl_{n}" for n in BASELINES]
    all_names = strategy_names + baseline_names

    summary_rows = []
    for name in all_names:
        pts_col = f"{name}_pts"
        cat_col = f"{name}_category"

        # Filter rows where this strategy has data
        valid = [r for r in all_results if r.get(pts_col) is not None]
        if not valid:
            continue

        total_pts = sum(r[pts_col] for r in valid)
        n = len(valid)
        avg_pts = total_pts / n if n > 0 else 0

        cats = [r.get(cat_col, "wrong") for r in valid]
        n_exact = cats.count("exact")
        n_gdiff = cats.count("goal_diff")
        n_tend = cats.count("tendency")
        n_wrong = cats.count("wrong")

        row_summary = {
            "strategy": name,
            "n_matches": n,
            "total_pts": total_pts,
            "avg_pts": round(avg_pts, 3),
            "exact": n_exact,
            "goal_diff": n_gdiff,
            "tendency": n_tend,
            "wrong": n_wrong,
        }

        # Auto-EV (model's expected points per match)
        ev_col = f"{name}_ev"
        if ev_col in valid[0]:
            ev_vals = [r[ev_col] for r in valid if r.get(ev_col) is not None]
            if ev_vals:
                row_summary["avg_ev"] = round(np.mean(ev_vals), 3)

        # RPS and logloss (only for model strategies)
        rps_col = f"{name}_rps"
        ll_col = f"{name}_logloss"
        if rps_col in valid[0]:
            rps_vals = [r[rps_col] for r in valid if r.get(rps_col) is not None]
            ll_vals = [r[ll_col] for r in valid if r.get(ll_col) is not None]
            if rps_vals:
                row_summary["mean_rps"] = round(np.mean(rps_vals), 4)
            if ll_vals:
                row_summary["mean_logloss"] = round(np.mean(ll_vals), 4)

        summary_rows.append(row_summary)

    summary_df = pd.DataFrame(summary_rows)

    # Clean up matrix columns from results (not serialisable)
    results_clean = []
    for r in all_results:
        results_clean.append({k: v for k, v in r.items()
                              if not k.endswith("_matrix")})

    return {
        "results": results_clean,
        "results_df": pd.DataFrame(results_clean),
        "summary_df": summary_df,
        "raw_results": all_results,  # includes matrices for calibration
    }


def print_summary(summary_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table."""
    cols = ["strategy", "n_matches", "total_pts", "avg_pts",
            "exact", "goal_diff", "tendency", "wrong"]
    extra = [c for c in ["avg_ev", "mean_rps", "mean_logloss"]
             if c in summary_df.columns]
    cols += extra

    print("\n" + "=" * 80)
    print("COMPARATIVE BACKTEST RESULTS")
    print("=" * 80)
    print(summary_df[cols].to_string(index=False))
    print("=" * 80)

    # Highlight winner
    best = summary_df.loc[summary_df["total_pts"].idxmax()]
    print(f"\nBest strategy on kicktipp points: {best['strategy']} "
          f"({best['total_pts']} pts, {best['avg_pts']:.3f} avg)")
