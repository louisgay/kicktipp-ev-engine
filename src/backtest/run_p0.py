"""P0 Backtest: comparative evaluation of all strategies.

Runs on WC 2022 (64 matches) and Euro 2024 (51 matches).
Compares: Dixon-Coles Elo, odds-normalise, odds-Shin, ensemble, baselines.

Usage:
    python -m src.backtest.run_p0
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.compare import print_summary, run_comparative_backtest
from src.data.clean import normalise_team
from src.models.dixon_coles import DixonColesModel
from src.models.odds_model import EnsembleModel, OddsModel
from src.odds.historical import load_historical_odds

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


def _match_odds_to_results(
    odds_records: list[dict],
    test_matches: pd.DataFrame,
) -> list[dict]:
    """Match odds records to test matches by normalised team names.

    Returns odds records with team names normalised to match the
    cleaned results data.
    """
    matched = []
    test_pairs = set()
    for _, m in test_matches.iterrows():
        test_pairs.add((m["home_team"], m["away_team"]))

    for rec in odds_records:
        h = normalise_team(rec["home_team"])
        a = normalise_team(rec["away_team"])
        if (h, a) in test_pairs:
            matched.append({**rec, "home_team": h, "away_team": a})
        else:
            logger.debug("Odds record %s vs %s not matched to any test match",
                         h, a)

    logger.info("Matched %d/%d odds records to test matches",
                len(matched), len(odds_records))
    return matched


def _build_elo_lookup(elo_df: pd.DataFrame, cutoff: str) -> dict[str, float]:
    """Build team -> latest Elo lookup from history, before cutoff."""
    elo_df = elo_df[pd.to_datetime(elo_df["date"]) < pd.Timestamp(cutoff)]
    lookup: dict[str, float] = {}
    for _, row in elo_df.tail(2000).iterrows():
        lookup[row["home_team"]] = row["elo_home_after"]
        lookup[row["away_team"]] = row["elo_away_after"]
    return lookup


def run_p0_backtest(tournament: str = "both") -> dict:
    """Run the full P0 comparative backtest.

    Parameters
    ----------
    tournament : 'wc2022', 'euro2024', or 'both'
    """
    cfg = _load_config()

    # Load cleaned results
    results_path = _ROOT / cfg["paths"]["processed_data"] / "results_clean.csv"
    results_df = pd.read_csv(results_path, parse_dates=["date"])

    # Load Elo history
    elo_path = _ROOT / cfg["paths"]["processed_data"] / "elo_history.csv"
    elo_df = pd.read_csv(elo_path, parse_dates=["date"])

    all_summaries = []
    all_results = []

    tournaments_to_run = []
    if tournament in ("wc2022", "both"):
        tournaments_to_run.append({
            "name": "World Cup 2022",
            "odds_key": "wc2022",
            "cutoff": "2022-11-20",
            "tournament_filter": "FIFA World Cup",
            "year": 2022,
        })
    if tournament in ("euro2024", "both"):
        tournaments_to_run.append({
            "name": "Euro 2024",
            "odds_key": "euro2024",
            "cutoff": "2024-06-14",
            "tournament_filter": "UEFA Euro",
            "year": 2024,
        })

    for t_cfg in tournaments_to_run:
        logger.info("=" * 60)
        logger.info("Running P0 backtest: %s", t_cfg["name"])
        logger.info("=" * 60)

        cutoff = pd.Timestamp(t_cfg["cutoff"])
        train_start = pd.Timestamp(cfg["model"]["train_start"])

        # Training data (before cutoff)
        train = results_df[
            (results_df["date"] >= train_start) &
            (results_df["date"] < cutoff)
        ].copy()

        # Test data (tournament matches)
        test = results_df[
            (results_df["date"] >= cutoff) &
            (results_df["tournament"].str.contains(
                t_cfg["tournament_filter"], case=False, na=False)) &
            (results_df["date"].dt.year == t_cfg["year"])
        ].copy()

        logger.info("Training: %d matches, Testing: %d matches",
                     len(train), len(test))

        if len(test) == 0:
            logger.warning("No test matches found for %s!", t_cfg["name"])
            continue

        # Elo lookup
        elo_lookup = _build_elo_lookup(elo_df, t_cfg["cutoff"])

        # -- Strategy 1: Dixon-Coles Elo ------------------------------
        dc_model = DixonColesModel(
            max_goals=cfg["model"]["max_goals"],
            half_life_days=cfg["model"]["half_life_days"],
        )
        dc_model.fit(train)

        # -- Strategy 2: Odds (normalisation) -------------------------
        odds_data = load_historical_odds(t_cfg["odds_key"])
        odds_matched = _match_odds_to_results(odds_data, test)

        odds_norm = OddsModel(
            devig_method="normalise",
            rho=dc_model.params_[-1] if dc_model.params_ is not None else -0.04,
            max_goals=cfg["model"]["max_goals"],
        )
        odds_norm.load_odds(odds_matched)

        # -- Strategy 3: Odds (Shin) ----------------------------------
        odds_shin = OddsModel(
            devig_method="shin",
            rho=dc_model.params_[-1] if dc_model.params_ is not None else -0.04,
            max_goals=cfg["model"]["max_goals"],
        )
        odds_shin.load_odds(odds_matched)

        # -- Strategy 4: Ensemble (DC 30% + odds 70%) -----------------
        # Use normalise-devigged odds + DC model (weights 0.3 / 0.7)
        ensemble = EnsembleModel(
            models=[dc_model, odds_norm],
            weights=[0.3, 0.7],  # odds are usually sharper
            max_goals=cfg["model"]["max_goals"],
        )

        strategies = {
            "dc_elo": dc_model,
            "odds_norm": odds_norm,
            "odds_shin": odds_shin,
            "ensemble_30_70": ensemble,
        }

        result = run_comparative_backtest(
            test_matches=test,
            strategies=strategies,
            elo_lookup=elo_lookup,
        )

        result["summary_df"]["tournament"] = t_cfg["name"]
        all_summaries.append(result["summary_df"])

        result["results_df"]["tournament"] = t_cfg["name"]
        all_results.append(result["results_df"])

        print(f"\n{'='*60}")
        print(f"  {t_cfg['name']}")
        print(f"{'='*60}")
        print_summary(result["summary_df"])

    # Combined summary
    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)
        # Aggregate across tournaments
        agg = combined.groupby("strategy").agg({
            "n_matches": "sum",
            "total_pts": "sum",
            "exact": "sum",
            "goal_diff": "sum",
            "tendency": "sum",
            "wrong": "sum",
        }).reset_index()
        agg["avg_pts"] = (agg["total_pts"] / agg["n_matches"]).round(3)

        # Add avg_ev, mean_rps and mean_logloss if available
        for col in ["avg_ev", "mean_rps", "mean_logloss"]:
            if col in combined.columns:
                rps_agg = combined.groupby("strategy")[col].mean().reset_index()
                agg = agg.merge(rps_agg, on="strategy", how="left")

        print(f"\n{'='*60}")
        print("  COMBINED (all tournaments)")
        print(f"{'='*60}")
        print_summary(agg)

        return {
            "per_tournament": all_summaries,
            "combined": agg,
            "all_results": pd.concat(all_results, ignore_index=True),
        }

    return {
        "per_tournament": all_summaries,
        "all_results": pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_p0_backtest("both")
