"""Calibration validation for the SHARP (curated) 1X2 odds.

Purpose: the draw-calibration fix blends kicktipp 1X2 *toward* sharp 1X2. That is
only legitimate if the sharp odds are themselves well-calibrated. This module
measures that on the 115 curated closing-odds matches (WC2022 + Euro2024 in
``src/odds/historical.py``) joined to their realised scores in
``data/processed/results_clean.csv``:

  * DRAW-class reliability bins (predicted P(draw) vs realised draw frequency),
  * multiclass Brier score and log-loss for the 1X2 outcome,
  * for both de-vig methods (shin vs normalise).

IMPORTANT SCOPE NOTE: this validates the *sharp* odds (the blend target). The
kicktipp->blend improvement itself can only be validated live going forward,
because no historical kicktipp odds exist (kicktipp is scraped live only). That
forward validation is what ``src/snapshot.py`` accumulates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import normalise_team
from src.odds.devig import devig_1x2
from src.odds.historical import EURO_2024_ODDS, WC_2022_ODDS

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_CSV = _ROOT / "data" / "processed" / "results_clean.csv"

# Outcome encoding from the (odds-home, odds-away) perspective.
_H, _D, _A = 0, 1, 2


# -- Pure metrics (unit-testable on synthetic data) -------------------


def multiclass_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean multiclass Brier score for 1X2. probs:(n,3), outcomes:(n,) in {0,1,2}.

    Brier = mean_i sum_k (p_ik - 1{outcome_i==k})^2  ∈ [0, 2].
    """
    probs = np.asarray(probs, float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def log_loss_1x2(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the realised 1X2 outcome."""
    probs = np.clip(np.asarray(probs, float), eps, 1.0)
    p_actual = probs[np.arange(len(outcomes)), outcomes]
    return float(-np.log(p_actual).mean())


def draw_reliability(
    p_draw: np.ndarray,
    is_draw: np.ndarray,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Reliability table for the draw class: predicted vs realised per bin."""
    if bins is None:
        bins = [0.0, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0]
    p_draw = np.asarray(p_draw, float)
    is_draw = np.asarray(is_draw, float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_draw >= lo) & (p_draw < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"[{lo:.2f},{hi:.2f})",
            "n": int(m.sum()),
            "mean_predicted": round(float(p_draw[m].mean()), 3),
            "realised_freq": round(float(is_draw[m].mean()), 3),
        })
    return pd.DataFrame(rows)


# -- Join curated odds -> realised scores ------------------------------


def _result_lookup(results_csv: str | Path | None = None) -> dict:
    """{(frozenset{home,away}, 'YYYY-MM-DD') -> (results_home_team, hs, as)}."""
    p = Path(results_csv) if results_csv else _RESULTS_CSV
    df = pd.read_csv(p, parse_dates=["date"])
    lk: dict = {}
    for r in df.itertuples(index=False):
        key = (frozenset({r.home_team, r.away_team}), str(r.date.date()))
        lk[key] = (r.home_team, int(r.home_score), int(r.away_score))
    return lk


def collect_sharp_predictions(method: str = "shin", results_csv: str | Path | None = None):
    """Devig the curated odds and align to realised outcomes.

    Returns (probs (n,3), outcomes (n,), n_total, n_matched).
    """
    lk = _result_lookup(results_csv)
    probs, outcomes = [], []
    records = WC_2022_ODDS + EURO_2024_ODDS
    for rec in records:
        h, a = normalise_team(rec["home_team"]), normalise_team(rec["away_team"])
        key = (frozenset({h, a}), rec["date"])
        if key not in lk:
            continue
        res_home, hs, as_ = lk[key]
        # orient scores to the odds' (home, away) ordering
        gh, ga = (hs, as_) if res_home == h else (as_, hs)
        p_h, p_d, p_a = devig_1x2(rec["h2h_odds"][0], rec["h2h_odds"][1], rec["h2h_odds"][2], method=method)
        probs.append([p_h, p_d, p_a])
        outcomes.append(_H if gh > ga else (_D if gh == ga else _A))
    return np.array(probs), np.array(outcomes), len(records), len(outcomes)


def evaluate_sharp_calibration(results_csv: str | Path | None = None) -> str:
    """Build and format the full calibration report (shin vs normalise)."""
    lines = ["Sharp-odds calibration on curated closing odds (WC2022 + Euro2024)"]
    for method in ("shin", "normalise"):
        probs, outcomes, n_total, n_matched = collect_sharp_predictions(method, results_csv)
        if n_matched == 0:
            lines.append(f"\n[{method}] no matches joined ({n_total} odds records).")
            continue
        brier = multiclass_brier(probs, outcomes)
        ll = log_loss_1x2(probs, outcomes)
        draw_rate = float((outcomes == _D).mean())
        rel = draw_reliability(probs[:, _D], outcomes == _D)
        lines.append(
            f"\n[{method}]  matched {n_matched}/{n_total} | "
            f"multiclass Brier={brier:.4f}  log-loss={ll:.4f}  "
            f"realised draw rate={draw_rate:.3f}  mean pred P(draw)={probs[:, _D].mean():.3f}"
        )
        lines.append("  draw reliability (predicted vs realised):")
        lines.append("    " + rel.to_string(index=False).replace("\n", "\n    "))
    lines.append("\nNOTE: validates the SHARP odds only. The kicktipp->blend gain "
                 "is validated live going forward via src/snapshot.py.")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(evaluate_sharp_calibration())
