"""Pure compute layer for the Streamlit demo (no streamlit/plotly imports).

Everything the app renders is derived here from the real src/ primitives (never a
forked model), so this module is unit-testable on its own and CI can smoke-test it
without a browser. The Streamlit UI in streamlit_app.py only adds widgets and
charts on top of these functions.
"""
from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np
import pandas as pd

# Make src importable no matter where streamlit launches from.
REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# reconstruct_lambdas logs a WARNING when the market fit is not perfect; that is
# expected and noisy in a UI, so quiet it to errors only.
logging.getLogger("src.odds.reconstruct").setLevel(logging.ERROR)

from src.odds.devig import devig_1x2  # noqa: E402
from src.odds.reconstruct import reconstruct_matrix  # noqa: E402
from src.scoring.kicktipp import ev_table, optimal_prediction  # noqa: E402

SNAPSHOTS = REPO / "data" / "history" / "snapshots.csv"
GRID = 5  # show scorelines 0..5 for both teams


def load_snapshots() -> pd.DataFrame:
    """Banked pre-kickoff market snapshots, one usable 3-way match per row."""
    df = pd.read_csv(SNAPSHOTS)
    df = df[(df["kt_draw"] > 1e-3) & df["ou_over_2_5"].notna()].copy()
    df["label"] = (
        "MD" + df["spieltag"].astype(int).astype(str)
        + " - " + df["home"] + " vs " + df["away"]
    )
    return df.reset_index(drop=True)


def devig_odds(home_odds: float, draw_odds: float, away_odds: float,
               method: str = "shin") -> tuple[float, float, float]:
    """Decimal 1X2 odds to fair (de-vigged) probabilities."""
    ph, pdr, pa = devig_1x2(home_odds, draw_odds, away_odds, method=method)
    return float(ph), float(pdr), float(pa)


def score_bundle(p_home: float, p_draw: float, p_away: float,
                 p_over_2_5: float) -> dict:
    """Fair probs to Dixon-Coles matrix to EV table to the EV-max pick.

    Returns everything the two heatmaps and the callouts need.
    """
    matrix = reconstruct_matrix(p_home, p_draw, p_away, p_over_2_5)
    prob = matrix[: GRID + 1, : GRID + 1]
    ev = ev_table(matrix, max_pred=GRID)
    ml = tuple(int(x) for x in np.unravel_index(np.argmax(prob), prob.shape))
    pick, ev_val = optimal_prediction(matrix, max_pred=GRID)
    return {
        "prob": prob,
        "ev": ev,
        "most_likely": ml,
        "most_likely_p": float(prob[ml]),
        "ev_pick": (int(pick[0]), int(pick[1])),
        "ev_value": float(ev_val),
        "ev_of_most_likely": float(ev[ml]),
        "agree": ml == (int(pick[0]), int(pick[1])),
    }
