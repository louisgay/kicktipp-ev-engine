"""Pure compute layer for the Streamlit demo — NO streamlit/plotly imports.

Everything the app renders is derived here from the real `src/` + `analysis/`
primitives (never a forked model), so this module is unit-testable on its own and
CI can smoke-test it without a browser. The Streamlit UI in `streamlit_app.py` only
adds widgets and charts on top of these functions.
"""
from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np
import pandas as pd

# Make `src` / `analysis` importable no matter where streamlit launches from.
REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The odds reconstruction logs a WARNING when the market fit isn't perfect; that's
# expected and noisy in a UI, so quiet it to errors only.
logging.getLogger("src.odds.reconstruct").setLevel(logging.ERROR)

from src.odds.devig import devig_1x2  # noqa: E402
from src.odds.reconstruct import reconstruct_matrix  # noqa: E402
from src.scoring.kicktipp import ev_table, optimal_prediction  # noqa: E402

SNAPSHOTS = REPO / "data" / "history" / "snapshots.csv"
PICKS = REPO / "data" / "opponents" / "picks.csv"
GRID = 5  # show scorelines 0..5 for both teams


# ── Tab 1 — the pick explorer ────────────────────────────────────────────

def load_snapshots() -> pd.DataFrame:
    """Banked pre-kickoff market snapshots, one usable 3-way match per row."""
    df = pd.read_csv(SNAPSHOTS)
    df = df[(df["kt_draw"] > 1e-3) & df["ou_over_2_5"].notna()].copy()
    df["label"] = (
        "MD" + df["spieltag"].astype(int).astype(str)
        + " · " + df["home"] + " vs " + df["away"]
    )
    return df.reset_index(drop=True)


def devig_odds(home_odds: float, draw_odds: float, away_odds: float,
               method: str = "shin") -> tuple[float, float, float]:
    """Decimal 1X2 odds → fair (de-vigged) probabilities."""
    ph, pdr, pa = devig_1x2(home_odds, draw_odds, away_odds, method=method)
    return float(ph), float(pdr), float(pa)


def score_bundle(p_home: float, p_draw: float, p_away: float,
                 p_over_2_5: float) -> dict:
    """The heart of the demo: fair probs → Dixon-Coles matrix → EV table → pick.

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


# ── Tab 2 — the field model ──────────────────────────────────────────────

def field_table() -> tuple[pd.DataFrame, dict]:
    """Per-player shrunk follow-rate / draw-share from the hierarchical field model."""
    from src.field_model import FieldModel

    fm = FieldModel.from_disk()
    players = sorted(set(pd.read_csv(PICKS)["player"]))
    rows = [
        {
            "player": p,
            "follow_rate": fm.follow_rate(p),
            "draw_share": fm.draw_share(p),
            "discriminating_picks": fm.discriminating_count(p),
        }
        for p in players
    ]
    df = pd.DataFrame(rows).sort_values("follow_rate", ascending=False).reset_index(drop=True)
    pop = {"pop_follow": float(fm.pop_follow), "pop_draw_share": float(fm.pop_draw_share)}
    return df, pop


# ── Tab 3 — the honest behavioral edge ───────────────────────────────────

def edge_bundle() -> dict:
    """Run the market-odds behavioral-edge anchor and shape it for display."""
    import contextlib
    import io

    from analysis.behavioral_edge import run

    with contextlib.redirect_stdout(io.StringIO()):  # run() prints a text report
        out = run()
    leaks = pd.DataFrame(
        [
            {
                "player": lk.player,
                "n": lk.n,
                "leak_per_match": lk.leak_per_match,
                "from_draws": lk.leak_from_draws,
                "from_other": lk.leak_from_other,
                "n_draws": lk.n_draws,
            }
            for lk in out["leaks"].values()
        ]
    ).sort_values("leak_per_match", ascending=False).reset_index(drop=True)

    paired = pd.DataFrame(
        [
            {
                "opponent": pr.opponent,
                "n": pr.n,
                "mean_edge_per_match": pr.mu_d,
                "t_now": pr.t_now,
                "t_proj_104": pr.t_proj_104,
                "detectable_now": pr.t_now >= 2.0,
            }
            for pr in out["paired"]
        ]
    ).sort_values("t_now", ascending=False).reset_index(drop=True)

    reg = out["reg"]
    reg_summary = {
        "slope": float(reg.slope),
        "ci_lo": float(reg.ci_lo),
        "ci_hi": float(reg.ci_hi),
        "r2": float(reg.r2),
        "p_value": float(reg.p_value),
        "n": int(reg.n),
    }
    return {"leaks": leaks, "paired": paired, "reg": reg_summary}
