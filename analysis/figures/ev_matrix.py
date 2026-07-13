"""Generate the README hero figure: P(scoreline) vs E[points] side by side.

The whole thesis of the engine in one picture — the scoreline that is *most
likely* (argmax of the probability matrix, left) is usually **not** the scoreline
that *maximises expected pool points* under the asymmetric 4/3/2/0 rules (argmax
of the EV table, right). We pick the right-hand cell, not the left-hand one.

Reproducible, offline: reads the banked market snapshot, reconstructs a
Dixon-Coles score matrix from the de-vigged 1X2 + O/U 2.5 odds, then scores every
candidate scoreline.

    python -m analysis.figures.ev_matrix        # -> docs/img/ev_matrix.png
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.odds.reconstruct import reconstruct_matrix
from src.scoring.kicktipp import ev_table, optimal_prediction

REPO = pathlib.Path(__file__).resolve().parents[2]
SNAPSHOTS = REPO / "data" / "history" / "snapshots.csv"
OUT = REPO / "docs" / "img" / "ev_matrix.png"

GRID = 5  # show scorelines 0..5 for both teams


def _pick_illustrative_match(df: pd.DataFrame) -> pd.Series:
    """Prefer a clear-favourite match where most-likely != EV-max (the point)."""
    best = None
    for _, row in df.iterrows():
        if not np.isfinite(row.get("ou_over_2_5", np.nan)):
            continue
        matrix = reconstruct_matrix(
            row["kt_home"], row["kt_draw"], row["kt_away"], row["ou_over_2_5"]
        )
        ml = np.unravel_index(np.argmax(matrix[: GRID + 1, : GRID + 1]), (GRID + 1, GRID + 1))
        ev_pick, _ = optimal_prediction(matrix, max_pred=GRID)
        fav = max(row["kt_home"], row["kt_away"])
        if tuple(ml) != tuple(ev_pick) and 0.45 < fav < 0.75:
            # a decisive-but-not-crushing favourite makes the clearest picture
            return row
        if best is None:
            best = row
    return best


def main() -> None:
    df = pd.read_csv(SNAPSHOTS)
    row = _pick_illustrative_match(df)

    matrix = reconstruct_matrix(
        row["kt_home"], row["kt_draw"], row["kt_away"], row["ou_over_2_5"]
    )
    prob = matrix[: GRID + 1, : GRID + 1]
    ev = ev_table(matrix, max_pred=GRID)

    ml = np.unravel_index(np.argmax(prob), prob.shape)
    ev_pick, ev_val = optimal_prediction(matrix, max_pred=GRID)

    home, away = row["home"], row["away"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    panels = [
        (axes[0], prob, "P(scoreline)  —  most likely", "Blues", ml, "%.1f%%", 100),
        (axes[1], ev, "E[points]  —  what we actually pick", "YlOrRd", tuple(ev_pick), "%.2f", 1),
    ]
    for ax, data, title, cmap, hi, fmt, scale in panels:
        im = ax.imshow(data, cmap=cmap, origin="upper")
        for i in range(GRID + 1):
            for j in range(GRID + 1):
                ax.text(j, i, fmt % (data[i, j] * scale), ha="center", va="center",
                        fontsize=7.5, color="#222")
        # highlight the argmax cell
        ax.add_patch(plt.Rectangle((hi[1] - 0.5, hi[0] - 0.5), 1, 1,
                                   fill=False, edgecolor="#111", lw=2.6))
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel(f"{away} goals")
        ax.set_ylabel(f"{home} goals")
        ax.set_xticks(range(GRID + 1))
        ax.set_yticks(range(GRID + 1))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ml_s = f"{ml[0]}-{ml[1]}"
    ev_s = f"{ev_pick[0]}-{ev_pick[1]}"
    fig.suptitle(
        f"{home} vs {away}   —   most likely = {ml_s},   EV-max pick = {ev_s} "
        f"(E[pts] = {ev_val:.2f})",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}  ({home} vs {away}: ML {ml_s} / EV {ev_s})")


if __name__ == "__main__":
    main()
