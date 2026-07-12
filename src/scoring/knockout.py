"""a.PSO (after penalty shoot-out) scoring for knockout matches.

In the WC2026 knockouts kicktipp scores the final result including extra time
and the penalty shoot-out (``a.PSO``): "if the score is 2:2 after extra time
and 5:4 in the penalty shoot-out, the score is 7:6." Two consequences for the
EV-max pick:

1. The result is never a draw. A draw scoreline therefore scores 0 always
   (its tendency can never match), so every rational pick is decisive.
2. The tendency of the result == who advances. kicktipp posts a 2-way
   *advance* price (no draw); its devigged ``q_home`` is exactly P(the result's
   tendency is a home win).

This module turns a 90-minute regulation score matrix (built the normal way
from the 3-way market odds - the score-engine invariant is preserved) into an
a.PSO final-result matrix, then reuses :func:`optimal_prediction`. The
factorisation cleanly separates the two markets:

* Tendency = who advances = kicktipp's 2-way price ``q_home`` (authoritative;
  it is the direct market for advancement). The final matrix's home-win mass is
  exactly ``q_home``.
* Conditional scoreline shape from the regulation matrix. Conditional on a
  side advancing, the scoreline is either a regulation win for that side or a
  former draw resolved its way - a draw ``(k,k)`` becomes ``(k+m, k)`` with the
  deciding margin ``m`` drawn from ``margin_kernel`` (ET / shoot-out deciders are
  mostly 1-2 goals). Each side's conditional shape is normalised, then mixed
  ``q_home·H + (1-q_home)·A``.

Modelling note (documented approximation): a shoot-out really records an
inflated score (e.g. 6:5), but for *kicktipp points* only the tendency and the
goal difference matter, plus the (tiny, near-unhittable) exact term. We place a
draw-resolved tie at ``(k+m, k)`` - correct tendency, correct GD ``m`` - which
slightly over-credits the *exact*-hittability of 1-2 goal margins on resolved
ties. The effect is small and nudges toward 1-goal-margin picks, which is
already where knockout EV-max lands. The tendency and GD tiers (which dominate
EV) are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.scoring.kicktipp import optimal_prediction

# Extra-time / penalty-shoot-out deciding margin. A reg draw is broken by an ET
# goal (margin ~1) or a shoot-out (margin ~1-2). Tunable; sums to 1.
DEFAULT_MARGIN_KERNEL: dict[int, float] = {1: 0.60, 2: 0.27, 3: 0.09, 4: 0.04}


@dataclass
class APSOInfo:
    """Diagnostics from the a.PSO transform."""
    q_home: float            # P(home advances) (kicktipp 2-way) - the tendency, honoured exactly
    reg_home_win: float      # P(home wins in regulation) from the matrix
    reg_draw: float          # P(draw in regulation) (resolved into the conditional shapes)
    reg_away_win: float      # P(away wins in regulation)
    realised_q_home: float   # actual P(home advances) in the final matrix (== q_home)


def _outcome_masses(reg_matrix: np.ndarray) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) in regulation from a score matrix."""
    g = reg_matrix.shape[0]
    home = draw = away = 0.0
    for i in range(g):
        for j in range(g):
            p = reg_matrix[i, j]
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
    return home, draw, away


def apso_result_matrix(
    reg_matrix: np.ndarray,
    q_home: float,
    *,
    margin_kernel: dict[int, float] | None = None,
) -> tuple[np.ndarray, APSOInfo]:
    """Transform a 90-minute regulation matrix into an a.PSO final-result matrix.

    Parameters
    ----------
    reg_matrix : (G+1)x(G+1) regulation score matrix (may carry draw mass).
    q_home : devigged P(home advances) from the kicktipp 2-way price.
    margin_kernel : {margin: prob} for the ET/PSO decider (default
        :data:`DEFAULT_MARGIN_KERNEL`).

    Returns
    -------
    (final_matrix, info) - ``final_matrix`` has no draw mass and its home-win
    (advance) mass equals ``q_home`` exactly.
    """
    kernel = margin_kernel or DEFAULT_MARGIN_KERNEL
    ksum = sum(kernel.values())
    kernel = {m: w / ksum for m, w in kernel.items()}    # normalise defensively

    g = reg_matrix.shape[0]
    home_win, draw_mass, away_win = _outcome_masses(reg_matrix)

    # Conditional scoreline shapes: H = scoreline | home advances, A = | away.
    # Each = that side's regulation wins + former draws resolved its way, with the
    # deciding margin from the kernel. Normalised, so the regulation odds drive the
    # *shape* only; the *tendency* comes entirely from q_home below.
    H = np.zeros_like(reg_matrix, dtype=float)
    A = np.zeros_like(reg_matrix, dtype=float)
    for i in range(g):
        for j in range(g):
            p = reg_matrix[i, j]
            if i > j:
                H[i, j] += p
            elif i < j:
                A[i, j] += p
    for k in range(g):                                   # resolve each reg draw (k,k)
        d = reg_matrix[k, k]
        if d <= 0:
            continue
        for m, w in kernel.items():
            hi = min(k + m, g - 1)                        # clamp margin into the grid
            if hi <= k:                                   # no room for a decisive score
                continue                                  #   (top row; mass ~0) - drop
            H[hi, k] += d * w                             # -> home win by m
            A[k, hi] += d * w                             # -> away win by m

    H_sum, A_sum = H.sum(), A.sum()
    if H_sum > 0:
        H /= H_sum
    if A_sum > 0:
        A /= A_sum

    final = q_home * H + (1.0 - q_home) * A               # tendency = q_home, exactly

    info = APSOInfo(
        q_home=q_home, reg_home_win=home_win, reg_draw=draw_mass,
        reg_away_win=away_win, realised_q_home=q_home,
    )
    return final, info


def apso_optimal_prediction(
    reg_matrix: np.ndarray,
    q_home: float,
    *,
    margin_kernel: dict[int, float] | None = None,
    max_pred: int | None = None,
) -> tuple[tuple[int, int], float, np.ndarray, APSOInfo]:
    """a.PSO EV-max pick: build the final-result matrix, then ``optimal_prediction``.

    Returns ``(pick, ev, final_matrix, info)``. ``pick`` is always decisive (a
    draw can never be the a.PSO result, so it can never be EV-max).
    """
    final, info = apso_result_matrix(reg_matrix, q_home, margin_kernel=margin_kernel)
    pick, ev = optimal_prediction(final, max_pred)
    return pick, ev, final, info
