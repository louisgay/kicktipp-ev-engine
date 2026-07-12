"""Kicktipp pool scoring rules and optimal prediction.

This is the core decision layer. Given a probability matrix P(i, j)
from any model, it selects the prediction (ph, pa) that maximises
expected pool points.

Scoring rules (pool)
--------------------------
- Exact score:       4 pts
- Correct goal diff: 3 pts  (same winner + same goal difference, but different score)
- Correct tendency:  2 pts  (same winner, wrong goal difference)
- Wrong:             0 pts

Note: for draws, there is no "goal diff" tier since all draws have
diff=0. So a draw prediction is either exact (4) or tendency (2).
"""

from __future__ import annotations

import numpy as np


def points(pred: tuple[int, int], actual: tuple[int, int]) -> int:
    """Compute pool points for a single prediction vs actual result.

    Parameters
    ----------
    pred : (predicted_home, predicted_away)
    actual : (actual_home, actual_away)

    Returns
    -------
    Points: 4 (exact), 3 (goal diff), 2 (tendency), or 0 (wrong).
    """
    ph, pa = pred
    ah, aa = actual

    # Exact score
    if (ph, pa) == (ah, aa):
        return 4

    # Check tendency (same outcome: home win / draw / away win)
    pred_sign = (ph > pa) - (ph < pa)   # +1, 0, -1
    actual_sign = (ah > aa) - (ah < aa)
    if pred_sign != actual_sign:
        return 0

    # Correct tendency - check goal difference (only for non-draws)
    if pred_sign != 0 and (ph - pa) == (ah - aa):
        return 3

    return 2


def expected_points(pred: tuple[int, int], prob_matrix: np.ndarray) -> float:
    """Compute expected pool points for prediction *pred*
    given a joint probability matrix P(i, j).

    Parameters
    ----------
    pred : (ph, pa) - the candidate prediction
    prob_matrix : 2D array of shape (G+1, G+1) where G = max_goals.
                  prob_matrix[i, j] = P(home scores i, away scores j).

    Returns
    -------
    E[points] = sum over (i,j) of P(i,j) * points(pred, (i,j))
    """
    max_goals = prob_matrix.shape[0] - 1
    ev = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = prob_matrix[i, j]
            if p > 0:
                ev += p * points(pred, (i, j))
    return ev


def optimal_prediction(prob_matrix: np.ndarray,
                       max_pred: int | None = None) -> tuple[tuple[int, int], float]:
    """Find the prediction that maximises expected pool points.

    Parameters
    ----------
    prob_matrix : (G+1) x (G+1) probability matrix
    max_pred : maximum score to consider in predictions (default: same as matrix)

    Returns
    -------
    (best_pred, best_ev) where best_pred = (home_goals, away_goals)
    """
    if max_pred is None:
        max_pred = prob_matrix.shape[0] - 1

    # Precompute: for efficiency, we compute EV for all candidates at once
    # using vectorised operations on the probability matrix.
    best_ev = -1.0
    best_pred = (0, 0)

    for ph in range(max_pred + 1):
        for pa in range(max_pred + 1):
            ev = expected_points((ph, pa), prob_matrix)
            if ev > best_ev:
                best_ev = ev
                best_pred = (ph, pa)

    return best_pred, best_ev


def ev_table(prob_matrix: np.ndarray,
             max_pred: int = 5) -> np.ndarray:
    """Compute a table of expected points for all predictions up to max_pred.

    Returns an (max_pred+1) x (max_pred+1) array where entry [i, j]
    is E[points] for predicting (i, j).
    """
    table = np.zeros((max_pred + 1, max_pred + 1))
    for ph in range(max_pred + 1):
        for pa in range(max_pred + 1):
            table[ph, pa] = expected_points((ph, pa), prob_matrix)
    return table
