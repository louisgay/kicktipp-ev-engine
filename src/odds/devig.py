"""Devigging (margin removal) for bookmaker odds.

Bookmakers set odds so that the implied probabilities sum to more than 1
(the "overround" or "vig"). We need to remove this margin to recover
fair probabilities.

Two methods implemented:
1. Normalisation (additive): divide each implied prob by the total.
   Simple but biased - underestimates favourites, overestimates longshots.

2. Shin's method: assumes the overround comes from informed trading
   (adverse selection). Corrects for the favourite-longshot bias.
   Reference: Shin (1993), "Measuring the Incidence of Insider Trading
   in a Market for State-Contingent Claims".
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def implied_probabilities(odds: list[float]) -> list[float]:
    """Convert decimal odds to implied probabilities (with vig).

    Parameters
    ----------
    odds : list of decimal odds (e.g. [2.10, 3.40, 3.50] for 1X2)

    Returns
    -------
    List of implied probabilities (sum > 1.0 due to vig).
    """
    return [1.0 / o for o in odds]


def overround(odds: list[float]) -> float:
    """Compute the overround (total implied probability minus 1).

    Example: odds [2.10, 3.40, 3.50] -> implied sum ≈ 1.057 -> overround ≈ 0.057 (5.7%)
    """
    return sum(implied_probabilities(odds)) - 1.0


def devig_normalise(odds: list[float]) -> list[float]:
    """Remove vig by simple normalisation (additive method).

    Each implied probability is divided by the sum of all implied probs.
    Simple but biased: treats all outcomes equally.

    Parameters
    ----------
    odds : list of decimal odds

    Returns
    -------
    List of fair probabilities summing to 1.0.
    """
    implied = implied_probabilities(odds)
    total = sum(implied)
    if total <= 0:
        raise ValueError(f"Invalid odds: implied sum = {total}")
    fair = [p / total for p in implied]
    return fair


def devig_shin(odds: list[float], max_iter: int = 100, tol: float = 1e-12) -> list[float]:
    """Remove vig using Shin's method (true fixed point).

    Shin's model assumes a proportion z of bettors are "insiders". The fair
    probabilities are

        p_i(z) = (sqrt(z^2 + 4*(1-z)*q_i^2 / Q) - z) / (2*(1-z))

    where q_i = 1/odds_i are the implied probabilities and Q = sum(q_i) is the
    book sum. The correct z is the one for which sum_i p_i(z) = 1. Since
    sum_i p_i(z) decreases monotonically from sqrt(Q) > 1 (at z=0) toward
    sum(q_i^2)/Q < 1 (as z -> 1), we solve for it by bisection - this enforces
    the unit-sum constraint exactly, unlike the first-order approximation
    z = (Q-1)/(n-1).

    Reference: Shin (1993), "Measuring the Incidence of Insider Trading...";
    Jacobs/Keith, "Efficiently Computing Shin Probabilities."
    """
    implied = np.array(implied_probabilities(odds))
    n = len(implied)
    if n < 2:
        return [1.0]

    Q = implied.sum()

    def fair_for_z(z: float) -> np.ndarray:
        disc = z * z + 4.0 * (1.0 - z) * implied ** 2 / Q
        return (np.sqrt(disc) - z) / (2.0 * (1.0 - z))

    # Bisection on z in [0, 1): sum(fair_for_z(z)) is monotone decreasing.
    lo, hi, z = 0.0, 1.0 - 1e-9, 0.0
    for _ in range(max_iter):
        z = 0.5 * (lo + hi)
        s = float(fair_for_z(z).sum())
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:          # sum too high -> need larger z
            lo = z
        else:
            hi = z

    fair = fair_for_z(z)
    fair = fair / fair.sum()  # tidy any residual rounding

    logger.debug("Shin devig: overround=%.4f, z=%.4f, fair_sum=%.6f",
                 Q - 1, z, fair.sum())
    return fair.tolist()


def devig(odds: list[float], method: str = "normalise") -> list[float]:
    """Convenience wrapper: remove vig using specified method.

    Parameters
    ----------
    odds : list of decimal odds
    method : 'normalise' or 'shin'

    Returns
    -------
    List of fair probabilities summing to 1.0.
    """
    if method == "normalise":
        return devig_normalise(odds)
    elif method == "shin":
        return devig_shin(odds)
    else:
        raise ValueError(f"Unknown devig method: {method!r}. Use 'normalise' or 'shin'.")


# -- Helpers for specific markets --------------------------------------


def devig_1x2(home_odds: float, draw_odds: float, away_odds: float,
              method: str = "normalise") -> tuple[float, float, float]:
    """Devig 1X2 (3-way) odds.

    Returns (p_home_win, p_draw, p_away_win).
    """
    probs = devig([home_odds, draw_odds, away_odds], method=method)
    return probs[0], probs[1], probs[2]


def devig_over_under(over_odds: float, under_odds: float,
                     method: str = "normalise") -> tuple[float, float]:
    """Devig Over/Under odds.

    Returns (p_over, p_under).
    """
    probs = devig([over_odds, under_odds], method=method)
    return probs[0], probs[1]
