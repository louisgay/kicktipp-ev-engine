"""Reconstruct a full P(i,j) score matrix from devigged odds.

Given devigged 1X2 probabilities (and optionally Over/Under),
find (λ_home, λ_away) such that a Dixon-Coles model reproduces
those probabilities as closely as possible. Then generate the
full score matrix.

This bridges bookmaker odds -> the P(i,j) matrix interface expected
by the existing scoring/decision layer (src/scoring/kicktipp.py).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import optimize
from scipy.stats import poisson

from src.models.dixon_coles import _dc_tau

logger = logging.getLogger(__name__)


def _score_matrix_from_lambdas(
    lam_h: float,
    lam_a: float,
    rho: float = 0.0,
    max_goals: int = 8,
) -> np.ndarray:
    """Build a Dixon-Coles score matrix from (λ_h, λ_a, ρ).

    Reuses the _dc_tau correction from the Phase 1 Dixon-Coles model.
    """
    goals = np.arange(max_goals + 1)
    prob_h = poisson.pmf(goals, max(lam_h, 1e-6))
    prob_a = poisson.pmf(goals, max(lam_a, 1e-6))

    matrix = np.outer(prob_h, prob_a)

    # Apply Dixon-Coles correction to (0,0), (1,0), (0,1), (1,1)
    if abs(rho) > 1e-10:
        matrix[0, 0] *= _dc_tau(0, 0, lam_h, lam_a, rho)
        matrix[1, 0] *= _dc_tau(1, 0, lam_h, lam_a, rho)
        matrix[0, 1] *= _dc_tau(0, 1, lam_h, lam_a, rho)
        matrix[1, 1] *= _dc_tau(1, 1, lam_h, lam_a, rho)
        matrix = np.clip(matrix, 0, None)

    matrix /= matrix.sum()
    return matrix


def _1x2_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract P(home_win), P(draw), P(away_win) from a score matrix."""
    p_home = np.tril(matrix, k=-1).sum()   # home_goals > away_goals
    p_draw = np.trace(matrix)               # home_goals == away_goals
    p_away = np.triu(matrix, k=1).sum()     # home_goals < away_goals
    return float(p_home), float(p_draw), float(p_away)


def _over_under_from_matrix(
    matrix: np.ndarray,
    line: float = 2.5,
) -> tuple[float, float]:
    """Extract P(over line), P(under line) from a score matrix."""
    max_g = matrix.shape[0] - 1
    p_under = 0.0
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            if (i + j) < line:
                p_under += matrix[i, j]
    p_over = 1.0 - p_under
    return float(p_over), float(p_under)


def reconstruct_lambdas(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over_2_5: float | None = None,
    rho: float = -0.04,
    max_goals: int = 8,
) -> tuple[float, float, float]:
    """Find (λ_h, λ_a) that best reproduce the devigged probabilities.

    Parametrisation: we optimise (supremacy, total) where
        λ_h = (total + supremacy) / 2
        λ_a = (total - supremacy) / 2

    This is more natural because 1X2 odds constrain supremacy directly
    and O/U constrains the total.

    Parameters
    ----------
    p_home, p_draw, p_away : devigged 1X2 probabilities
    p_over_2_5 : devigged P(over 2.5), optional
    rho : Dixon-Coles correction parameter (use model's fitted rho)
    max_goals : grid size

    Returns
    -------
    (lambda_home, lambda_away, fit_error)
    """

    def _objective(params: np.ndarray) -> float:
        sup, tot = params
        lam_h = (tot + sup) / 2.0
        lam_a = (tot - sup) / 2.0
        if lam_h <= 0.10 or lam_a <= 0.10:
            return 1e6

        matrix = _score_matrix_from_lambdas(lam_h, lam_a, rho, max_goals)
        model_h, model_d, model_a = _1x2_from_matrix(matrix)

        # Squared error on 1X2
        err = (model_h - p_home)**2 + (model_d - p_draw)**2 + (model_a - p_away)**2

        # Add O/U constraint if available
        if p_over_2_5 is not None:
            model_over, _ = _over_under_from_matrix(matrix, 2.5)
            err += (model_over - p_over_2_5)**2

        return err

    # Initial guess: supremacy from home-away prob diff, total from typical match
    sup_init = (p_home - p_away) * 1.5  # rough scaling
    tot_init = 2.5

    result = optimize.minimize(
        _objective,
        x0=[sup_init, tot_init],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000},
    )

    sup, tot = result.x
    lam_h = (tot + sup) / 2.0
    lam_a = (tot - sup) / 2.0

    # Ensure positive
    lam_h = max(lam_h, 0.15)
    lam_a = max(lam_a, 0.15)

    logger.debug("Reconstructed: λ_h=%.3f, λ_a=%.3f (sup=%.3f, tot=%.3f), "
                 "fit_err=%.6f", lam_h, lam_a, sup, tot, result.fun)

    return lam_h, lam_a, float(result.fun)


def reconstruct_matrix(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over_2_5: float | None = None,
    rho: float = -0.04,
    max_goals: int = 8,
) -> np.ndarray:
    """Reconstruct a full P(i,j) score matrix from devigged probabilities.

    This is the main entry point: odds -> matrix -> optimal_prediction().

    Parameters
    ----------
    p_home, p_draw, p_away : devigged 1X2 probabilities (must sum to ~1)
    p_over_2_5 : devigged P(over 2.5), optional
    rho : Dixon-Coles correction parameter
    max_goals : grid size

    Returns
    -------
    (max_goals+1) x (max_goals+1) probability matrix, compatible with
    src/scoring/kicktipp.optimal_prediction().
    """
    lam_h, lam_a, err = reconstruct_lambdas(
        p_home, p_draw, p_away, p_over_2_5, rho, max_goals
    )

    if err > 0.01:
        logger.warning("Reconstruction fit error is high: %.4f "
                       "(probs may not match well)", err)

    matrix = _score_matrix_from_lambdas(lam_h, lam_a, rho, max_goals)
    return matrix


# -- Batch processing of odds data ------------------------------------


def extract_odds_for_event(
    event: dict,
    preferred_bookmakers: list[str] | None = None,
) -> dict | None:
    """Extract 1X2 and O/U odds from an Odds API event dict.

    Prefers Pinnacle, falls back to consensus (average across books).

    Parameters
    ----------
    event : event dict from The Odds API response
    preferred_bookmakers : ordered list of preferred bookmakers
        (first found wins). Default: ['pinnacle', 'betfair_ex_eu'].

    Returns
    -------
    Dict with keys: home_team, away_team, commence_time,
    h2h_odds (list[3]), totals_line, totals_odds (list[2]),
    bookmaker_used, or None if insufficient data.
    """
    if preferred_bookmakers is None:
        preferred_bookmakers = ["pinnacle", "betfair_ex_eu", "sport888",
                                "unibet_eu", "betclic"]

    home = event.get("home_team", "")
    away = event.get("away_team", "")
    commence = event.get("commence_time", "")

    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    # Find 1X2 odds
    h2h_odds = None
    h2h_book = None
    for book_name in preferred_bookmakers:
        for bm in bookmakers:
            if bm.get("key", "") == book_name:
                for market in bm.get("markets", []):
                    if market.get("key") == "h2h":
                        outcomes = market.get("outcomes", [])
                        if len(outcomes) == 3:
                            # Order: Home, Draw, Away
                            h2h = {}
                            for o in outcomes:
                                h2h[o["name"]] = o["price"]
                            if home in h2h and "Draw" in h2h and away in h2h:
                                h2h_odds = [h2h[home], h2h["Draw"], h2h[away]]
                                h2h_book = book_name
                                break
            if h2h_odds:
                break
        if h2h_odds:
            break

    # Fallback: average across all bookmakers
    if h2h_odds is None:
        h2h_all = []
        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) == 3:
                        h2h = {}
                        for o in outcomes:
                            h2h[o["name"]] = o["price"]
                        if home in h2h and "Draw" in h2h and away in h2h:
                            h2h_all.append([h2h[home], h2h["Draw"], h2h[away]])
        if h2h_all:
            h2h_odds = list(np.mean(h2h_all, axis=0))
            h2h_book = "consensus"

    if h2h_odds is None:
        logger.warning("No h2h odds for %s vs %s", home, away)
        return None

    # Find O/U 2.5 odds. Over and Under MUST come from the SAME (bookmaker,
    # market, point) - over_price/under_price are reset per market so a stale
    # Over from one book can never pair with an Under from another. If a book
    # quotes only one side at 2.5, that book is skipped (line treated as absent).
    def _ou_25(market: dict) -> list[float] | None:
        if market.get("key") != "totals":
            return None
        over_price = under_price = None
        for o in market.get("outcomes", []):
            if o.get("point") == 2.5:
                if o.get("name") == "Over":
                    over_price = o.get("price")
                elif o.get("name") == "Under":
                    under_price = o.get("price")
        if over_price is not None and under_price is not None:
            return [over_price, under_price]
        return None

    totals_odds = None
    totals_line = None
    for book_name in preferred_bookmakers:
        for bm in bookmakers:
            if bm.get("key", "") != book_name:
                continue
            for market in bm.get("markets", []):
                pair = _ou_25(market)
                if pair is not None:
                    totals_odds = pair
                    totals_line = 2.5
                    break
            if totals_odds:
                break
        if totals_odds:
            break

    # Fallback: consensus = average across books that quote BOTH sides at 2.5.
    if totals_odds is None:
        over_all, under_all = [], []
        for bm in bookmakers:
            for market in bm.get("markets", []):
                pair = _ou_25(market)
                if pair is not None:
                    over_all.append(pair[0])
                    under_all.append(pair[1])
        if over_all and under_all:
            totals_odds = [float(np.mean(over_all)), float(np.mean(under_all))]
            totals_line = 2.5

    return {
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "h2h_odds": h2h_odds,
        "totals_line": totals_line,
        "totals_odds": totals_odds,
        "bookmaker_used": h2h_book,
    }
