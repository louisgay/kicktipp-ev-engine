"""Monte-Carlo group stage simulation.

Derives P(win group), P(finish 1st/2nd/3rd) for each team from
match-level λ values.

λ sources (in priority order):
1. Live kicktipp 1X2 odds -> reconstruct_lambdas()
2. Cached odds (JSON with timestamp, staleness alert after 24h)
3. Hardcoded fixture (test-only fallback)

Each group match is simulated N times using Poisson draws with
Dixon-Coles correction. Group standings are computed per simulation
(3 pts for win, 1 for draw; tiebreaker: GD, GF, random).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from src.models.dixon_coles import _dc_tau

logger = logging.getLogger(__name__)

# -- pool group compositions ------------------------------------
# Scraped from the kicktipp bonus page (group winner questions).
# These differ from the official FIFA WC 2026 draw.

POOL_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["Australia", "Paraguay", "Turkey", "United States"],
    "E": ["Curaçao", "Germany", "Ecuador", "Ivory Coast"],
    "F": ["Japan", "Netherlands", "Sweden", "Tunisia"],
    "G": ["Egypt", "Belgium", "Iran", "New Zealand"],
    "H": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I": ["France", "Iraq", "Norway", "Senegal"],
    "J": ["Algeria", "Argentina", "Jordan", "Austria"],
    "K": ["DR Congo", "Colombia", "Portugal", "Uzbekistan"],
    "L": ["England", "Ghana", "Croatia", "Panama"],
}

# Backward-compatible alias
WC2026_GROUPS = POOL_GROUPS

# All 48 teams (for bracket simulation)
ALL_TEAMS: set[str] = set()
for _teams in POOL_GROUPS.values():
    ALL_TEAMS.update(_teams)

# Reverse lookup: team -> group letter
TEAM_TO_GROUP: dict[str, str] = {}
for _g, _teams in POOL_GROUPS.items():
    for _t in _teams:
        TEAM_TO_GROUP[_t] = _g

# -- Cache management ------------------------------------------------

_CACHE_DIR = Path.home() / ".kicktipp_cache"
_CACHE_FILE = _CACHE_DIR / "group_matches.json"
_CACHE_MAX_AGE_HOURS = 24


def _cache_is_fresh() -> bool:
    """Check if the cached group match data exists and is fresh (<24h)."""
    if not _CACHE_FILE.exists():
        return False
    try:
        data = json.loads(_CACHE_FILE.read_text())
        ts = data.get("timestamp", 0)
        age_hours = (time.time() - ts) / 3600
        return age_hours < _CACHE_MAX_AGE_HOURS
    except (json.JSONDecodeError, KeyError):
        return False


def _read_cache() -> list[tuple[str, str, float, float, float]] | None:
    """Read cached group match probabilities."""
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text())
        ts = data.get("timestamp", 0)
        age_hours = (time.time() - ts) / 3600
        matches = [tuple(m) for m in data["matches"]]
        if age_hours > _CACHE_MAX_AGE_HOURS:
            logger.warning(
                "Group match cache is %.1f hours old (max %d). "
                "Re-scrape recommended: rebuild with build_group_lambdas(live=True)",
                age_hours, _CACHE_MAX_AGE_HOURS,
            )
        else:
            logger.info("Using cached group match odds (%.1fh old, %d matches)",
                        age_hours, len(matches))
        return matches
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Cache read error: %s", e)
        return None


def _write_cache(
    matches: list[tuple[str, str, float, float, float]],
) -> None:
    """Write group match probabilities to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": time.time(),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "matches": matches,
    }
    _CACHE_FILE.write_text(json.dumps(data, indent=2))
    logger.info("Cached %d group match odds to %s", len(matches), _CACHE_FILE)


# -- Live scraping -> group match extraction --------------------------


def _scrape_group_matches() -> list[tuple[str, str, float, float, float]]:
    """Scrape all 10 group-stage matchdays and extract 1X2 probabilities.

    Filters scraped matches to only those involving teams in
    POOL_GROUPS (group-stage matches only).

    Returns
    -------
    List of (home, away, P(H), P(D), P(A)) tuples.
    """
    from src.data.kicktipp_scrape import scrape_all_matchdays

    logger.info("Scraping all group-stage matchdays from kicktipp...")
    all_matches = scrape_all_matchdays(n_matchdays=10)

    from src.data.clean import normalise_team

    group_matches = []
    for m in all_matches:
        # Canonicalise names so kicktipp spellings (USA, Türkiye,
        # Bosnien-Herzegowina) match the POOL_GROUPS keys.
        home = normalise_team(m.home_team)
        away = normalise_team(m.away_team)
        # Filter: both teams must be in pool groups
        if home in ALL_TEAMS and away in ALL_TEAMS:
            group_matches.append((home, away, m.prob_home, m.prob_draw, m.prob_away))

    logger.info("Extracted %d group-stage matches from %d total scraped",
                len(group_matches), len(all_matches))

    if len(group_matches) < 72:
        logger.warning(
            "Expected 72 group matches, got %d. "
            "Some matchdays may not have odds yet.",
            len(group_matches),
        )

    return group_matches


# -- Simulation -------------------------------------------------------


def _simulate_match(
    lam_h: float,
    lam_a: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Draw a single match result from Poisson(λ_h) × Poisson(λ_a) with DC correction.

    Uses rejection sampling to apply the Dixon-Coles low-score correction.
    """
    for _ in range(100):  # rejection loop, almost always accepts first try
        gh = rng.poisson(lam_h)
        ga = rng.poisson(lam_a)
        # DC correction only affects (0,0), (1,0), (0,1), (1,1)
        if gh <= 1 and ga <= 1 and abs(rho) > 1e-10:
            tau = _dc_tau(gh, ga, lam_h, lam_a, rho)
            if rng.random() > tau:
                continue  # reject
        return int(gh), int(ga)
    # Fallback: accept uncorrected
    return int(rng.poisson(lam_h)), int(rng.poisson(lam_a))


def build_group_lambdas(
    matches: list[tuple[str, str, float, float, float]] | None = None,
    *,
    live: bool = False,
    rho: float = -0.04,
    max_goals: int = 8,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Reconstruct λ values for group matches from kicktipp odds.

    Data source priority:
    1. ``matches`` parameter (if provided explicitly)
    2. ``live=True`` -> scrape kicktipp now, update cache
    3. Cached odds from ~/.kicktipp_cache/group_matches.json
    4. Raises RuntimeError (no hardcoded fallback in production)

    Parameters
    ----------
    matches : pre-scraped match odds, or None to auto-resolve
    live : if True, force re-scrape from kicktipp (requires KICKTIPP_SESSION)
    rho : Dixon-Coles correction parameter
    max_goals : grid size for λ reconstruction

    Returns
    -------
    {(home, away) -> (λ_home, λ_away)}
    """
    from src.odds.reconstruct import reconstruct_lambdas

    if matches is None:
        if live:
            matches = _scrape_group_matches()
            _write_cache(matches)
        else:
            matches = _read_cache()
            if matches is None:
                raise RuntimeError(
                    "No group match data available. Either:\n"
                    "  1. Run with live=True to scrape from kicktipp\n"
                    "  2. Provide matches explicitly\n"
                    "  3. Populate cache: python -c "
                    "'from src.bonus.group_sim import build_group_lambdas; "
                    "build_group_lambdas(live=True)'"
                )

    lambdas: dict[tuple[str, str], tuple[float, float]] = {}
    for home, away, p_h, p_d, p_a in matches:
        lam_h, lam_a, err = reconstruct_lambdas(
            p_h, p_d, p_a, p_over_2_5=None, rho=rho, max_goals=max_goals,
        )
        lambdas[(home, away)] = (lam_h, lam_a)
        if err > 0.01:
            logger.debug("High fit error for %s vs %s: %.4f", home, away, err)

    logger.info("Reconstructed λ for %d group matches", len(lambdas))
    return lambdas


def simulate_group(
    group: str,
    match_lambdas: dict[tuple[str, str], tuple[float, float]],
    rho: float = -0.04,
    n_sims: int = 100_000,
    rng: np.random.Generator | None = None,
) -> list[list[str]]:
    """Simulate a group and return full standings per simulation.

    Parameters
    ----------
    group : group letter (A-L)
    match_lambdas : {(home, away) -> (λ_home, λ_away)} for group matches
    rho : Dixon-Coles correction parameter
    n_sims : number of simulations
    rng : random number generator (shared across tournament sim)

    Returns
    -------
    List of n_sims rankings, each a list of 4 team names sorted 1st->4th.
    """
    teams = POOL_GROUPS[group]
    assert len(teams) == 4, f"Group {group} has {len(teams)} teams, expected 4"

    matches = list(combinations(teams, 2))

    # Resolve λ values
    resolved: list[tuple[str, str, float, float]] = []
    for t1, t2 in matches:
        if (t1, t2) in match_lambdas:
            lh, la = match_lambdas[(t1, t2)]
            resolved.append((t1, t2, lh, la))
        elif (t2, t1) in match_lambdas:
            lh, la = match_lambdas[(t2, t1)]
            resolved.append((t2, t1, lh, la))
        else:
            raise ValueError(
                f"No λ values for {t1} vs {t2} (or reverse). "
                f"Available: {list(match_lambdas.keys())}"
            )

    if rng is None:
        rng = np.random.default_rng(42)

    rankings: list[list[str]] = []
    for _ in range(n_sims):
        pts = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}

        for home, away, lam_h, lam_a in resolved:
            gh, g_a = _simulate_match(lam_h, lam_a, rho, rng)
            gf[home] += gh
            ga[home] += g_a
            gf[away] += g_a
            ga[away] += gh

            if gh > g_a:
                pts[home] += 3
            elif gh == g_a:
                pts[home] += 1
                pts[away] += 1
            else:
                pts[away] += 3

        ranking = sorted(
            teams,
            key=lambda t: (pts[t], gf[t] - ga[t], gf[t], rng.random()),
            reverse=True,
        )
        rankings.append(ranking)

    return rankings


def simulate_group_winner(
    group: str,
    match_lambdas: dict[tuple[str, str], tuple[float, float]],
    rho: float = -0.04,
    n_sims: int = 100_000,
    seed: int = 42,
) -> dict[str, float]:
    """Simulate a group and return P(win group) for each team.

    Backward-compatible wrapper around simulate_group().
    """
    rng = np.random.default_rng(seed)
    rankings = simulate_group(group, match_lambdas, rho, n_sims, rng)

    teams = POOL_GROUPS[group]
    win_counts = {t: 0 for t in teams}
    for ranking in rankings:
        win_counts[ranking[0]] += 1

    return {t: win_counts[t] / n_sims for t in teams}


def simulate_all_groups(
    match_lambdas: dict[tuple[str, str], tuple[float, float]] | None = None,
    rho: float = -0.04,
    n_sims: int = 100_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Simulate all 12 groups and return P(win group) per team.

    Parameters
    ----------
    match_lambdas : {(home, away) -> (λ_home, λ_away)} for all group matches.
        If None, builds from kicktipp group match odds (cache or live).
    """
    if match_lambdas is None:
        match_lambdas = build_group_lambdas(rho=rho)

    results = {}
    for group in sorted(POOL_GROUPS.keys()):
        try:
            probs = simulate_group_winner(
                group, match_lambdas, rho, n_sims, seed,
            )
            results[group] = probs
            leader = max(probs, key=probs.get)
            logger.info("Group %s: %s (%.1f%%)", group, leader, probs[leader] * 100)
        except ValueError as e:
            logger.warning("Skipping Group %s: %s", group, e)
    return results
