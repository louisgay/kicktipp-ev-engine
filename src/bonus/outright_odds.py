"""Outright odds and tournament probabilities for FIFA World Cup 2026.

Probability sources (in priority order):
1. MC bracket simulation (derived from kicktipp match-level odds)
   -> get_winner_probs(), get_semi_finalist_probs()
   These are PROVISIONAL until the bracket structure is confirmed.
2. ESPN/BetMGM curated odds (for golden boot, devig examples)
   -> GOLDEN_BOOT_TEAM_PROBS, _WINNER_ODDS_AMERICAN

The ESPN outright winner odds are kept for reference and golden boot
derivation, but NOT used for the 'winner' bonus question - the bracket
simulation produces probabilities consistent with the pool
custom draw.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _american_to_implied(american: int | float) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def _american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def _normalise_probs(odds_map: dict[str, int | float]) -> dict[str, float]:
    """Convert American odds dict to normalised probabilities (additive)."""
    implied = {team: _american_to_implied(o) for team, o in odds_map.items()}
    total = sum(implied.values())
    return {team: p / total for team, p in implied.items()}


def _shin_probs(odds_map: dict[str, int | float]) -> dict[str, float]:
    """Convert American odds dict to Shin-devigged probabilities.

    Uses Shin's method which corrects for favourite-longshot bias.
    Better than additive normalization for high-overround markets
    (e.g., group winners at ~43% overround).
    """
    from src.odds.devig import devig_shin

    teams = list(odds_map.keys())
    decimal_odds = [_american_to_decimal(odds_map[t]) for t in teams]
    fair_probs = devig_shin(decimal_odds)
    return {t: p for t, p in zip(teams, fair_probs)}


# -- World Cup Winner -------------------------------------------------
# Source: ESPN/BetMGM, 2 June 2026

_WINNER_ODDS_AMERICAN: dict[str, int] = {
    "Spain": 475,
    "France": 500,
    "England": 650,
    "Brazil": 850,
    "Argentina": 900,
    "Portugal": 1000,
    "Germany": 1400,
    "Netherlands": 2200,
    "Belgium": 3500,
    "Norway": 3500,
    "Colombia": 4000,
    "Uruguay": 5000,
    "Morocco": 5000,
    "United States": 6000,
    "Switzerland": 6500,
    "Japan": 6500,
    "Mexico": 8000,
    "Croatia": 8000,
    "Ecuador": 8000,
    "Senegal": 9000,
    "Turkey": 10000,
    "Sweden": 10000,
    "Austria": 15000,
    "Canada": 20000,
    "Scotland": 20000,
    "Ivory Coast": 25000,
    "Czech Republic": 25000,
    "Paraguay": 30000,
    "Egypt": 30000,
    "South Korea": 40000,
    "Bosnia and Herzegovina": 50000,
    "Australia": 60000,
    "Iran": 70000,
    "DR Congo": 100000,
    "Saudi Arabia": 100000,
    "South Africa": 100000,
    "Qatar": 150000,
    "Uzbekistan": 150000,
    "New Zealand": 150000,
    "Iraq": 150000,
    "Jordan": 250000,
    "Haiti": 250000,
    # Teams without listed odds get a long-shot default
    "China": 200000,
    "Indonesia": 200000,
    "Slovenia": 100000,
    "Hungary": 80000,
    "Ukraine": 80000,
    "Denmark": 15000,
    "Serbia": 25000,
    "Mali": 100000,
    "Cameroon": 80000,
    "Nigeria": 60000,
}

# ESPN odds -> Shin probs (used for golden boot derivation and devig examples,
# NOT for the winner bonus question - bracket sim is used instead)
_WINNER_PROBS_ESPN: dict[str, float] = _shin_probs(_WINNER_ODDS_AMERICAN)

# Legacy alias - replaced by lazy-loaded bracket simulation below
WINNER_PROBS: dict[str, float] = _WINNER_PROBS_ESPN


# -- Golden Boot (top scorer's TEAM) ----------------------------------
# Derived from player-level Golden Boot odds.
# P(team) ≈ sum of P(player_i from team), since events are exclusive.
# Source: NY Sports Day / FOX Sports, 2 June 2026

_GOLDEN_BOOT_PLAYER_ODDS: list[tuple[str, str, int]] = [
    # (player, team, american_odds)
    ("Mbappé", "France", 600),
    ("Kane", "England", 700),
    ("Haaland", "Norway", 1400),
    ("Messi", "Argentina", 1600),
    ("Yamal", "Spain", 1800),
    ("Oyarzabal", "Spain", 1800),
    ("Ronaldo", "Portugal", 2000),
    ("Vinícius Jr", "Brazil", 2200),
    ("Lautaro Martínez", "Argentina", 2500),
    ("Dembélé", "France", 2800),
    ("Julián Álvarez", "Argentina", 3500),
    ("Lukaku", "Belgium", 5000),
    ("Salah", "Egypt", 5000),
    ("Son", "South Korea", 6000),
    ("Isak", "Sweden", 5000),
    ("Morata", "Spain", 6000),
    ("Saka", "England", 4000),
    ("Palmer", "England", 5000),
    ("Raphinha", "Brazil", 5000),
    ("Endrick", "Brazil", 6000),
    ("Müller/Havertz", "Germany", 4000),
    ("Gakpo", "Netherlands", 5000),
    ("Luis Díaz", "Colombia", 6000),
    ("Osimhen", "Nigeria", 6000),
]

# Remove non-qualified (Lewandowski/Poland)
_GOLDEN_BOOT_PLAYER_ODDS = [
    (p, t, o) for p, t, o in _GOLDEN_BOOT_PLAYER_ODDS
    if t != "Poland"
]


def _compute_golden_boot_team_probs() -> dict[str, float]:
    """Aggregate player-level golden boot odds to team-level probs.

    Player-level odds are first Shin-devigged (they have significant
    overround), then aggregated to team level by summing each team's
    player probabilities.
    """
    from src.odds.devig import devig_shin

    # Shin-devig at player level first
    players = [(p, t, o) for p, t, o in _GOLDEN_BOOT_PLAYER_ODDS]
    decimal_odds = [_american_to_decimal(o) for _, _, o in players]
    fair_probs = devig_shin(decimal_odds)

    # Aggregate to team level
    team_probs: dict[str, float] = {}
    for (_, team, _), p in zip(players, fair_probs):
        team_probs[team] = team_probs.get(team, 0.0) + p

    # Re-normalise (should be close to 1 already since we're aggregating
    # within-team, but some teams may not have all players listed)
    total = sum(team_probs.values())
    return {t: p / total for t, p in team_probs.items()}


GOLDEN_BOOT_TEAM_PROBS: dict[str, float] = _compute_golden_boot_team_probs()


# -- Group Winners (A-L) ---------------------------------------------
# Derived via Monte-Carlo simulation from kicktipp match-level odds.
# NOT from scraped outright odds (which may not match pool's draw).


def _compute_group_winner_probs() -> dict[str, dict[str, float]]:
    """Compute group winner probabilities via MC simulation.

    Uses all 72 group-stage match odds from kicktipp, reconstructed
    into Poisson λ values, then simulated 100k times per group.
    """
    from src.bonus.group_sim import simulate_all_groups
    return simulate_all_groups(n_sims=100_000)


# Lazy-loaded to avoid circular import at module level
_GROUP_WINNER_PROBS_CACHE: dict[str, dict[str, float]] | None = None


def get_group_winner_probs() -> dict[str, dict[str, float]]:
    """Get group winner probabilities (computed on first call, then cached)."""
    global _GROUP_WINNER_PROBS_CACHE
    if _GROUP_WINNER_PROBS_CACHE is None:
        _GROUP_WINNER_PROBS_CACHE = _compute_group_winner_probs()
    return _GROUP_WINNER_PROBS_CACHE


# For backward compatibility - computed lazily
GROUP_WINNER_PROBS: dict[str, dict[str, float]] = {}


# -- Winner & Semi-finalist probs from bracket-agnostic simulation ---
# Derived from full tournament MC simulation with random knockout pairing.
# This marginalises over all possible bracket structures - necessary
# because the pool bracket is unpublished ("unknown vs unknown").
#
# AWAITING BRACKET - once the bracket is published, switch to
# deterministic pairings for more precise probabilities.
#
# P(reach SF) sums to ~4 (4 semi-final slots), not 1.
# P(win tournament) sums to ~1.

_TOURNAMENT_PROBS_CACHE: dict[str, dict[str, float]] | None = None


def _get_tournament_probs() -> dict[str, dict[str, float]]:
    """Lazy-load tournament probabilities from bracket simulation."""
    global _TOURNAMENT_PROBS_CACHE
    if _TOURNAMENT_PROBS_CACHE is None:
        from src.bonus.bracket_sim import simulate_tournament
        _TOURNAMENT_PROBS_CACHE = simulate_tournament(n_sims=50_000)
    return _TOURNAMENT_PROBS_CACHE


def get_winner_probs() -> dict[str, float]:
    """Get P(win tournament) from bracket-agnostic simulation.

    AWAITING BRACKET - these are bracket-averaged estimates
    (random knockout pairing). Less precise but not wrong.
    """
    return _get_tournament_probs()["winner"]


def get_semi_finalist_probs() -> dict[str, float]:
    """Get P(reach semi-final) from bracket-agnostic simulation.

    Returns probabilities that sum to ~4 (4 SF slots).
    Each value is the independent probability of reaching SF.
    AWAITING BRACKET - bracket-averaged estimates.
    """
    return _get_tournament_probs()["semi_finalist"]


# -- Devigging utilities -----------------------------------------------


def devig_outright(
    odds_map: dict[str, int | float],
    method: str = "shin",
) -> tuple[dict[str, float], float]:
    """Devig an outright odds map.

    Parameters
    ----------
    odds_map : {team -> American odds}
    method : 'shin' (default) or 'normalise'

    Returns (devigged_probs, overround) where overround is the sum
    of raw implied probabilities (>1 indicates bookmaker margin).
    """
    implied = {team: _american_to_implied(o) for team, o in odds_map.items()}
    overround = sum(implied.values())

    if method == "shin":
        devigged = _shin_probs(odds_map)
    else:
        devigged = {team: p / overround for team, p in implied.items()}

    return devigged, overround


def format_devig_example(
    odds_map: dict[str, int | float] | None = None,
    top_n: int = 8,
) -> str:
    """Format a transparent devigging example for display.

    Shows raw American odds, implied probabilities, overround,
    devigged probabilities (additive and Shin), and the argmax pick.
    """
    if odds_map is None:
        odds_map = _WINNER_ODDS_AMERICAN

    devigged_shin, overround = devig_outright(odds_map, method="shin")
    devigged_add, _ = devig_outright(odds_map, method="normalise")
    implied = {team: _american_to_implied(o) for team, o in odds_map.items()}

    # Sort by implied probability (highest first)
    sorted_teams = sorted(implied, key=implied.get, reverse=True)
    top_teams = sorted_teams[:top_n]

    lines = []
    lines.append("Outright Devigging Example")
    lines.append("=" * 76)
    lines.append(f"{'Team':>22}  {'Odds':>8}  {'Implied':>8}  "
                 f"{'Additive':>8}  {'Shin':>8}")
    lines.append("-" * 76)

    for team in top_teams:
        odds_val = odds_map[team]
        odds_str = f"+{odds_val}" if odds_val > 0 else str(odds_val)
        lines.append(
            f"{team:>22}  {odds_str:>8}  {implied[team]:8.4f}  "
            f"{devigged_add[team]:8.4f}  {devigged_shin[team]:8.4f}"
        )

    lines.append("-" * 76)
    lines.append(f"{'Sum of implied:':>33} {overround:8.4f}  "
                 f"(overround: {(overround - 1) * 100:.1f}%)")
    lines.append(f"{'Sum devigged (Shin):':>33} "
                 f"{sum(devigged_shin.values()):8.4f}")

    pick = max(devigged_shin, key=devigged_shin.get)
    lines.append(f"\nArgmax pick (Shin): {pick} ({devigged_shin[pick]:.4f})")
    return "\n".join(lines)
