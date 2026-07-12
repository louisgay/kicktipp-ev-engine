"""Bracket-agnostic tournament simulation.

Simulates groups -> knockout rounds -> Final to derive:
- P(win tournament) for each team
- P(reach semi-final) for each team

Since the pool community uses a custom draw (12/12 groups differ
from the real FIFA WC 2026), and the knockout bracket is unpublished
("unknown vs unknown" on all matchdays 11-15), we CANNOT use the FIFA
bracket template.

Instead we use a bracket-agnostic approach: after simulating the
group stage, qualifying teams are paired randomly in each knockout
round. This marginalises over all possible bracket structures and
avoids encoding false pairing assumptions.

The resulting probabilities are bracket-averaged estimates:
- Less precise than a simulation with the real bracket
- But NOT wrong (unlike using a fake bracket)
- Suitable for pre-tournament picks (deadline 11 June, before bracket
  publication)

STATUS: AWAITING BRACKET - these are bracket-averaged estimates.
Once the bracket is published, switch to deterministic pairings for
more precise probabilities.

Tournament format (pool):
- 12 groups of 4 -> top 2 qualify + 8 best 3rd-place -> 32 teams
- R32 (16 matches) -> R16 (8) -> QF (4) -> SF (2) -> 3rd-place + Final
- MD15 has 2 matches: 3rd-place (18/07) and Final (19/07)
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from src.bonus.group_sim import (
    ALL_TEAMS,
    POOL_GROUPS,
    _simulate_match,
    build_group_lambdas,
    simulate_group,
)

logger = logging.getLogger(__name__)


# -- Team strength estimation ----------------------------------------


def _estimate_team_strengths(
    match_lambdas: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, float]:
    """Estimate per-team attack strength from group match λ values.

    Averages λ_attack across all group matches for each team.
    Used as a proxy for knockout match λ prediction.

    Returns {team -> avg_lambda_attack}.
    """
    team_attack: dict[str, list[float]] = {}

    for (home, away), (lam_h, lam_a) in match_lambdas.items():
        team_attack.setdefault(home, []).append(lam_h)
        team_attack.setdefault(away, []).append(lam_a)

    return {t: float(np.mean(vals)) for t, vals in team_attack.items()}


def _knockout_lambdas(
    team_a: str,
    team_b: str,
    team_strengths: dict[str, float],
    avg_total: float = 2.3,
) -> tuple[float, float]:
    """Estimate (λ_a, λ_b) for a knockout match between two teams.

    Uses relative attack strengths to split the expected total goals.
    Knockout matches are neutral venue, so no home advantage.

    Parameters
    ----------
    team_a, team_b : team names
    team_strengths : {team -> avg_attack_lambda} from group stage
    avg_total : expected total goals in a knockout match
    """
    str_a = team_strengths.get(team_a, 0.8)
    str_b = team_strengths.get(team_b, 0.8)

    total = str_a + str_b
    if total < 0.01:
        return avg_total / 2, avg_total / 2

    # Scale so total goals ≈ avg_total (knockout matches are lower-scoring)
    scale = avg_total / total
    return str_a * scale, str_b * scale


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    team_strengths: dict[str, float],
    rho: float,
    rng: np.random.Generator,
    avg_total: float = 2.3,
) -> str:
    """Simulate a single knockout match and return the winner.

    Draws go to penalties (weighted coin flip by team strength).
    """
    lam_a, lam_b = _knockout_lambdas(team_a, team_b, team_strengths, avg_total)
    gh, ga = _simulate_match(lam_a, lam_b, rho, rng)

    if gh > ga:
        return team_a
    elif ga > gh:
        return team_b
    else:
        # Penalty shootout: slight edge to stronger team
        p_a = team_strengths.get(team_a, 0.8)
        p_b = team_strengths.get(team_b, 0.8)
        return team_a if rng.random() < p_a / (p_a + p_b) else team_b


# -- Full tournament simulation --------------------------------------


def simulate_tournament(
    match_lambdas: dict[tuple[str, str], tuple[float, float]] | None = None,
    rho: float = -0.04,
    n_sims: int = 50_000,
    seed: int = 42,
    knockout_total: float = 2.3,
) -> dict[str, dict[str, float]]:
    """Simulate the full tournament with bracket-agnostic knockout pairing.

    Group stage: full MC simulation using match λ values.
    Knockout stage: qualifying teams are randomly paired each round.
    This marginalises over all possible bracket structures.

    Returns
    -------
    Dict with keys:
        'winner': {team -> P(win tournament)}
        'semi_finalist': {team -> P(reach semi-final)}
        'qualify_knockout': {team -> P(qualify from group)}
    """
    if match_lambdas is None:
        match_lambdas = build_group_lambdas(rho=rho)

    team_strengths = _estimate_team_strengths(match_lambdas)

    rng = np.random.default_rng(seed)

    # Counters
    qualify_count: Counter[str] = Counter()
    sf_count: Counter[str] = Counter()
    win_count: Counter[str] = Counter()

    # Pre-simulate all groups (store full rankings per sim)
    logger.info("Simulating %d group stages...", n_sims)
    group_sims: dict[str, list[list[str]]] = {}
    for group in sorted(POOL_GROUPS.keys()):
        group_sims[group] = simulate_group(
            group, match_lambdas, rho, n_sims, rng,
        )

    logger.info("Simulating %d knockout brackets (random pairing)...", n_sims)
    for sim in range(n_sims):
        # -- Determine qualifiers --
        group_rankings: dict[str, list[str]] = {}
        for group in sorted(POOL_GROUPS.keys()):
            group_rankings[group] = group_sims[group][sim]

        # Top 2 from each group = 24 teams
        auto_qualifiers: list[str] = []
        for group in sorted(group_rankings):
            auto_qualifiers.append(group_rankings[group][0])  # 1st
            auto_qualifiers.append(group_rankings[group][1])  # 2nd

        # 8 best 3rd-place teams (sorted by strength as proxy for pts/GD)
        third_place_teams = []
        for group in sorted(group_rankings):
            third_place_teams.append(group_rankings[group][2])

        third_place_teams.sort(
            key=lambda t: team_strengths.get(t, 0) + rng.random() * 0.1,
            reverse=True,
        )
        qualifying_thirds = third_place_teams[:8]

        qualifiers = auto_qualifiers + qualifying_thirds
        assert len(qualifiers) == 32

        for t in qualifiers:
            qualify_count[t] += 1

        # -- Knockout rounds (bracket-agnostic: random pairing) --
        pool = list(qualifiers)
        rng.shuffle(pool)

        # R32: 32 -> 16
        r32_winners = []
        for i in range(0, 32, 2):
            winner = _simulate_knockout_match(
                pool[i], pool[i + 1], team_strengths, rho, rng, knockout_total,
            )
            r32_winners.append(winner)

        # R16: 16 -> 8
        rng.shuffle(r32_winners)
        r16_winners = []
        for i in range(0, 16, 2):
            winner = _simulate_knockout_match(
                r32_winners[i], r32_winners[i + 1],
                team_strengths, rho, rng, knockout_total,
            )
            r16_winners.append(winner)

        # QF: 8 -> 4
        rng.shuffle(r16_winners)
        qf_winners = []
        for i in range(0, 8, 2):
            winner = _simulate_knockout_match(
                r16_winners[i], r16_winners[i + 1],
                team_strengths, rho, rng, knockout_total,
            )
            qf_winners.append(winner)

        # SF: 4 -> 2 winners + 2 losers
        rng.shuffle(qf_winners)
        sf_winners = []
        sf_losers = []
        for i in range(0, 4, 2):
            team_a, team_b = qf_winners[i], qf_winners[i + 1]
            # All 4 QF winners reach the semi-final
            sf_count[team_a] += 1
            sf_count[team_b] += 1
            winner = _simulate_knockout_match(
                team_a, team_b, team_strengths, rho, rng, knockout_total,
            )
            loser = team_b if winner == team_a else team_a
            sf_winners.append(winner)
            sf_losers.append(loser)

        # 3rd-place match (MD15 match 1): sf_losers[0] vs sf_losers[1]
        # (tracked but not used for bonus questions)

        # Final (MD15 match 2): sf_winners[0] vs sf_winners[1]
        champion = _simulate_knockout_match(
            sf_winners[0], sf_winners[1],
            team_strengths, rho, rng, knockout_total,
        )
        win_count[champion] += 1

    # -- Normalise to probabilities --
    all_teams = sorted(ALL_TEAMS)
    winner_probs = {t: win_count[t] / n_sims for t in all_teams}
    sf_probs = {t: sf_count[t] / n_sims for t in all_teams}
    qualify_probs = {t: qualify_count[t] / n_sims for t in all_teams}

    # Log top teams
    top_winners = sorted(winner_probs.items(), key=lambda x: x[1], reverse=True)[:8]
    logger.info("Top 8 winners (bracket-averaged): %s",
                ", ".join(f"{t} {p:.1%}" for t, p in top_winners))

    top_sf = sorted(sf_probs.items(), key=lambda x: x[1], reverse=True)[:8]
    logger.info("Top 8 semi-finalists (bracket-averaged): %s",
                ", ".join(f"{t} {p:.1%}" for t, p in top_sf))

    return {
        "winner": winner_probs,
        "semi_finalist": sf_probs,
        "qualify_knockout": qualify_probs,
    }


# -- Convenience access ----------------------------------------------

_TOURNAMENT_CACHE: dict[str, dict[str, float]] | None = None


def get_tournament_probs(
    force: bool = False,
    n_sims: int = 50_000,
) -> dict[str, dict[str, float]]:
    """Get tournament probabilities (cached after first call).

    Returns dict with keys 'winner', 'semi_finalist', etc.
    These are bracket-averaged estimates (bracket-agnostic).
    """
    global _TOURNAMENT_CACHE
    if _TOURNAMENT_CACHE is None or force:
        _TOURNAMENT_CACHE = simulate_tournament(n_sims=n_sims)
    return _TOURNAMENT_CACHE
