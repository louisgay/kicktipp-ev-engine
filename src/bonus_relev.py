"""Bonus-question relative-EV: flag contested outright questions where the field
leaves separation on the table.

Each bonus question is worth 4 pts and resolves to a single winner, so a CONTESTED
question (no dominant favourite - e.g. a group at ~55%) on which the field
over-concentrates on the consensus favourite is the cheapest separation in the
whole pool: picking a live alternative is a lone-correct-picker lottery.

Recommend a deviation ONLY where the question is BOTH:
  * contested      - the favourite's (sharp) probability is below ``contested_max``; and
  * over-concentrated - the consensus-anchored field is predicted to pile on the
    favourite MORE than its true probability warrants.
Otherwise take the favourite (a dominant favourite is everyone's pick - deviating
is pure -EV with no separation upside).

Probabilities are SOURCE-AGNOSTIC - pass whatever you trust. The default source
pulls tournament winner from the bracket sim and golden boot from curated
(sharp-ish) outright odds; group winners come from the MC sim, which is
kicktipp-fed and FAVOURITE-OVERCONFIDENT - so contestedness is HIDDEN there and
the group analysis only becomes meaningful once fed SHARP group-winner
probabilities.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def analyse_question(probs: dict[str, float], *, contested_max: float = 0.60,
                     field_temp: float = 0.5) -> dict:
    """Analyse one single-select bonus question from its (sharp) option probs."""
    items = sorted(probs.items(), key=lambda x: -x[1])
    fav, p1 = items[0]
    runner, p2 = items[1] if len(items) > 1 else (None, 0.0)

    contested = p1 < contested_max
    # Consensus-anchored field: sharpen the probs toward the favourite (temp < 1
    # over-concentrates). field_fav is the share of the field expected on the favourite.
    sharp = {o: p ** (1.0 / field_temp) for o, p in probs.items()}
    z = sum(sharp.values()) or 1.0
    field_share = {o: v / z for o, v in sharp.items()}
    field_fav = field_share[fav]
    over_concentrated = field_fav > p1          # field piles on more than the true prob

    deviate = bool(contested and over_concentrated and runner is not None)
    ev_cost = round(4.0 * (p1 - p2), 3) if deviate else 0.0          # 4 pts per question
    # Upside: when the runner wins (~p2) while the field sits on the favourite (~field_fav)
    # we are (nearly) the lone correct picker -> +4 separation on most of the field.
    separation_upside = round(4.0 * p2 * field_fav, 3) if deviate else 0.0

    return {
        "favourite": fav, "p_fav": round(p1, 3),
        "runner_up": runner, "p_runner": round(p2, 3),
        "contested": contested, "field_fav_share": round(field_fav, 3),
        "over_concentrated": over_concentrated,
        "recommend": "DEVIATE" if deviate else "favourite",
        "target": runner if deviate else fav,
        "ev_cost": ev_cost, "separation_upside": separation_upside,
    }


def analyse_bonus(prob_sources: dict[str, dict[str, float]], **kw) -> dict[str, dict]:
    """Run analyse_question over a set of questions {label -> {option: prob}}."""
    return {label: analyse_question(probs, **kw) for label, probs in prob_sources.items()}


def build_default_sources() -> dict[str, dict[str, float]]:
    """Pull bonus-question probabilities from the existing bonus/ modules.

    Tournament winner = bracket sim; golden boot = curated (sharp-ish) outright odds;
    group winners = MC sim (kicktipp-fed, FAVOURITE-OVERCONFIDENT - feed sharp group
    probs for a real contested signal). Semi-finalists (multi-select) are omitted.
    """
    from src.bonus.outright_odds import (
        GOLDEN_BOOT_TEAM_PROBS, get_group_winner_probs, get_winner_probs,
    )
    sources: dict[str, dict[str, float]] = {}
    for g, probs in get_group_winner_probs().items():
        sources[f"Group {g} winner"] = dict(probs)
    sources["Tournament winner"] = dict(get_winner_probs())
    sources["Golden boot (team)"] = dict(GOLDEN_BOOT_TEAM_PROBS)
    return sources


def format_report(results: dict[str, dict]) -> str:
    lines = ["Bonus relative-EV scan (DEVIATE only where contested AND field over-concentrates)",
             f"{'Question':<22}{'favourite':>16}{'p_fav':>7}{'cont?':>6}{'fieldFav':>9}  recommend"]
    lines.append("-" * 78)
    for label, a in results.items():
        rec = a["recommend"]
        tail = (f" -> {a['target']} (EV cost {a['ev_cost']}, sep. upside {a['separation_upside']})"
                if rec == "DEVIATE" else "")
        flag = "   <<<" if rec == "DEVIATE" else ""
        lines.append(f"{label:<22}{str(a['favourite'])[:16]:>16}{a['p_fav']:>7.0%}"
                     f"{str(a['contested']):>6}{a['field_fav_share']:>9.0%}  {rec}{tail}{flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(format_report(analyse_bonus(build_default_sources())))
