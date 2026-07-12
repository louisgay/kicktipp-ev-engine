"""Bonus question optimizer.

Given parsed BonusQuestions (from kicktipp scraper) and market
probabilities (outright odds, Polymarket, etc.), recommends the
EV-maximising answer for each question.

Strategy
--------
- Single-select: pick the option with highest probability (argmax).
- Multi-select (pick K): pick the top-K options by probability.

Each bonus question scores 4 points per correct answer (confirmed
from kicktipp Spielregeln page). For multi-select questions (e.g.
semi-finalists, pick 4), each correct pick earns 4 points independently
- up to 16 points total. This is NOT all-or-nothing.

Since payoff is linear per-pick, the optimal strategy for both
single-select and multi-select is identical: pick the top-K options
by probability (argmax for K=1, top-K for K>1).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.data.clean import normalise_team
from src.data.kicktipp_scrape import BonusQuestion

from .outright_odds import (
    GOLDEN_BOOT_TEAM_PROBS,
    get_group_winner_probs,
    get_semi_finalist_probs,
    get_winner_probs,
)

logger = logging.getLogger(__name__)


# Markets that depend on the unpublished knockout bracket.
# Picks for these are bracket-averaged estimates, not firm recommendations.
_BRACKET_DEPENDENT_MARKETS = {"winner", "semi_finalist"}


@dataclass
class BonusRecommendation:
    """Recommended answer for a single bonus question."""

    question_id: str
    question_text: str
    picks: list[dict]  # [{"label": str, "value": str, "prob": float}, ...]
    question_type: str  # "single" or "multi"
    market_type: str  # "winner", "golden_boot_team", "group_X", "semi_finalist"
    source: str  # "mc_group_sim", "bracket_avg", "curated_espn", etc.
    awaiting_bracket: bool = False  # True if picks depend on unpublished bracket


# -- Question classifier ----------------------------------------------


def _classify_question(q: BonusQuestion) -> str:
    """Infer market type from question text.

    Returns one of:
        'winner', 'golden_boot_team', 'group_X' (X=A..L),
        'semi_finalist', 'unknown'
    """
    text = q.question_text.lower()

    if any(k in text for k in ("world champion", "win the world cup",
                                "win the tournament")):
        return "winner"

    if any(k in text for k in ("goal scorer", "golden boot", "top scorer")):
        return "golden_boot_team"

    if "semi-final" in text or "semi final" in text:
        return "semi_finalist"

    # Group winner: "win group A", "group A winner", etc.
    m = re.search(r"group\s+([a-l])", text, re.IGNORECASE)
    if m:
        return f"group_{m.group(1).upper()}"

    return "unknown"


# -- Core optimizer ----------------------------------------------------


class BonusOptimizer:
    """Recommends optimal bonus answers from market probabilities.

    Reliability tiers:
    - Group winners (group_A..L): RELIABLE - derived from MC simulation
      of kicktipp match-level odds. Independent of bracket structure.
    - Golden boot team: RELIABLE - derived from player-level betting odds.
    - Winner / Semi-finalists: AWAITING BRACKET - derived from
      bracket-agnostic simulation (random pairing, marginalised over
      all possible brackets). Best available estimate given that the
      pool bracket is unpublished.

    Override any market via ``load_probs()``.
    """

    def __init__(self) -> None:
        # market_type -> {canonical_team_name -> probability}
        # Winner + semi-finalists from bracket simulation
        self._markets: dict[str, dict[str, float]] = {
            "winner": dict(get_winner_probs()),
            "golden_boot_team": dict(GOLDEN_BOOT_TEAM_PROBS),
            "semi_finalist": dict(get_semi_finalist_probs()),
        }
        # Add group winner markets (computed via MC simulation)
        for group, probs in get_group_winner_probs().items():
            self._markets[f"group_{group}"] = dict(probs)

    def load_probs(
        self,
        market_type: str,
        probs: dict[str, float],
        source: str = "custom",
    ) -> None:
        """Override probabilities for a market.

        Parameters
        ----------
        market_type : e.g. 'winner', 'group_A', 'golden_boot_team'
        probs : {team_name -> probability} (must sum to ~1)
        source : label for provenance tracking
        """
        # Normalise team names
        normalised = {}
        for team, p in probs.items():
            normalised[normalise_team(team)] = p
        # Re-normalise in case of rounding
        total = sum(normalised.values())
        if total > 0:
            normalised = {t: p / total for t, p in normalised.items()}
        self._markets[market_type] = normalised
        logger.info("Loaded %s probs (%d teams, source=%s)",
                     market_type, len(normalised), source)

    # Default source labels per market type
    _DEFAULT_SOURCES: dict[str, str] = {
        "winner": "bracket_avg",
        "semi_finalist": "bracket_avg",
        "golden_boot_team": "curated_espn",
    }

    def recommend(
        self,
        question: BonusQuestion,
        source: str | None = None,
    ) -> BonusRecommendation:
        """Recommend the optimal answer for a bonus question.

        Parameters
        ----------
        question : parsed BonusQuestion from kicktipp scraper
        source : label for the probability source used (auto-detected if None)

        Returns
        -------
        BonusRecommendation with the top pick(s) and probabilities.
        For bracket-dependent markets (winner, semi_finalist), the
        recommendation is marked ``awaiting_bracket=True``.
        """
        market_type = _classify_question(question)

        if market_type == "unknown":
            logger.warning("Cannot classify question: %s", question.question_text)
            return BonusRecommendation(
                question_id=question.question_id,
                question_text=question.question_text,
                picks=[],
                question_type=question.question_type,
                market_type="unknown",
                source=source or "unknown",
            )

        # Auto-detect source label
        if source is None:
            if market_type.startswith("group_"):
                source = "mc_group_sim"
            else:
                source = self._DEFAULT_SOURCES.get(market_type, "curated_espn")

        bracket_dependent = market_type in _BRACKET_DEPENDENT_MARKETS

        market_probs = self._markets.get(market_type, {})
        if not market_probs:
            logger.warning("No probability data for market: %s", market_type)
            return BonusRecommendation(
                question_id=question.question_id,
                question_text=question.question_text,
                picks=[],
                question_type=question.question_type,
                market_type=market_type,
                source=source,
                awaiting_bracket=bracket_dependent,
            )

        # Match question options to market probabilities
        scored_options = []
        for opt in question.options:
            label = opt["label"]
            canonical = normalise_team(label)
            prob = market_probs.get(canonical, 0.0)
            scored_options.append({
                "label": label,
                "value": opt["value"],
                "prob": round(prob, 6),
                "canonical": canonical,
            })

        # Sort by probability descending
        scored_options.sort(key=lambda x: x["prob"], reverse=True)

        # Select top picks
        if question.question_type == "multi" and question.select_count:
            n_pick = question.select_count
        else:
            n_pick = 1

        picks = scored_options[:n_pick]

        # Remove internal canonical field from output
        for p in picks:
            del p["canonical"]

        # Log unmatched options
        unmatched = [o for o in scored_options if o["prob"] == 0.0]
        if unmatched:
            labels = [o["label"] for o in unmatched]
            logger.debug("Unmatched options for %s: %s",
                         question.question_id, labels)

        return BonusRecommendation(
            question_id=question.question_id,
            question_text=question.question_text,
            picks=picks,
            question_type=question.question_type,
            market_type=market_type,
            source=source,
            awaiting_bracket=bracket_dependent,
        )

    def recommend_all(
        self,
        questions: list[BonusQuestion],
        source: str | None = None,
    ) -> list[BonusRecommendation]:
        """Recommend answers for all bonus questions."""
        return [self.recommend(q, source) for q in questions]


# -- Display helper ----------------------------------------------------


def format_recommendations(recs: list[BonusRecommendation]) -> str:
    """Format recommendations as a readable table."""
    lines = []
    for r in recs:
        lines.append(f"\n[{r.question_id}] {r.question_text}")
        lines.append(f"  Market: {r.market_type} | Source: {r.source}")
        if r.awaiting_bracket:
            lines.append("  ** AWAITING BRACKET - picks are bracket-averaged "
                         "estimates, not firm recommendations **")
        if not r.picks:
            lines.append("  >>> NO RECOMMENDATION (unknown market or no data)")
            continue
        for i, p in enumerate(r.picks, 1):
            prob_pct = p["prob"] * 100
            prefix = "  ~?~ " if r.awaiting_bracket else "  >>> "
            lines.append(f"{prefix}Pick {i}: {p['label']} "
                         f"(val={p['value']}, P={prob_pct:.1f}%)")
    return "\n".join(lines)
