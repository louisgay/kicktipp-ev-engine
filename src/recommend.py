"""Consolidated recommendation engine.

Single entry point that produces the full prediction sheet for a
kicktipp matchday:

1. Match predictions: kicktipp 1X2 -> score matrix -> optimal (h, a)
2. Bonus answers: outright market probs -> argmax / top-K

Modes
-----
- Offline (default): parse saved HTML files, use curated odds.
- Live: scrape kicktipp + fetch Odds API O/U, use live markets.

Usage
-----
    # Offline (from fixtures / saved HTML)
    python -m src.recommend --matchday tests/fixtures/predict_matchday1.html \\
                            --bonus tests/fixtures/bonus.html

    # Live (requires KICKTIPP_SESSION + ODDS_API_KEY)
    python -m src.recommend --live --md 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.bonus.optimizer import BonusOptimizer, BonusRecommendation, format_recommendations
from src.data.clean import normalise_team
from src.data.kicktipp_scrape import (
    BonusQuestion,
    MatchOdds,
    parse_bonus_page,
    parse_prediction_page,
    scrape_bonus,
    scrape_matchday,
)
from src.models.odds_model import KicktippOddsModel
from src.odds.reconstruct import _1x2_from_matrix
from src.scoring.kicktipp import optimal_prediction

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


# -- Data classes -----------------------------------------------------


@dataclass
class MatchRecommendation:
    """Recommended prediction for a single match."""
    home_team: str
    away_team: str
    datetime_str: str
    pred_home: int
    pred_away: int
    ev: float             # expected kicktipp points
    prob_home: float      # P(home win) from kicktipp odds
    prob_draw: float
    prob_away: float
    has_ou: bool          # whether O/U constraint was used
    ou_source: str = "market"  # "market", "model" (1X2-only), or "none"


@dataclass
class RecommendationSheet:
    """Full prediction sheet for one matchday + bonus."""
    matches: list[MatchRecommendation]
    bonus: list[BonusRecommendation]
    total_ev: float       # sum of match EVs (bonus is binary, not EV-additive)


# -- O/U fetching ----------------------------------------------------


def fetch_live_totals(matches: list[MatchOdds]) -> list[dict]:
    """Fetch live O/U 2.5 odds from The Odds API and match to kicktipp matches.

    Calls get_odds() for the totals market, extracts O/U 2.5 from
    bookmaker data directly, and matches API events to kicktipp
    matches by normalised team name.

    Prefers O/U 2.5 line; if unavailable, uses the closest available
    line (e.g. 2.25 or 2.75) as an approximation.

    Returns totals_records list ready for model.load_totals().
    """
    from src.odds.client import get_odds

    events = get_odds(
        sport="soccer_fifa_world_cup",
        markets="h2h,totals",   # one call carries totals + sharp h2h (2 credits)
        regions="eu",
    )

    # Build lookup from normalised team names -> best totals odds
    api_totals: dict[tuple[str, str], dict] = {}
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        if not home or not away:
            continue

        # Collect all totals across bookmakers, prefer line 2.5
        best_totals = None
        best_line_dist = float("inf")
        best_book = None

        for bm in ev.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                outcomes = mkt.get("outcomes", [])
                over_price = under_price = line = None
                for o in outcomes:
                    if o.get("name") == "Over":
                        over_price = o["price"]
                        line = o.get("point")
                    elif o.get("name") == "Under":
                        under_price = o["price"]
                if over_price and under_price and line is not None:
                    dist = abs(line - 2.5)
                    if dist < best_line_dist:
                        best_line_dist = dist
                        best_totals = [over_price, under_price]
                        best_book = bm.get("key", "unknown")
                        if dist == 0:
                            break  # exact 2.5, stop searching
            if best_line_dist == 0:
                break

        if best_totals is None:
            continue

        h = normalise_team(home)
        a = normalise_team(away)
        # Use 2.5 as the line for the model regardless of actual line
        # (the model always uses 2.5; nearby lines are close enough)
        api_totals[(h, a)] = {
            "home_team": home,
            "away_team": away,
            "totals_odds": best_totals,
            "totals_line": 2.5,
            "actual_line": 2.5 - best_line_dist if best_line_dist > 0 else 2.5,
            "bookmaker": best_book,
        }
        logger.debug("O/U for %s vs %s: line=%.2f odds=[%.2f, %.2f] book=%s",
                      h, a, 2.5 - best_line_dist if best_line_dist > 0 else 2.5,
                      best_totals[0], best_totals[1], best_book)

    totals_records: list[dict] = []
    matched = []
    unmatched = []

    for m in matches:
        h = normalise_team(m.home_team)
        a = normalise_team(m.away_team)
        key = (h, a)
        if key in api_totals:
            rec = api_totals[key]
            totals_records.append({
                "home_team": rec["home_team"],
                "away_team": rec["away_team"],
                "totals_odds": rec["totals_odds"],
                "totals_line": rec["totals_line"],
            })
            matched.append(f"{h} vs {a} (book={rec['bookmaker']})")
        else:
            unmatched.append(f"{h} vs {a}")

    logger.info("O/U matched: %d/%d - %s",
                len(matched), len(matches), ", ".join(matched))
    if unmatched:
        logger.warning("O/U unmatched: %s", ", ".join(unmatched))

    return totals_records


def fetch_live_sharp_1x2(matches: list[MatchOdds]) -> list[dict]:
    """Pinnacle-led sharp h2h per kicktipp match (consensus fallback).

    Reuses the SAME cached ``h2h,totals`` events (no extra credits) and extracts
    h2h via ``extract_odds_for_event`` (prefers Pinnacle/Betfair, else consensus)
    so the live sharp 1X2 matches the Pinnacle-led odds validated in calibration.py.
    Returns records ready for ``KicktippOddsModel.load_sharp_1x2()``.
    """
    from src.odds.client import get_odds
    from src.odds.reconstruct import extract_odds_for_event

    events = get_odds(sport="soccer_fifa_world_cup", markets="h2h,totals", regions="eu")
    by_key: dict[tuple[str, str], list[float]] = {}
    for ev in events:
        rec = extract_odds_for_event(ev)
        if rec and rec.get("h2h_odds"):
            by_key[(normalise_team(rec["home_team"]),
                    normalise_team(rec["away_team"]))] = rec["h2h_odds"]

    out = []
    for m in matches:
        k = (normalise_team(m.home_team), normalise_team(m.away_team))
        if k in by_key:
            out.append({"home_team": m.home_team, "away_team": m.away_team,
                        "h2h_odds": by_key[k]})
    logger.info("Sharp 1X2 matched: %d/%d", len(out), len(matches))
    return out


# -- Core pipeline ----------------------------------------------------


def build_match_recommendations(
    matches: list[MatchOdds],
    totals_records: list[dict] | None = None,
    rho: float = -0.04,
    max_goals: int = 8,
    allow_model_total: bool = False,
    sharp_records: list[dict] | None = None,
    blend_weight: float = 0.0,
) -> list[MatchRecommendation]:
    """Convert parsed kicktipp odds -> optimal score predictions.

    Parameters
    ----------
    matches : parsed MatchOdds from kicktipp scraper
    totals_records : optional O/U data from Odds API
        [{home_team, away_team, totals_odds: [over, under], totals_line: 2.5}]
    rho : Dixon-Coles correction parameter
    max_goals : score grid size
    allow_model_total : if True, fall back to 1X2-only reconstruction
        for matches without market O/U (labelled ``ou_source='model'``).
        If False (default), raises ValueError for missing O/U.
    """
    model = KicktippOddsModel(rho=rho, max_goals=max_goals, require_ou=False,
                              blend_weight=blend_weight)
    model.load_kicktipp_odds(matches)
    if sharp_records:
        model.load_sharp_1x2(sharp_records)
    if totals_records:
        model.load_totals(totals_records)

    # Anti-fabrication: verify all matches have O/U data
    missing_ou = []
    for m in matches:
        h = normalise_team(m.home_team)
        a = normalise_team(m.away_team)
        if (h, a) not in model._p_over:
            missing_ou.append(f"{h} vs {a}")

    if missing_ou and not allow_model_total:
        raise ValueError(
            f"O/U data missing for {len(missing_ou)} match(es): "
            + ", ".join(missing_ou)
            + ". Provide O/U totals via --totals (offline) or --live mode. "
            + "Or pass allow_model_total=True for 1X2-only fallback."
        )
    elif missing_ou:
        logger.warning(
            "O/U data missing for %d match(es) - using 1X2-only "
            "reconstruction (model total, no market O/U): %s",
            len(missing_ou), ", ".join(missing_ou),
        )

    recommendations = []
    for m in matches:
        h = normalise_team(m.home_team)
        a = normalise_team(m.away_team)
        has_ou = (h, a) in model._p_over

        mat = model.predict_score_matrix(h, a)
        (pred_h, pred_a), ev = optimal_prediction(mat)

        ou_source = "market" if has_ou else "model"

        recommendations.append(MatchRecommendation(
            home_team=m.home_team,
            away_team=m.away_team,
            datetime_str=m.datetime_str,
            pred_home=int(pred_h),
            pred_away=int(pred_a),
            ev=round(ev, 3),
            prob_home=m.prob_home,
            prob_draw=m.prob_draw,
            prob_away=m.prob_away,
            has_ou=has_ou,
            ou_source=ou_source,
        ))

    return recommendations


def build_bonus_recommendations(
    questions: list[BonusQuestion],
    custom_probs: dict[str, dict[str, float]] | None = None,
    source: str = "mc_simulation",
) -> list[BonusRecommendation]:
    """Convert parsed bonus questions -> optimal answers.

    Parameters
    ----------
    questions : parsed BonusQuestions from kicktipp scraper
    custom_probs : optional overrides {market_type -> {team -> prob}}
    source : label for probability source
    """
    optimizer = BonusOptimizer()

    if custom_probs:
        for market_type, probs in custom_probs.items():
            optimizer.load_probs(market_type, probs, source=source)
    return optimizer.recommend_all(questions, source=source)


def build_recommendation_sheet(
    matches: list[MatchOdds],
    questions: list[BonusQuestion] | None = None,
    totals_records: list[dict] | None = None,
    rho: float = -0.04,
    max_goals: int = 8,
    allow_model_total: bool = False,
    sharp_records: list[dict] | None = None,
    blend_weight: float = 0.0,
) -> RecommendationSheet:
    """Build the complete recommendation sheet."""
    match_recs = build_match_recommendations(
        matches, totals_records, rho, max_goals,
        allow_model_total=allow_model_total,
        sharp_records=sharp_records, blend_weight=blend_weight,
    )
    bonus_recs = (
        build_bonus_recommendations(questions)
        if questions else []
    )
    total_ev = sum(r.ev for r in match_recs)

    return RecommendationSheet(
        matches=match_recs,
        bonus=bonus_recs,
        total_ev=round(total_ev, 2),
    )


# -- Display ----------------------------------------------------------


def format_match_recommendations(recs: list[MatchRecommendation]) -> str:
    """Format match recommendations as a table."""
    lines = []
    lines.append(
        f"{'Date':>16}  {'Home':>20}  {'Pred':>5}  {'Away':<20}  "
        f"{'EV':>5}  {'P(H)':>5} {'P(D)':>5} {'P(A)':>5}  {'O/U':>6}"
    )
    lines.append("-" * 103)

    for r in recs:
        pred_str = f"{r.pred_home}-{r.pred_away}"
        ou_str = r.ou_source if r.has_ou else "model"
        lines.append(
            f"{r.datetime_str:>16}  {r.home_team:>20}  {pred_str:>5}  "
            f"{r.away_team:<20}  {r.ev:5.2f}  "
            f"{r.prob_home:5.1%} {r.prob_draw:5.1%} {r.prob_away:5.1%}  {ou_str:>6}"
        )

    return "\n".join(lines)


def format_sheet(sheet: RecommendationSheet) -> str:
    """Format the full recommendation sheet."""
    sections = []

    # Header
    sections.append("=" * 100)
    sections.append("  KICKTIPP EV-ENGINE - PREDICTION SHEET")
    sections.append("=" * 100)

    # Matches
    if sheet.matches:
        sections.append(f"\n  MATCH PREDICTIONS ({len(sheet.matches)} matches)")
        sections.append(f"  Expected total: {sheet.total_ev:.1f} pts "
                        f"(avg {sheet.total_ev / len(sheet.matches):.2f} per match)")
        sections.append("")
        sections.append(format_match_recommendations(sheet.matches))

    # Bonus
    if sheet.bonus:
        sections.append(f"\n{'=' * 100}")
        sections.append(f"  BONUS PREDICTIONS ({len(sheet.bonus)} questions, "
                        f"4 pts each if correct)")
        sections.append("")
        sections.append(format_recommendations(sheet.bonus))

    sections.append(f"\n{'=' * 100}")
    return "\n".join(sections)


# -- Bonus summary (standalone, no kicktipp HTML required) ------------


def generate_bonus_summary() -> str:
    """Generate a complete summary of all bonus EV-max picks.

    Produces the full answer set directly from probability sources,
    without requiring the kicktipp bonus HTML page. Useful for
    pre-tournament preparation and offline review.

    Covers all 15 bonus questions:
    - 12 group winners (A-L) - MC simulation, RELIABLE
    - 1 tournament winner - bracket-averaged, AWAITING BRACKET
    - 1 golden boot team - ESPN/Shin devigged, RELIABLE
    - 1 semi-finalists (pick 4) - bracket-averaged, AWAITING BRACKET
    """
    from src.bonus.outright_odds import (
        GOLDEN_BOOT_TEAM_PROBS,
        get_group_winner_probs,
        get_semi_finalist_probs,
        get_winner_probs,
    )

    lines = []
    lines.append("=" * 80)
    lines.append("  BONUS EV-MAX ANSWER SET - COMPLETE")
    lines.append("=" * 80)

    # Group winners
    lines.append("\n  GROUP WINNERS (MC simulation, 100K sims, RELIABLE)")
    lines.append(f"  {'Group':>7}  {'Pick':>20}  {'P(win)':>8}  {'Runner-up':>20}  {'P':>8}")
    lines.append("  " + "-" * 72)
    gw = get_group_winner_probs()
    for g in sorted(gw.keys()):
        ranked = sorted(gw[g].items(), key=lambda x: x[1], reverse=True)
        pick, p1 = ranked[0]
        runner, p2 = ranked[1]
        lines.append(
            f"  {g:>7}  {pick:>20}  {p1:8.1%}  {runner:>20}  {p2:8.1%}"
        )

    # Golden boot team
    lines.append("\n  GOLDEN BOOT TEAM (ESPN/BetMGM, Shin devig, RELIABLE)")
    gb_ranked = sorted(GOLDEN_BOOT_TEAM_PROBS.items(), key=lambda x: x[1], reverse=True)
    pick_gb = gb_ranked[0][0]
    lines.append(f"  >>> Pick: {pick_gb} ({gb_ranked[0][1]:.1%})")
    for t, p in gb_ranked[:5]:
        lines.append(f"       {t:>15}: {p:.1%}")

    # Tournament winner
    lines.append("\n  TOURNAMENT WINNER (bracket-avg, AWAITING BRACKET)")
    wp = get_winner_probs()
    wp_ranked = sorted(wp.items(), key=lambda x: x[1], reverse=True)
    pick_w = wp_ranked[0][0]
    lines.append(f"  ~?~ Pick: {pick_w} ({wp_ranked[0][1]:.1%})")
    for t, p in wp_ranked[:8]:
        lines.append(f"       {t:>15}: {p:.1%}")

    # Semi-finalists
    lines.append("\n  SEMI-FINALISTS (pick 4, bracket-avg, AWAITING BRACKET)")
    sf = get_semi_finalist_probs()
    sf_ranked = sorted(sf.items(), key=lambda x: x[1], reverse=True)
    picks_sf = [t for t, _ in sf_ranked[:4]]
    lines.append(f"  ~?~ Picks: {', '.join(picks_sf)}")
    for t, p in sf_ranked[:8]:
        marker = " <--" if t in picks_sf else ""
        lines.append(f"       {t:>15}: {p:.1%}{marker}")

    lines.append(f"\n{'=' * 80}")
    lines.append("  RELIABILITY KEY:")
    lines.append("  >>> RELIABLE - independent of bracket structure")
    lines.append("  ~?~ AWAITING BRACKET - bracket-averaged estimates")
    lines.append(f"{'=' * 80}")

    return "\n".join(lines)


# -- Tournament runner ------------------------------------------------


@dataclass
class MatchdayDeadline:
    """Deadline metadata for a single matchday."""
    matchday: int
    first_kickoff: str      # earliest match datetime string
    n_matches: int


def scrape_deadlines(
    n_matchdays: int = 15,
) -> list[MatchdayDeadline]:
    """Scrape deadline info for all matchdays from kicktipp.

    For each matchday, returns the first match datetime (= deadline)
    and how many matches are in that matchday.

    Requires KICKTIPP_SESSION. Read-only GET requests only.
    """
    from src.data.kicktipp_scrape import _make_session, _fetch_page, BASE_URL, COMMUNITY

    session = _make_session()
    deadlines = []

    for md in range(1, n_matchdays + 1):
        url = f"{BASE_URL}/{COMMUNITY}/tippabgabe?spieltagIndex={md}"
        soup = _fetch_page(session, url)
        matches = parse_prediction_page(str(soup))
        if not matches:
            logger.info("No matches for matchday %d, stopping", md)
            break

        # First match datetime is the deadline
        first_dt = matches[0].datetime_str
        deadlines.append(MatchdayDeadline(
            matchday=md,
            first_kickoff=first_dt,
            n_matches=len(matches),
        ))
        logger.info("MD%d: %d matches, deadline %s", md, len(matches), first_dt)

    return deadlines


def format_deadlines(deadlines: list[MatchdayDeadline]) -> str:
    """Format deadline calendar as a table."""
    lines = ["  DEADLINE CALENDAR"]
    lines.append(f"  {'MD':>3}  {'Deadline':>20}  {'Matches':>8}")
    lines.append("  " + "-" * 40)
    for d in deadlines:
        lines.append(f"  {d.matchday:>3}  {d.first_kickoff:>20}  {d.n_matches:>8}")
    return "\n".join(lines)


def run_matchday(
    matchday: int,
    include_bonus: bool = True,
    rho: float = -0.04,
    allow_model_total: bool = True,
) -> RecommendationSheet:
    """Run the full pipeline for a single matchday.

    1. Scrape fresh 1X2 odds from kicktipp
    2. Fetch fresh O/U from The Odds API
    3. For matches without O/U: fall back to 1X2-only (labelled)
    4. Build match recommendations (EV-max)
    5. Optionally process bonus questions

    Requires KICKTIPP_SESSION and ODDS_API_KEY env vars.
    All requests are read-only GET.

    Parameters
    ----------
    matchday : matchday index (1-based)
    include_bonus : whether to also process bonus questions
    rho : Dixon-Coles ρ parameter
    allow_model_total : allow 1X2-only fallback for missing O/U
    """
    logger.info("=" * 60)
    logger.info("RUNNING MATCHDAY %d", matchday)
    logger.info("=" * 60)

    # Step 1: fresh 1X2 from kicktipp
    logger.info("Step 1: Scraping kicktipp matchday %d...", matchday)
    matches = scrape_matchday(matchday)
    if not matches:
        raise RuntimeError(f"No matches found for matchday {matchday}")
    logger.info("  Got %d matches", len(matches))

    # Step 2: fresh O/U from Odds API
    logger.info("Step 2: Fetching O/U from Odds API...")
    try:
        totals_records = fetch_live_totals(matches)
    except Exception as e:
        logger.warning("Odds API fetch failed: %s", e)
        totals_records = []

    # Step 2b: sharp h2h (same cached call) + blend weight from config
    import yaml
    blend_weight = (yaml.safe_load((_ROOT / "config" / "config.yaml").read_text())
                    .get("model", {}).get("blend_weight", 0.0))
    try:
        sharp_records = fetch_live_sharp_1x2(matches)
    except Exception as e:
        logger.warning("Sharp h2h fetch failed: %s", e)
        sharp_records = []
    logger.info("  Sharp 1X2: %d matches | blend_weight=%.2f",
                len(sharp_records), blend_weight)

    n_with_ou = len(totals_records) if totals_records else 0
    n_without = len(matches) - n_with_ou
    logger.info("  O/U matched: %d/%d", n_with_ou, len(matches))
    if n_without > 0:
        logger.warning(
            "  %d match(es) without market O/U - %s",
            n_without,
            "using 1X2-only model total" if allow_model_total else "WILL RAISE",
        )

    # Step 3: bonus questions
    questions: list[BonusQuestion] = []
    if include_bonus:
        logger.info("Step 3: Scraping bonus questions...")
        questions = scrape_bonus(matchday=matchday)
        logger.info("  Got %d bonus questions", len(questions))

    # Step 4: build recommendations
    logger.info("Step 4: Building recommendations...")
    sheet = build_recommendation_sheet(
        matches=matches,
        questions=questions if questions else None,
        totals_records=totals_records if totals_records else None,
        rho=rho,
        allow_model_total=allow_model_total,
        sharp_records=sharp_records if sharp_records else None,
        blend_weight=blend_weight,
    )

    # Step 5: summary
    n_model = sum(1 for r in sheet.matches if r.ou_source == "model")
    logger.info("  %d match recommendations (EV total: %.1f)",
                len(sheet.matches), sheet.total_ev)
    if n_model > 0:
        logger.warning("  %d match(es) use model total (no market O/U)", n_model)
    if sheet.bonus:
        n_awaiting = sum(1 for b in sheet.bonus if b.awaiting_bracket)
        logger.info("  %d bonus recommendations (%d awaiting bracket)",
                    len(sheet.bonus), n_awaiting)

    # Step 6: log the full per-match tuple (kicktipp, sharp, O/U) + backfill results
    try:
        from src.snapshot import backfill_results, record_match
        from src.odds.devig import devig_1x2, devig_over_under
        sharp_lk = {(normalise_team(r["home_team"]), normalise_team(r["away_team"])): r["h2h_odds"]
                    for r in (sharp_records or [])}
        ou_lk = {(normalise_team(r["home_team"]), normalise_team(r["away_team"])): r["totals_odds"]
                 for r in (totals_records or [])}
        for i, m in enumerate(matches):
            k = (normalise_team(m.home_team), normalise_team(m.away_team))
            sharp = devig_1x2(*sharp_lk[k], method="shin") if k in sharp_lk else None
            over = devig_over_under(*ou_lk[k], method="normalise")[0] if k in ou_lk else None
            record_match(matchday, i, m.home_team, m.away_team,
                         kicktipp_1x2=(m.prob_home, m.prob_draw, m.prob_away),
                         sharp_1x2=sharp, ou_over_2_5=over)
        from src.data.leaderboard import scrape_leaderboard
        backfill_results(matchday, scrape_leaderboard(spieltag=matchday).fixtures)
        logger.info("  Snapshot: logged %d matches + backfilled results", len(matches))
    except Exception as e:
        logger.warning("Snapshot logging failed (non-fatal): %s", e)

    return sheet


# -- CLI --------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate kicktipp prediction sheet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--matchday", type=str, default=None,
        help="Path to saved matchday HTML file (offline mode)",
    )
    mode.add_argument(
        "--live", action="store_true",
        help="Scrape kicktipp live (requires KICKTIPP_SESSION)",
    )
    mode.add_argument(
        "--deadlines", action="store_true",
        help="Scrape and display all matchday deadlines",
    )
    mode.add_argument(
        "--bonus-summary", action="store_true",
        help="Display complete bonus EV-max answer set (no network required)",
    )
    p.add_argument(
        "--bonus", type=str, default=None,
        help="Path to saved bonus HTML file (offline mode)",
    )
    p.add_argument(
        "--totals", type=str, default=None,
        help="Path to saved O/U totals JSON file (offline mode)",
    )
    p.add_argument(
        "--md", type=int, default=1,
        help="Matchday index for live mode (default: 1)",
    )
    p.add_argument(
        "--rho", type=float, default=-0.04,
        help="Dixon-Coles rho parameter (default: -0.04)",
    )
    p.add_argument(
        "--allow-model-total", action="store_true",
        help="Allow 1X2-only fallback when O/U is missing (labelled 'model')",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> RecommendationSheet | None:
    """Main entry point."""
    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Deadline calendar mode
    if args.deadlines:
        deadlines = scrape_deadlines()
        print(format_deadlines(deadlines))
        return None

    # Bonus summary mode (standalone, no network)
    if args.bonus_summary:
        print(generate_bonus_summary())
        return None

    matches: list[MatchOdds] = []
    questions: list[BonusQuestion] = []
    totals_records: list[dict] | None = None

    if args.live:
        # Live runner mode
        sheet = run_matchday(
            matchday=args.md,
            include_bonus=True,
            rho=args.rho,
            allow_model_total=args.allow_model_total,
        )
        print(format_sheet(sheet))
        return sheet
    else:
        # Offline mode
        if args.matchday:
            path = Path(args.matchday)
            logger.info("Offline mode: parsing %s", path)
            html = path.read_text(encoding="utf-8")
            matches = parse_prediction_page(html)

        if args.bonus:
            path = Path(args.bonus)
            logger.info("Offline mode: parsing bonus from %s", path)
            html = path.read_text(encoding="utf-8")
            questions = parse_bonus_page(html)

        if args.totals:
            totals_path = Path(args.totals)
            logger.info("Loading O/U totals from %s", totals_path)
            totals_records = json.loads(totals_path.read_text(encoding="utf-8"))

    if not matches and not questions:
        print("No input provided. Use --matchday <file> or --live.")
        print("Run with --help for usage.")
        sys.exit(1)

    sheet = build_recommendation_sheet(
        matches=matches,
        questions=questions if questions else None,
        totals_records=totals_records,
        rho=args.rho,
        allow_model_total=args.allow_model_total,
    )

    print(format_sheet(sheet))
    return sheet


if __name__ == "__main__":
    main()
