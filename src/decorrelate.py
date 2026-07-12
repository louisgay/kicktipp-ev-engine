"""Decorrelation helper - differentiate from the field at minimal EV cost.

When you are *tied* with other players, pure EV-maximisation is not the whole
objective. Picking a near-equal-probability alternative scoreline does not raise
your expected points, but it decorrelates your exact-score outcome from the
field's. In a multi-way tie that raises P(finish strictly ahead) for a tiny EV
cost. This module quantifies that trade for a whole matchday.

For each match it reports:
- the EV-optimal ("max-EV") prediction,
- the cheapest *decorrelating* alternative - the highest-EV score that differs
  from the max-EV pick while (by default) keeping the same tendency, so the
  2-point tendency floor is preserved,
- the EV cost (ΔEV) of making that swap.

Pipeline (same as ``src.recommend``):
    Kicktipp 1X2  ->  de-vig  ->  reconstruct (λ_h, λ_a) under O/U 2.5
                  ->  Dixon-Coles matrix  ->  EV-optimal under pool scoring.

Usage
-----
    # Live scrape + O/U from a saved Odds-API cache file
    python -m src.decorrelate --live --md 1 \
        --odds-cache data/raw/odds_cache/6a2a23108b57ff68.json

    # Offline from a saved matchday HTML
    python -m src.decorrelate --matchday data/raw/live_matchday1.html \
        --odds-cache data/raw/odds_cache/6a2a23108b57ff68.json

    # Restrict the report to specific home teams (e.g. only unplayed matches)
    python -m src.decorrelate --live --md 1 --odds-cache <file> \
        --only Canada,USA,Qatar,Brazil,Haiti,Australia
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.clean import normalise_team
from src.data.kicktipp_scrape import MatchOdds, parse_prediction_page, scrape_matchday
from src.models.odds_model import KicktippOddsModel
from src.scoring.kicktipp import expected_points, optimal_prediction

logger = logging.getLogger(__name__)


# -- Result container --------------------------------------------------


@dataclass
class DecorrelationPick:
    """Max-EV vs decorrelated pick for one match."""

    home_team: str
    away_team: str
    prob_home: float
    prob_draw: float
    prob_away: float
    max_ev_pred: tuple[int, int]
    max_ev: float
    max_ev_p_exact: float
    decorr_pred: tuple[int, int]
    decorr_ev: float
    decorr_p_exact: float
    same_tendency: bool

    @property
    def delta_ev(self) -> float:
        """EV cost of choosing the decorrelated pick (<= 0)."""
        return self.decorr_ev - self.max_ev


# -- Core helpers ------------------------------------------------------


def _tendency(pred: tuple[int, int]) -> int:
    """+1 home win, 0 draw, -1 away win."""
    return (pred[0] > pred[1]) - (pred[0] < pred[1])


def consensus_over_under_25(odds_cache_path: str | Path) -> dict[tuple[str, str], list[float]]:
    """Extract consensus Over/Under 2.5 odds from a saved Odds-API cache file.

    Averages [over, under] decimal odds across all bookmakers quoting the 2.5
    line. Keyed by (normalised home, normalised away).
    """
    cache = json.loads(Path(odds_cache_path).read_text())
    events = cache.get("data", cache)
    if isinstance(events, dict):  # historical-endpoint shape
        events = events.get("data", [])

    lookup: dict[tuple[str, str], list[float]] = {}
    for ev in events:
        overs, unders = [], []
        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m.get("key") != "totals":
                    continue
                over = under = None
                for oc in m.get("outcomes", []):
                    if oc.get("point") == 2.5 and oc.get("name") == "Over":
                        over = oc["price"]
                    elif oc.get("point") == 2.5 and oc.get("name") == "Under":
                        under = oc["price"]
                if over and under:
                    overs.append(over)
                    unders.append(under)
        if overs and unders:
            key = (normalise_team(ev["home_team"]), normalise_team(ev["away_team"]))
            lookup[key] = [float(np.mean(overs)), float(np.mean(unders))]
    logger.info("Loaded consensus O/U 2.5 for %d matches from cache", len(lookup))
    return lookup


def cheapest_decorrelation(
    prob_matrix: np.ndarray,
    max_ev_pred: tuple[int, int],
    *,
    keep_tendency: bool = True,
    max_pred: int = 5,
) -> tuple[tuple[int, int], float, float]:
    """Highest-EV scoreline that differs from the max-EV pick.

    With ``keep_tendency=True`` (default) the alternative must share the
    max-EV pick's tendency, preserving the 2-pt tendency floor.

    Returns (decorr_pred, decorr_ev, decorr_p_exact).
    """
    target_sign = _tendency(max_ev_pred)
    best_alt: tuple[int, int] | None = None
    best_alt_ev = -1.0
    for i in range(max_pred + 1):
        for j in range(max_pred + 1):
            if (i, j) == max_ev_pred:
                continue
            if keep_tendency and _tendency((i, j)) != target_sign:
                continue
            ev = expected_points((i, j), prob_matrix)
            if ev > best_alt_ev:
                best_alt_ev = ev
                best_alt = (i, j)
    if best_alt is None:  # degenerate fallback: allow any tendency
        return cheapest_decorrelation(
            prob_matrix, max_ev_pred, keep_tendency=False, max_pred=max_pred
        )
    return best_alt, best_alt_ev, float(prob_matrix[best_alt])


# -- Matchday analysis -------------------------------------------------


def analyse_matchday(
    matches: list[MatchOdds],
    over_under: dict[tuple[str, str], list[float]],
    *,
    rho: float = -0.04,
    max_goals: int = 8,
    only: set[str] | None = None,
    keep_tendency: bool = True,
) -> list[DecorrelationPick]:
    """Compute max-EV and decorrelated picks for each match."""
    model = KicktippOddsModel(rho=rho, max_goals=max_goals, require_ou=False)
    model.load_kicktipp_odds(matches)

    totals_records = []
    for m in matches:
        key = (normalise_team(m.home_team), normalise_team(m.away_team))
        if key in over_under:
            totals_records.append({
                "home_team": m.home_team,
                "away_team": m.away_team,
                "totals_odds": over_under[key],
                "totals_line": 2.5,
            })
    model.load_totals(totals_records)

    picks: list[DecorrelationPick] = []
    for m in matches:
        if only and m.home_team not in only:
            continue
        h, a = normalise_team(m.home_team), normalise_team(m.away_team)
        matrix = model.predict_score_matrix(h, a)
        best_pred, best_ev = optimal_prediction(matrix)
        decorr_pred, decorr_ev, decorr_pe = cheapest_decorrelation(
            matrix, best_pred, keep_tendency=keep_tendency
        )
        picks.append(DecorrelationPick(
            home_team=m.home_team,
            away_team=m.away_team,
            prob_home=m.prob_home,
            prob_draw=m.prob_draw,
            prob_away=m.prob_away,
            max_ev_pred=best_pred,
            max_ev=best_ev,
            max_ev_p_exact=float(matrix[best_pred]),
            decorr_pred=decorr_pred,
            decorr_ev=decorr_ev,
            decorr_p_exact=decorr_pe,
            same_tendency=_tendency(best_pred) == _tendency(decorr_pred),
        ))
    return picks


def format_picks(picks: list[DecorrelationPick]) -> str:
    """Render the analysis as a table + EV-cost summary."""
    lines = [
        f"{'Match':<34} {'H/D/A':>14}  {'maxEV':>6} {'pick':>5} {'Pex':>5}  "
        f"{'decorr':>6} {'pick':>5} {'ΔEV':>6}",
        "-" * 96,
    ]
    total_max = total_decorr = 0.0
    for p in picks:
        total_max += p.max_ev
        total_decorr += p.decorr_ev
        mx = f"{p.max_ev_pred[0]}-{p.max_ev_pred[1]}"
        dc = f"{p.decorr_pred[0]}-{p.decorr_pred[1]}"
        flag = "" if p.same_tendency else " (TEND!)"
        lines.append(
            f"{p.home_team + ' v ' + p.away_team:<34} "
            f"{p.prob_home:4.0%}/{p.prob_draw:3.0%}/{p.prob_away:3.0%}  "
            f"{p.max_ev:6.2f} {mx:>5} {p.max_ev_p_exact:5.0%}  "
            f"{p.decorr_ev:6.2f} {dc:>5} {p.delta_ev:6.2f}{flag}"
        )
    lines.append("-" * 96)
    lines.append(
        f"{'TOTAL':<34} {'':>14}  {total_max:6.2f}{'':>17}{total_decorr:6.2f}"
        f"{'':>6} {total_decorr - total_max:6.2f}"
    )
    lines.append(
        f"\nDecorrelating the full card costs {total_max - total_decorr:.2f} EV "
        f"({(total_max - total_decorr) / total_max * 100:.1f}% of max-EV) "
        f"while differing from the field on every exact score."
    )
    lines.append("\nDecorrelated card: " + " · ".join(
        f"{p.home_team} {p.decorr_pred[0]}-{p.decorr_pred[1]}" for p in picks
    ))
    return "\n".join(lines)


# -- CLI ---------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decorrelation analysis for a matchday.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true",
                     help="Scrape Kicktipp live (needs KICKTIPP_SESSION)")
    src.add_argument("--matchday", type=str,
                     help="Path to a saved matchday HTML file (offline)")
    p.add_argument("--md", type=int, default=1,
                   help="Matchday index for --live (default: 1)")
    p.add_argument("--odds-cache", type=str, required=True,
                   help="Path to an Odds-API cache JSON for consensus O/U 2.5")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated home teams to restrict the report to")
    p.add_argument("--rho", type=float, default=-0.04, help="Dixon-Coles rho")
    p.add_argument("--allow-tendency-flip", action="store_true",
                   help="Allow the decorrelated pick to change tendency "
                        "(default: keep tendency, preserving the 2-pt floor)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> list[DecorrelationPick]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.live:
        matches = scrape_matchday(args.md)
    else:
        html = Path(args.matchday).read_text(encoding="utf-8")
        matches = parse_prediction_page(html)
    if not matches:
        raise SystemExit("No matches found.")

    over_under = consensus_over_under_25(args.odds_cache)
    only = set(s.strip() for s in args.only.split(",")) if args.only else None

    picks = analyse_matchday(
        matches, over_under,
        rho=args.rho, only=only,
        keep_tendency=not args.allow_tendency_flip,
    )
    print(format_picks(picks))
    return picks


if __name__ == "__main__":
    main()
