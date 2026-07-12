"""Opponent pick logger and behavioural profiler.

Strategy context
----------------
Against a field that is "a bit informed but not data-driven", the dominant
early strategy is pure EV-maximisation: harvest your edge and let the sample
size work. The payoff of *deviating* (decorrelation / targeted contrarianism)
comes later, once we can model how opponents actually pick - and that requires
data. This module banks that data from matchday 1.

What it does
------------
1. Logs every opponent's visible picks from the leaderboard each time it's
   run, accumulating an append-only record (picks are hidden until kickoff, so
   coverage grows over time). Raw HTML snapshots are archived for provenance.
2. Profiles each player's behaviour from the accumulated picks: score-line
   distribution, tendency split (home/draw/away), average goals predicted, and
   - once results are known - exact/tendency hit rates.

Phase 2 (once enough data accrues): join picks to the matchday odds we already
scrape to measure *market-tracking* (do they follow the favourite?) and detect
systematic biases, then feed ``P(opponent pick | match)`` into a relative-EV
optimiser instead of the raw-EV one.

Usage
-----
    # Live: scrape leaderboard, archive, upsert into the log, print summary
    python -m src.opponents

    # Offline: ingest a saved leaderboard HTML instead of scraping
    python -m src.opponents --html data/raw/leaderboard.html

    # Just print the behavioural summary from the existing log
    python -m src.opponents --summary-only
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.kicktipp_scrape import COMMUNITY
from src.data.leaderboard import Leaderboard, parse_leaderboard, scrape_leaderboard

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _ROOT / "data" / "opponents"
_SNAP_DIR = _DATA_DIR / "snapshots"
_PICKS_CSV = _DATA_DIR / "picks.csv"

_COLUMNS = ["scraped_at", "spieltag", "player", "match_index", "home", "away",
            "group", "result", "pick", "points"]

# Picks are keyed globally by (player, spieltag, home, away) - match_index
# resets per matchday, so spieltag is required for uniqueness (and the same
# fixture can recur across group + knockout matchdays).
_KEY = ["player", "spieltag", "home", "away"]


# -- Conversion: Leaderboard -> long-format rows ----------------------


def leaderboard_to_rows(lb: Leaderboard, scraped_at: str) -> pd.DataFrame:
    """Flatten a Leaderboard into long-format rows of *visible* picks only."""
    fx_by_idx = {f.index: f for f in lb.fixtures}
    rows = []
    for p in lb.players:
        for idx, (pick, points) in p.picks.items():
            if pick is None:
                continue  # pick hidden / not made -> nothing to log yet
            fx = fx_by_idx.get(idx)
            rows.append({
                "scraped_at": scraped_at,
                "spieltag": lb.spieltag,
                "player": p.name,
                "match_index": idx,
                "home": fx.home if fx else "",
                "away": fx.away if fx else "",
                "group": fx.group if fx else "",
                "result": (f"{fx.result[0]}-{fx.result[1]}"
                           if fx and fx.result else ""),
                "pick": f"{pick[0]}-{pick[1]}",
                "points": points,
            })
    return pd.DataFrame(rows, columns=_COLUMNS)


# -- Persistence ------------------------------------------------------


def load_picks() -> pd.DataFrame:
    """Load the accumulated pick log (empty frame if none yet)."""
    if _PICKS_CSV.exists():
        return pd.read_csv(_PICKS_CSV)
    return pd.DataFrame(columns=_COLUMNS)


def _upsert(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge new rows, keyed by (player, spieltag, home, away).

    A visible pick is immutable once seen, so we keep one row per
    (player, matchday, match). We prefer the row carrying the most information
    (a known result / non-zero points), which means later snapshots enrich
    earlier ones rather than duplicating them.
    """
    combined = pd.concat([existing, new], ignore_index=True)
    # Sort so the "richest" row per key lands last, then keep last.
    combined["_has_result"] = combined["result"].fillna("").astype(str).ne("")
    combined = combined.sort_values(
        _KEY + ["_has_result", "points", "scraped_at"]
    )
    before = len(existing.drop_duplicates(_KEY)) if not existing.empty else 0
    deduped = combined.drop_duplicates(_KEY, keep="last")
    deduped = deduped.drop(columns="_has_result").sort_values(
        ["spieltag", "match_index", "player"]
    ).reset_index(drop=True)
    after = len(deduped)
    return deduped, after - before


def log_snapshot(
    lb: Leaderboard,
    raw_html: str | None = None,
    scraped_at: str | None = None,
) -> int:
    """Archive the snapshot and upsert its visible picks into the log.

    Returns the number of *newly seen* (player, match) picks added.
    """
    scraped_at = scraped_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _SNAP_DIR.mkdir(parents=True, exist_ok=True)

    if raw_html is not None:
        stamp = scraped_at.replace(":", "").replace("-", "").replace("+0000", "Z")
        md = f"md{lb.spieltag}" if lb.spieltag is not None else "mdX"
        snap_path = _SNAP_DIR / f"leaderboard_{md}_{stamp}.html"
        snap_path.write_text(raw_html, encoding="utf-8")
        logger.info("Archived snapshot -> %s", snap_path.name)

    new = leaderboard_to_rows(lb, scraped_at)
    existing = load_picks()
    merged, n_new = _upsert(existing, new)
    merged.to_csv(_PICKS_CSV, index=False)
    logger.info("Pick log: %d total rows (%d newly seen) -> %s",
                len(merged), n_new, _PICKS_CSV)
    return n_new


def update_log(
    spieltags: range | list[int],
    *,
    tippsaison_id: int | None = None,
    community: str = COMMUNITY,
) -> int:
    """Scrape each matchday's leaderboard and upsert newly-visible picks.

    Re-scraping all matchdays each run is intentional: picks unlock at each
    match's kickoff, so iterating every matchday captures them as they appear.
    Matchdays with no fixtures yet are skipped. Returns total newly-seen picks.
    """
    from src.data.leaderboard import (
        DEFAULT_TIPPSAISON_ID, fetch_leaderboard_html, parse_leaderboard,
    )
    if tippsaison_id is None:
        tippsaison_id = DEFAULT_TIPPSAISON_ID

    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total_new = 0
    for sti in spieltags:
        html = fetch_leaderboard_html(
            community, spieltag=sti, tippsaison_id=tippsaison_id)
        lb = parse_leaderboard(html, spieltag=sti)
        if not lb.fixtures:
            logger.info("Matchday %d: no fixtures, skipping", sti)
            continue
        n_new = log_snapshot(lb, raw_html=html, scraped_at=scraped_at)
        n_visible = sum(
            1 for p in lb.players for v in p.picks.values() if v[0] is not None)
        logger.info("Matchday %d: %d fixtures, %d visible picks (%d new)",
                    sti, len(lb.fixtures), n_visible, n_new)
        total_new += n_new
    return total_new


# -- Behavioural profiling --------------------------------------------


def _tendency(pick: str) -> str:
    h, a = (int(x) for x in pick.split("-"))
    return "home" if h > a else ("draw" if h == a else "away")


def profile_players(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player behavioural summary from accumulated picks."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["tend"] = df["pick"].map(_tendency)
    df["pick_goals"] = df["pick"].map(lambda s: sum(int(x) for x in s.split("-")))
    df["scored"] = df["result"].fillna("").astype(str).ne("")
    df["exact_hit"] = df.apply(
        lambda r: r["scored"] and r["pick"] == r["result"], axis=1)
    df["tend_hit"] = df.apply(
        lambda r: r["scored"] and _tendency(r["pick"]) == _tendency(r["result"]),
        axis=1)

    out = []
    for name, g in df.groupby("player"):
        scored = g[g["scored"]]
        top_pick = g["pick"].mode()
        out.append({
            "player": name,
            "n_picks": len(g),
            "fav_score": top_pick.iloc[0] if len(top_pick) else "",
            "home%": round(100 * (g["tend"] == "home").mean(), 0),
            "draw%": round(100 * (g["tend"] == "draw").mean(), 0),
            "away%": round(100 * (g["tend"] == "away").mean(), 0),
            "avg_goals": round(g["pick_goals"].mean(), 2),
            "n_scored": len(scored),
            "exact%": round(100 * scored["exact_hit"].mean(), 0) if len(scored) else None,
            "tend%": round(100 * scored["tend_hit"].mean(), 0) if len(scored) else None,
        })
    prof = pd.DataFrame(out).sort_values("player").reset_index(drop=True)
    return prof


def format_summary(df: pd.DataFrame) -> str:
    prof = profile_players(df)
    if prof.empty:
        return "No picks logged yet."
    key_cols = [c for c in ("spieltag", "home", "away") if c in df.columns]
    n_matches = df[key_cols].drop_duplicates().shape[0]
    mds = sorted(df["spieltag"].dropna().unique().tolist()) if "spieltag" in df else []
    md_str = f" (matchdays {', '.join(str(int(m)) for m in mds)})" if mds else ""
    header = (f"Opponent pick log - {len(df)} picks across "
              f"{df['player'].nunique()} players, {n_matches} matches with "
              f"visible picks{md_str}.\n")
    note = ("(Early data - distributions stabilise as more matches kick off. "
            "Market-tracking analysis lands once picks are joined to odds.)\n")
    return (header + note + "\n" + prof.to_string(index=False)
            + "\n\n" + readiness_report(df))


# -- Discriminating-match tracker -------------------------------------
#
# Blowouts carry almost no opponent information: when one outcome dominates,
# everyone picks the favourite, so picks don't diverge. The matches that
# actually *separate* players - and therefore feed an opponent model - are the
# ones where picks spread out. We measure that spread directly from the picks
# (realised discrimination), which is more honest than guessing from odds: a
# match is "discriminating" when the field genuinely disagreed on the outcome.

# Rough readiness milestones, in *discriminating* matches seen:
_READY_FIELD = 6     # field-level tendencies become usable
_READY_RULES = 10    # rule-like per-player habits become identifiable
_READY_PRECISE = 20  # per-player proportions get reasonably precise

# A match discriminates when the modal tendency holds < this share of picks
# (i.e. at least ~30% of the field deviated from the consensus outcome).
_DISCRIMINATE_MAX_MODAL_SHARE = 0.70


def _norm_entropy(counts: np.ndarray, k: int) -> float:
    """Entropy of a count vector, normalised to [0, 1] by log(k)."""
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(k)) if len(p) > 1 else 0.0


def match_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    """Per-match measure of how much the field's picks diverged."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["tend"] = df["pick"].map(_tendency)
    rows = []
    for (st, home, away), g in df.groupby(["spieltag", "home", "away"]):
        n = len(g)
        tcounts = g["tend"].value_counts()
        modal_share = tcounts.iloc[0] / n
        res = g["result"].fillna("").astype(str)
        res = next((r for r in res if r), "")
        rows.append({
            "spieltag": int(st) if pd.notna(st) else None,
            "match": f"{home}-{away}",
            "result": res,
            "n": n,
            "modal_tend": tcounts.index[0],
            "modal_share": round(modal_share, 2),
            "tend_entropy": round(_norm_entropy(tcounts.values, 3), 2),
            "distinct_scores": g["pick"].nunique(),
            "discriminating": bool(modal_share < _DISCRIMINATE_MAX_MODAL_SHARE),
        })
    return (pd.DataFrame(rows)
            .sort_values(["spieltag", "match"]).reset_index(drop=True))


def readiness_report(df: pd.DataFrame) -> str:
    """Summarise opponent-model readiness by *discriminating* matches seen."""
    md = match_discrimination(df)
    if md.empty:
        return "Discriminating matches: none yet."
    n_total = len(md)
    n_disc = int(md["discriminating"].sum())

    def bar(target):
        pct = min(100, int(round(100 * n_disc / target)))
        return f"{n_disc}/{target} ({pct}%)"

    lines = [
        f"-- Opponent-model readiness --",
        f"Discriminating matches seen: {n_disc} of {n_total} played "
        f"(blowouts carry ~no opponent info, so only these count).",
        f"  - field-level tendencies   : {bar(_READY_FIELD)}",
        f"  - per-player rule detection : {bar(_READY_RULES)}",
        f"  - per-player precise model  : {bar(_READY_PRECISE)}",
        "",
        md.to_string(index=False),
    ]
    return "\n".join(lines)


def classify_by_odds(p_home: float, p_draw: float, p_away: float,
                     blowout_max_prob: float = 0.65) -> str:
    """Forecast a match's discriminating power from its 1X2 (a-priori).

    'blowout' when one outcome dominates (field will converge on the
    favourite -> low info); 'close' otherwise (likely to separate players).
    Use to flag upcoming matches before picks are visible.
    """
    return "blowout" if max(p_home, p_draw, p_away) >= blowout_max_prob else "close"


# -- CLI ---------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log and profile opponent picks.")
    p.add_argument("--matchday", type=int, default=None,
                   help="Scrape only this matchday (default: all 1..--max-matchday)")
    p.add_argument("--max-matchday", type=int, default=15,
                   help="Highest matchday to iterate when scraping all (default: 15)")
    p.add_argument("--season-id", type=int, default=None,
                   help="tippsaisonId override (default: current season constant)")
    p.add_argument("--html", type=str, default=None,
                   help="Ingest a saved leaderboard HTML instead of scraping live "
                        "(stamped with --matchday, default 1)")
    p.add_argument("--summary-only", action="store_true",
                   help="Print the summary from the existing log; don't scrape")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.summary_only:
        print(format_summary(load_picks()))
        return

    if args.html:
        raw = Path(args.html).read_text(encoding="utf-8")
        lb = parse_leaderboard(raw, spieltag=args.matchday or 1)
        n_new = log_snapshot(lb, raw_html=raw)
    else:
        spieltags = ([args.matchday] if args.matchday is not None
                     else range(1, args.max_matchday + 1))
        n_new = update_log(spieltags, tippsaison_id=args.season_id)

    print(f"Logged: {n_new} newly seen pick(s).\n")
    print(format_summary(load_picks()))


if __name__ == "__main__":
    main()
