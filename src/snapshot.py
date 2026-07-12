"""Append-only per-match history recorder.

Accumulates, per (spieltag, match_index), the full information tuple for every
match: kicktipp 1X2, devigged sharp 1X2, consensus O/U 2.5, and - once played -
the realised score. Writes to ``data/history/snapshots.csv``.

This is the foundation of the second-half edge. It coexists with:
  - ``data/opponents/oracle.csv`` (external source predictions), and
  - ``data/opponents/picks.csv`` (player picks),
so that for each match we can later join the complete tuple
``(kicktipp, sharp, sources, player picks, result)``.

Purpose:
  (i)  build the (kicktipp, sharp, result) set for a future *learned*
       recalibration map (no historical kicktipp odds exist today - this is how
       we start accumulating them); and
  (ii) build the (player pick, consensus) set for the field model.

Upsert is idempotent and field-level: only non-None fields overwrite existing
values, so a pre-match snapshot (odds) and a post-match update (result) merge
into one row.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_HISTORY_CSV = _ROOT / "data" / "history" / "snapshots.csv"

COLUMNS = [
    "spieltag", "match_index", "home", "away",
    "kt_home", "kt_draw", "kt_away",
    "sharp_home", "sharp_draw", "sharp_away",
    "ou_over_2_5", "result", "lead_minutes_to_kickoff", "updated_at",
]
_KEY = ["spieltag", "match_index"]


def load_history(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the accumulated match-history (empty frame with schema if none)."""
    p = Path(csv_path) if csv_path else _HISTORY_CSV
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=COLUMNS)


def record_match(
    spieltag: int,
    match_index: int,
    home: str,
    away: str,
    *,
    kicktipp_1x2: tuple[float, float, float] | None = None,
    sharp_1x2: tuple[float, float, float] | None = None,
    ou_over_2_5: float | None = None,
    result: tuple[int, int] | None = None,
    lead_minutes_to_kickoff: float | None = None,
    csv_path: str | Path | None = None,
    updated_at: str | None = None,
) -> pd.DataFrame:
    """Idempotent field-level upsert of one match snapshot.

    Keyed by (spieltag, match_index). Only the arguments you pass are written;
    omitted fields preserve any previously-recorded value. Returns the full
    updated history frame.

    ``lead_minutes_to_kickoff`` records how far from kickoff (in minutes) the
    odds line was captured - so the snapshot CSV doubles as an odds time-series
    with the capture distance attached (additive; omit to leave unchanged).
    """
    p = Path(csv_path) if csv_path else _HISTORY_CSV
    df = load_history(p)
    # Keep string columns as object dtype so a string (e.g. result "1-1") never
    # clashes with a float64 (all-NaN) dtype inferred on re-read (pandas FutureWarning).
    for _c in ("home", "away", "result", "updated_at"):
        if _c in df.columns:
            df[_c] = df[_c].astype("object")
    updated_at = updated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    row: dict = {
        "spieltag": int(spieltag), "match_index": int(match_index),
        "home": home, "away": away, "updated_at": updated_at,
    }
    if kicktipp_1x2 is not None:
        row["kt_home"], row["kt_draw"], row["kt_away"] = (round(float(x), 6) for x in kicktipp_1x2)
    if sharp_1x2 is not None:
        row["sharp_home"], row["sharp_draw"], row["sharp_away"] = (round(float(x), 6) for x in sharp_1x2)
    if ou_over_2_5 is not None:
        row["ou_over_2_5"] = round(float(ou_over_2_5), 6)
    if result is not None:
        row["result"] = f"{int(result[0])}-{int(result[1])}"
    if lead_minutes_to_kickoff is not None:
        row["lead_minutes_to_kickoff"] = round(float(lead_minutes_to_kickoff), 1)

    if not df.empty:
        mask = (df["spieltag"] == row["spieltag"]) & (df["match_index"] == row["match_index"])
    else:
        mask = pd.Series([], dtype=bool)

    if not df.empty and mask.any():
        idx = df.index[mask][0]
        for k, v in row.items():           # only provided (non-None) fields overwrite
            df.at[idx, k] = v
    else:
        # Build the new row with string columns as object dtype so the very first
        # insert (concat into an empty frame) doesn't trip the pandas all-NA
        # dtype-inference FutureWarning, matching the upsert path's dtypes.
        new_row = pd.DataFrame([row], columns=COLUMNS)
        for _c in ("home", "away", "result", "updated_at"):
            new_row[_c] = new_row[_c].astype("object")
        df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)

    df = df.sort_values(_KEY).reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info("snapshot upsert: spieltag=%s match=%s (%s v %s) -> %d rows",
                spieltag, match_index, home, away, len(df))
    return df


def backfill_results(spieltag, fixtures, csv_path=None):
    """Post-match upsert of realised scores, keyed by (spieltag, fixture.index).

    ``fixtures`` is any iterable of objects exposing .index, .home, .away, .result
    (result = (home_goals, away_goals) or None) - e.g. a Leaderboard's fixtures.
    Idempotent; only played fixtures are written.
    """
    last = None
    for f in fixtures:
        if getattr(f, "result", None) is None:
            continue
        last = record_match(spieltag, f.index, f.home, f.away,
                            result=f.result, csv_path=csv_path)
    return last if last is not None else load_history(csv_path)
