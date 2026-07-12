"""Prediction-oracle: external tip sources, consensus, and field-tracking.

Hypothesis under test: the (German-speaking) opponents anchor their picks
on German football "Tipp/Prognose" media rather than on the Kicktipp odds or
their own analysis. If true, those sources' predicted scorelines are a
*pre-kickoff proxy for where the field will cluster* - the exact input a
relative-EV strategy needs.

This module:
1. Stores per-match predictions from many sources (German tip sites, their
   individual tipsters, the French source, an AI/KI tip, ...) in
   ``data/opponents/oracle.csv``, keyed by (source, spieltag, match_index) so it
   joins cleanly to the opponent pick log (``picks.csv``) without team-name
   matching.
2. Consensus: aggregates sources into a per-match consensus tendency + modal
   scoreline.
3. Field-tracking: joins predictions to the opponents' actual picks and
   scores *which source best predicts the pool* - overall and per player.

Collection note: robustly scraping commercial betting sites in pure Python is
brittle (bot-walls, JS, GDPR gates). v1 is agent-assisted - predictions are
extracted with the web tools and written here via ``add_predictions``. Each
source is a pluggable record, so a future per-site scraper can feed the same
store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data" / "opponents"
_ORACLE_CSV = _DATA / "oracle.csv"
_PICKS_CSV = _DATA / "picks.csv"

_COLS = ["source", "kind", "spieltag", "match_index",
         "pred_home", "pred_away", "collected_at"]
_KEY = ["source", "spieltag", "match_index"]

_CONSENSUS_COLS = ["spieltag", "match_index", "n_sources", "cons_tendency",
                   "tend_agreement", "cons_score", "score_agreement"]

# Whose row in picks.csv is "us" (excluded from opponent-tracking metrics).
SELF = "self"


def _tend(h: int, a: int) -> str:
    return "home" if h > a else ("draw" if h == a else "away")


# -- Storage ----------------------------------------------------------


def load_oracle() -> pd.DataFrame:
    if _ORACLE_CSV.exists():
        return pd.read_csv(_ORACLE_CSV)
    return pd.DataFrame(columns=_COLS)


def add_predictions(
    source: str,
    spieltag: int,
    preds: list[tuple[int, int, int]],
    *,
    kind: str = "site",
    collected_at: str | None = None,
) -> int:
    """Upsert predictions for one source.

    Parameters
    ----------
    source : label, e.g. "wettanbieter.de" or "wettfreunde:KI" or "eurosport.fr"
    spieltag : matchday index
    preds : list of (match_index, pred_home, pred_away)
    kind : "site" | "tipster" | "ai" | "model" - provenance category
    """
    collected_at = collected_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{
        "source": source, "kind": kind, "spieltag": spieltag,
        "match_index": mi, "pred_home": ph, "pred_away": pa,
        "collected_at": collected_at,
    } for (mi, ph, pa) in preds]
    new = pd.DataFrame(rows, columns=_COLS)
    combined = pd.concat([load_oracle(), new], ignore_index=True)
    combined = combined.drop_duplicates(_KEY, keep="last").sort_values(
        ["spieltag", "match_index", "source"]).reset_index(drop=True)
    _DATA.mkdir(parents=True, exist_ok=True)
    combined.to_csv(_ORACLE_CSV, index=False)
    return len(new)


# -- Consensus across sources -----------------------------------------


def consensus(spieltag: int | None = None, sources: list[str] | None = None) -> pd.DataFrame:
    """Per-match consensus tendency + modal scoreline across sources."""
    df = load_oracle()
    if df.empty:
        return df
    if spieltag is not None:
        df = df[df["spieltag"] == spieltag]
    if sources is not None:
        df = df[df["source"].isin(sources)]
    if df.empty:           # no oracle rows for this spieltag/source filter
        return pd.DataFrame(columns=_CONSENSUS_COLS)
    df = df.copy()
    df["tend"] = [_tend(h, a) for h, a in zip(df["pred_home"], df["pred_away"])]
    df["score"] = df["pred_home"].astype(str) + "-" + df["pred_away"].astype(str)

    out = []
    for (st, mi), g in df.groupby(["spieltag", "match_index"]):
        tmode = g["tend"].value_counts()
        smode = g["score"].value_counts()
        out.append({
            "spieltag": st, "match_index": mi, "n_sources": len(g),
            "cons_tendency": tmode.index[0],
            "tend_agreement": round(tmode.iloc[0] / len(g), 2),
            "cons_score": smode.index[0],
            "score_agreement": round(smode.iloc[0] / len(g), 2),
        })
    return pd.DataFrame(out).sort_values(["spieltag", "match_index"]).reset_index(drop=True)


# -- Predicting opponent picks for an upcoming match ------------------


def consensus_pick(spieltag: int, match_index: int) -> tuple[int, int] | None:
    """The field's predicted modal scoreline for a match (None if no data)."""
    c = consensus(spieltag)
    if c.empty:
        return None
    row = c[c["match_index"] == match_index]
    if row.empty:
        return None
    a, b = str(row.iloc[0]["cons_score"]).split("-")
    return (int(a), int(b))


def field_picks_consensus(
    spieltag: int, match_index: int, opponents: list[str],
) -> dict[str, tuple[int, int]]:
    """Model every opponent as picking the German consensus scoreline.

    Reflects the validated hypothesis that the (German-speaking) field anchors
    on German tip media. Returns {opponent: (home, away)}, or {} if no consensus
    is available for that match. A richer per-player model (each opponent -> the
    pick of the source they best track, via ``correlate_per_player``) is the next
    refinement once enough discriminating matches accrue.
    """
    pick = consensus_pick(spieltag, match_index)
    return {p: pick for p in opponents} if pick else {}


# -- Field-tracking: which source predicts the opponents? --------------


def _load_field(include_self: bool = False) -> pd.DataFrame:
    if not _PICKS_CSV.exists():
        return pd.DataFrame()
    f = pd.read_csv(_PICKS_CSV)
    if not include_self:
        f = f[f["player"] != SELF]
    f = f.copy()
    f[["ph", "pa"]] = f["pick"].str.split("-", expand=True).astype(int)
    f["ptend"] = [_tend(h, a) for h, a in zip(f["ph"], f["pa"])]
    return f


def correlate_sources(include_self: bool = False) -> pd.DataFrame:
    """Per-source: how well it predicts opponents' actual picks.

    For every (source-prediction, opponent-pick) pair on the same match,
    record whether tendency and exact score matched. Aggregated per source.
    """
    oracle = load_oracle()
    field = _load_field(include_self)
    if oracle.empty or field.empty:
        return pd.DataFrame()

    oracle = oracle.copy()
    oracle["otend"] = [_tend(h, a) for h, a in zip(oracle["pred_home"], oracle["pred_away"])]
    j = field.merge(oracle, on=["spieltag", "match_index"], suffixes=("", "_o"))
    j["tend_hit"] = j["ptend"] == j["otend"]
    j["exact_hit"] = (j["ph"] == j["pred_home"]) & (j["pa"] == j["pred_away"])

    out = []
    for src, g in j.groupby("source"):
        out.append({
            "source": src,
            "kind": g["kind"].iloc[0],
            "n_matches": g["match_index"].nunique(),
            "n_compares": len(g),
            "tendency_match%": round(100 * g["tend_hit"].mean(), 0),
            "exact_match%": round(100 * g["exact_hit"].mean(), 0),
        })
    return (pd.DataFrame(out)
            .sort_values(["tendency_match%", "exact_match%"], ascending=False)
            .reset_index(drop=True))


def correlate_per_player(include_self: bool = True) -> pd.DataFrame:
    """Player × source tendency-match matrix - who follows which source."""
    oracle = load_oracle()
    field = _load_field(include_self=True)
    if oracle.empty or field.empty:
        return pd.DataFrame()
    oracle = oracle.copy()
    oracle["otend"] = [_tend(h, a) for h, a in zip(oracle["pred_home"], oracle["pred_away"])]
    j = field.merge(oracle, on=["spieltag", "match_index"], suffixes=("", "_o"))
    j["tend_hit"] = j["ptend"] == j["otend"]
    m = (j.groupby(["player", "source"])["tend_hit"].mean().mul(100).round(0)
         .unstack(fill_value=float("nan")))
    return m


def format_report() -> str:
    lines = ["=== ORACLE: source -> field-tracking ===\n"]
    cs = correlate_sources(include_self=False)
    if cs.empty:
        return "No oracle/field overlap yet - add predictions and log opponent picks."
    lines.append("How well each source predicts the OPPONENTS' picks "
                 "(across played, covered matches):")
    lines.append(cs.to_string(index=False))
    lines.append("\nConsensus per match (all sources):")
    lines.append(consensus().to_string(index=False))
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(format_report())
