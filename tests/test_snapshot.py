"""Tests for the append-only per-match history recorder (src/snapshot.py)."""

from __future__ import annotations

from src.snapshot import COLUMNS, load_history, record_match


def test_record_creates_row(tmp_path):
    p = tmp_path / "snap.csv"
    df = record_match(1, 4, "Qatar", "Switzerland",
                      kicktipp_1x2=(0.03, 0.03, 0.94),
                      sharp_1x2=(0.06, 0.13, 0.81),
                      ou_over_2_5=0.586, csv_path=p)
    assert len(df) == 1
    assert list(df.columns) == COLUMNS
    row = df.iloc[0]
    assert row["home"] == "Qatar" and row["away"] == "Switzerland"
    assert row["kt_away"] == 0.94 and row["sharp_draw"] == 0.13
    assert row["ou_over_2_5"] == 0.586


def test_partial_update_merges_result(tmp_path):
    p = tmp_path / "snap.csv"
    record_match(1, 4, "Qatar", "Switzerland", kicktipp_1x2=(0.03, 0.03, 0.94), csv_path=p)
    # later, only the result is known - must NOT wipe the earlier odds fields
    df = record_match(1, 4, "Qatar", "Switzerland", result=(1, 1), csv_path=p)
    assert len(df) == 1                       # still one row (upsert, not append)
    row = df.iloc[0]
    assert row["result"] == "1-1"
    assert row["kt_away"] == 0.94             # preserved from the first call


def test_idempotent_same_key(tmp_path):
    p = tmp_path / "snap.csv"
    record_match(1, 4, "Qatar", "Switzerland", kicktipp_1x2=(0.03, 0.03, 0.94), csv_path=p)
    df = record_match(1, 4, "Qatar", "Switzerland", kicktipp_1x2=(0.04, 0.03, 0.93), csv_path=p)
    assert len(df) == 1
    assert df.iloc[0]["kt_home"] == 0.04      # overwritten, not duplicated


def test_distinct_matches_distinct_rows(tmp_path):
    p = tmp_path / "snap.csv"
    record_match(1, 4, "Qatar", "Switzerland", kicktipp_1x2=(0.03, 0.03, 0.94), csv_path=p)
    record_match(1, 5, "Brazil", "Morocco", kicktipp_1x2=(0.84, 0.10, 0.05), csv_path=p)
    record_match(2, 4, "Spain", "Cape Verde", kicktipp_1x2=(0.90, 0.07, 0.03), csv_path=p)
    df = load_history(p)
    assert len(df) == 3
    # keyed by (spieltag, match_index): (1,4),(1,5),(2,4) are all distinct
    assert set(zip(df["spieltag"], df["match_index"])) == {(1, 4), (1, 5), (2, 4)}


def test_backfill_results_upserts_scores(tmp_path):
    from collections import namedtuple

    from src.snapshot import backfill_results
    p = tmp_path / "snap.csv"
    record_match(1, 0, "Qatar", "Switzerland", kicktipp_1x2=(0.03, 0.03, 0.94), csv_path=p)
    Fix = namedtuple("Fix", "index home away result")
    fixtures = [Fix(0, "Qatar", "Switzerland", (1, 1)), Fix(1, "Brazil", "Morocco", None)]
    df = backfill_results(1, fixtures, csv_path=p)
    row = df[(df["spieltag"] == 1) & (df["match_index"] == 0)].iloc[0]
    assert row["result"] == "1-1"            # realised score backfilled onto the odds row
    assert row["kt_away"] == 0.94            # earlier odds preserved (field-level merge)
    assert len(df) == 1                       # unplayed fixture (index 1) not written
