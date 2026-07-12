"""Tests for src/oracle.py consensus aggregation, incl. the empty-filter guard."""

from __future__ import annotations

import pandas as pd

from src import oracle


def _seed(monkeypatch, rows):
    df = pd.DataFrame(rows, columns=oracle._COLS)
    monkeypatch.setattr(oracle, "load_oracle", lambda: df)


class TestConsensus:
    def test_consensus_basic_modal(self, monkeypatch):
        _seed(monkeypatch, [
            ("siteA", "site", 1, 0, 2, 0, "t"),
            ("siteB", "site", 1, 0, 2, 1, "t"),
            ("siteC", "site", 1, 0, 1, 0, "t"),   # all home wins; 2-0 is modal score
        ])
        c = oracle.consensus(1)
        row = c[c["match_index"] == 0].iloc[0]
        assert row["cons_tendency"] == "home"
        assert row["cons_score"] == "2-0"
        assert oracle.consensus_pick(1, 0) == (2, 0)

    def test_consensus_empty_for_unseen_spieltag_does_not_raise(self, monkeypatch):
        # Regression: oracle only has spieltag 1; asking for spieltag 2 used to
        # raise KeyError('spieltag') from sort_values on a column-less empty frame.
        _seed(monkeypatch, [("siteA", "site", 1, 0, 2, 0, "t")])
        c = oracle.consensus(2)
        assert c.empty
        assert list(c.columns) == oracle._CONSENSUS_COLS   # columns preserved
        assert oracle.consensus_pick(2, 0) is None         # graceful, no crash

    def test_consensus_empty_for_unknown_source(self, monkeypatch):
        _seed(monkeypatch, [("siteA", "site", 1, 0, 2, 0, "t")])
        c = oracle.consensus(1, sources=["nonexistent"])
        assert c.empty and list(c.columns) == oracle._CONSENSUS_COLS

    def test_consensus_empty_oracle(self, monkeypatch):
        monkeypatch.setattr(oracle, "load_oracle",
                            lambda: pd.DataFrame(columns=oracle._COLS))
        assert oracle.consensus(1).empty
