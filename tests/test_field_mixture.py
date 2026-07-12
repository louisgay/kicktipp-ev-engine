"""Tests for the gated per-player source mixture (field_model Part A)."""

from __future__ import annotations

import pandas as pd

from src.field_model import FieldModel


def _build():
    # Tracker: 8 discriminating matches (sources A=home, B=away DISAGREE), always
    # picks A's score. Newbie: 2 such matches only (below the gate).
    rows = []
    for i in range(8):
        rows.append(("Tracker", 1, i, "1-0"))      # follows source A
    for i in range(2):
        rows.append(("Newbie", 1, i, "1-0"))
    consensus = {(1, i): (0, 1) for i in range(8)}  # consensus = away (B-ish)
    consensus[(1, 99)] = (1, 0)                      # the test match
    source_preds = {(1, i): {"A": (1, 0), "B": (0, 1)} for i in range(8)}   # disagree
    source_preds[(1, 99)] = {"A": (2, 1), "B": (0, 2)}                       # test match
    source_match = {"Tracker": {"A": 0.9, "B": 0.1}, "Newbie": {"A": 0.6, "B": 0.4}}
    return FieldModel(pd.DataFrame(rows, columns=["player", "spieltag", "match_index", "pick"]),
                      consensus, follow_k=5, exact_k=3, min_discriminating=8,
                      source_preds=source_preds, source_match=source_match)


class TestGate:
    def test_discriminating_counts(self):
        fm = _build()
        counts = fm.discriminating_counts()
        assert counts["Tracker"] == 8 and counts["Newbie"] == 2

    def test_below_gate_falls_back_to_tier2(self):
        fm = _build()
        # Newbie (2 < 8) must use the Tier-2 consensus model, NOT the mixture.
        got = fm.pick_distribution("Newbie", 1, 99)
        tier2 = fm.pick_distribution_for_consensus("Newbie", "home", (1, 0))
        assert got == tier2

    def test_above_gate_uses_mixture(self):
        fm = _build()
        # Tracker (8 >= 8) uses the mixture: weight concentrates on source A,
        # so it predicts A's pick (2-1) for the test match - NOT the consensus (1-0).
        d = fm.pick_distribution("Tracker", 1, 99)
        assert max(d, key=d.get) == (2, 1) and d[(2, 1)] > 0.6
        assert d != fm.pick_distribution_for_consensus("Tracker", "home", (1, 0))

    def test_source_weights_track_a(self):
        fm = _build()
        w = fm.source_weights("Tracker")
        assert w["A"] > w["B"] and abs(sum(w.values()) - 1.0) < 1e-9
