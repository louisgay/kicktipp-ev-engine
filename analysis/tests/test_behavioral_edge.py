"""Tests for analysis/behavioral_edge.py - the generator-independent edge anchor.

Relationship / sane-range assertions only (no hardcoded magic numbers): the leak is
non-negative by construction, a draw on a favourite leaks more than the same draw on
an even low-scoring match, and the realized stats + regression return finite outputs.
tests/ may import analysis (the one-way analysis->src contract only forbids src->analysis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.behavioral_edge import (
    build_match_evals,
    expected_leak,
    leak_points_regression,
    paired_vs_self,
)
from src.snapshot import COLUMNS


def _snap(spieltag, mi, home, away, kt, *, sharp=(None, None, None), ou=None, result=None):
    return {
        "spieltag": spieltag, "match_index": mi, "home": home, "away": away,
        "kt_home": kt[0], "kt_draw": kt[1], "kt_away": kt[2],
        "sharp_home": sharp[0], "sharp_draw": sharp[1], "sharp_away": sharp[2],
        "ou_over_2_5": ou, "result": result, "lead_minutes_to_kickoff": 60,
        "updated_at": "2026-01-01T00:00:00+00:00"}


def _snaps(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def _picks(rows):
    cols = ["scraped_at", "spieltag", "player", "match_index", "home", "away",
            "group", "result", "pick", "points"]
    return pd.DataFrame([dict(zip(cols, ("t", st, pl, mi, "H", "A", "G", res, pk, None)))
                         for (pl, st, mi, pk, res) in rows], columns=cols)


# A strong home favourite and a near-even low-scoring match.
_FAV = _snap(1, 0, "H", "A", (0.80, 0.13, 0.07), ou=0.45)        # decisive favourite
_EVEN = _snap(1, 1, "H", "A", (0.30, 0.42, 0.28), ou=0.30)       # even, low-scoring


class TestLeakNonNegative:
    def test_all_leaks_nonnegative(self):
        evals = build_match_evals(_snaps([_FAV, _EVEN]))
        picks = _picks([("P1", 1, 0, "0-0", None), ("P1", 1, 1, "1-1", None),
                        ("P2", 1, 0, "2-0", None), ("P2", 1, 1, "1-0", None)])
        leaks = expected_leak(picks, evals)
        assert all(p.leak_total >= -1e-9 for p in leaks.values())
        assert all(p.leak_from_draws >= -1e-9 and p.leak_from_other >= -1e-9
                   for p in leaks.values())

    def test_evmax_pick_has_zero_leak(self):
        evals = build_match_evals(_snaps([_FAV]))
        ev = evals[(1, 0)]
        picks = _picks([("EVmaxer", 1, 0, f"{ev.evmax_pick[0]}-{ev.evmax_pick[1]}", None)])
        leaks = expected_leak(picks, evals)
        assert leaks["EVmaxer"].leak_total < 1e-9


class TestDumbVsSmartDraw:
    def test_draw_on_favourite_leaks_more_than_draw_on_even_match(self):
        # SAME player, SAME draw scoreline (1-1): costly on the favourite, ~free on
        # the even low-scoring match where a draw is near the EV-max pick.
        evals = build_match_evals(_snaps([_FAV, _EVEN]))
        picks = _picks([("Drawer", 1, 0, "1-1", None), ("Drawer", 1, 1, "1-1", None)])
        # per-match leaks
        from src.scoring.kicktipp import expected_points
        leak_fav = evals[(1, 0)].evmax_ev - expected_points((1, 1), evals[(1, 0)].matrix)
        leak_even = evals[(1, 1)].evmax_ev - expected_points((1, 1), evals[(1, 1)].matrix)
        assert leak_fav > leak_even
        # and the favourite is flagged as such; the even match is not
        assert abs(evals[(1, 0)].supremacy) > abs(evals[(1, 1)].supremacy)


class TestRealizedStatsRunFinite:
    def _setup(self):
        # varied realized results so the paired differential has nonzero variance
        results = ["1-0", "2-0", "1-1", "0-1", "2-1", "0-0"]
        snaps = _snaps([_snap(1, i, "H", "A", (0.55, 0.25, 0.20), ou=0.5, result=results[i])
                        for i in range(6)])
        rows = []
        for i in range(6):
            rows.append(("self", 1, i, "1-0", results[i]))   # EV-max-ish home win
            rows.append(("Drawer", 1, i, "1-1", results[i]))   # always draws, leaks
        return build_match_evals(snaps), _picks(rows)

    def test_paired_outputs_finite_and_shaped(self):
        evals, picks = self._setup()
        res = paired_vs_self(picks, evals)
        assert res, "expected at least one opponent"
        for r in res:
            assert r.n >= 2 and np.isfinite(r.mu_d) and np.isfinite(r.sigma_d)
            # t at the long horizon is at least the t at current N (more breadth)
            assert abs(r.t_proj_104) >= abs(r.t_now) - 1e-9

    def test_regression_runs_and_is_finite(self):
        evals, picks = self._setup()
        leaks = expected_leak(picks, evals)
        reg = leak_points_regression(picks, evals, leaks)
        assert reg.n == len({p for p in leaks}) and np.isfinite(reg.slope)
        # more expected leak should not predict MORE realized points (slope ≤ ~0)
        assert reg.slope <= 0.5


class TestIdempotent:
    def test_same_snapshot_same_numbers(self):
        snaps = _snaps([_FAV, _EVEN])
        picks = _picks([("P", 1, 0, "1-1", None), ("P", 1, 1, "2-0", None)])
        a = expected_leak(picks, build_match_evals(snaps))
        b = expected_leak(picks, build_match_evals(snaps))
        assert a["P"].leak_total == b["P"].leak_total
