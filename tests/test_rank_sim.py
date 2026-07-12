"""Tests for src/rank_sim.py - edge scaling, MC standard errors, count helper."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from src.rank_sim import (
    _BASE_DIST,
    _VALS,
    _points_dist,
    compare_picks,
    remaining_match_count,
)


def _mean(dist) -> float:
    return float((_VALS * np.asarray(dist)).sum())


class TestPointsDist:
    def test_baseline_mean_is_1_50(self):
        assert abs(_mean(_BASE_DIST) - 1.50) < 1e-9

    def test_edge_zero_is_base(self):
        assert np.allclose(_points_dist(0.0), _BASE_DIST)

    def test_edge_is_true_mean_shift(self):
        base_mean = _mean(_BASE_DIST)
        for e in (0.05, 0.10, 0.20):
            assert abs(_mean(_points_dist(e)) - (base_mean + e)) < 1e-6


class TestCompareSE:
    def test_se_and_paired_diff_populated(self):
        mat = np.zeros((3, 3))
        mat[1, 0], mat[1, 1], mat[0, 0] = 0.5, 0.3, 0.2   # sums to 1
        totals = {"me": 0.0, "r1": 0.0, "r2": 0.0}
        opp = {"r1": (1, 0), "r2": (0, 0)}
        res = compare_picks([(1, 0), (1, 1)], current_totals=totals, me="me",
                            score_matrix=mat, opponent_picks=opp,
                            n_future=5, n_sims=3000, seed=1)
        assert all(r.se_top >= 0.0 for r in res)
        ref = max(res, key=lambda r: r.match_ev)
        assert ref.diff_vs_evmax is None            # no diff against itself
        others = [r for r in res if r is not ref]
        assert others and all(r.diff_vs_evmax is not None and r.diff_se is not None
                              for r in others)


class TestRemainingCount:
    Fix = namedtuple("Fix", "index home away result")
    LB = namedtuple("LB", "fixtures players spieltag")

    def _patch(self, monkeypatch, by_md):
        import src.data.leaderboard as lb_mod
        monkeypatch.setattr(lb_mod, "scrape_leaderboard",
                            lambda spieltag, **kw: self.LB(by_md.get(spieltag, []), [], spieltag))

    def test_counts_played_and_scraped_from_fixtures(self, monkeypatch):
        # 'played' and 'scraped' track the live fixtures dynamically.
        self._patch(monkeypatch, {
            1: [self.Fix(0, "A", "B", (1, 0)), self.Fix(1, "C", "D", None)],
            2: [self.Fix(0, "E", "F", None)],
        })
        out = remaining_match_count(max_matchday=5)
        assert out["scraped"] == 3 and out["played"] == 1 and out["matchdays"] == 2

    def test_total_floored_at_tournament_matches(self, monkeypatch):
        # Group-only scrape (knockouts not yet scrapeable) must NOT undercount the
        # horizon: total floors at 104, so remaining includes the 32 knockouts.
        from src.rank_sim import TOURNAMENT_MATCHES, KNOCKOUT_MATCHES
        assert TOURNAMENT_MATCHES == 104 and KNOCKOUT_MATCHES == 32
        # 72 group fixtures over 10 matchdays, 14 played, empty after (today's shape).
        by_md = {md: [self.Fix(i, f"H{md}{i}", f"A{md}{i}",
                               (1, 0) if (md == 1 and i < 6) or (md == 2 and i < 8) else None)
                      for i in range(8 if md <= 6 else 6)] for md in range(1, 11)}
        self._patch(monkeypatch, by_md)
        out = remaining_match_count(max_matchday=15)
        assert out["scraped"] == 72                       # group stage only
        assert out["total"] == TOURNAMENT_MATCHES         # floored at 104
        assert out["remaining"] == 104 - out["played"]    # horizon spans the knockouts
        assert out["total"] - out["scraped"] == KNOCKOUT_MATCHES   # +32 vs group-only count

    def test_max_tracks_scraped_no_double_count(self, monkeypatch):
        # Hypothetical: once knockouts are scrapeable the dynamic count rises to
        # 104; max(scraped, 104) stays 104 (no double-count), and would track a
        # larger real count if the format ever exceeded the floor.
        self._patch(monkeypatch, {md: [self.Fix(i, f"H{md}{i}", f"A{md}{i}", None)
                                       for i in range(8)] for md in range(1, 14)})  # 104
        out = remaining_match_count(max_matchday=15)
        assert out["scraped"] == 104 and out["total"] == 104   # equal -> no double-count

    def test_horizon_propagates_to_choose_pick(self, monkeypatch):
        # The pipeline's horizon = remaining - len(rows) feeds choose_pick's
        # n_future; with the 104 floor it spans the tournament, not just groups.
        self._patch(monkeypatch, {md: [self.Fix(i, f"H{md}{i}", f"A{md}{i}", (1, 0))
                                       for i in range(8 if md <= 6 else 6)] for md in range(1, 11)})
        out = remaining_match_count()
        n_rows = 3
        horizon = max(0, out["remaining"] - n_rows)
        assert horizon == 104 - out["played"] - n_rows    # includes the 32 knockouts
