"""Tests for the field-model future legs + relative-EV optimiser (rank_sim B+C)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.field_model import FieldModel
from src.rank_sim import (
    _future_leg_lambdas,
    _points_vec,
    _sim_future_correlated,
    choose_pick,
    simulate_rank_multi,
)
from src.scoring.kicktipp import points


def _matrix(lh, la, g=8):
    gh, ga = poisson.pmf(np.arange(g + 1), lh), poisson.pmf(np.arange(g + 1), la)
    m = np.outer(gh, ga)
    return m / m.sum()


def _empty_fm(consensus_lookup=None):
    return FieldModel(pd.DataFrame(columns=["player", "spieltag", "match_index", "pick"]),
                      consensus_lookup or {})


def _scattered_fm(cons_score=(1, 0)):
    """Symmetric, NON-degenerate field: pop_follow ≈ 0.6 over a consensus pick,
    so queried players (no own rows) all share the same 3-way-scattered pick
    distribution (60% consensus / 20% draw / 20% away) - symmetric, no edge."""
    cons = {(9, i): cons_score for i in range(10)}
    s = f"{cons_score[0]}-{cons_score[1]}"
    rows = [(f"seed{k}", 9, i, s if i < 6 else ("1-1" if i < 8 else "0-1"))
            for k in range(5) for i in range(10)]
    return FieldModel(pd.DataFrame(rows, columns=["player", "spieltag", "match_index", "pick"]),
                      cons, follow_k=0.01, exact_k=0.01)


class TestPointsVec:
    def test_matches_scalar_scoring(self):
        rng = np.random.default_rng(0)
        pa, pb = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
        ah, aa = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
        vec = _points_vec(pa, pb, ah, aa)
        scal = np.array([points((int(a), int(b)), (int(c), int(d)))
                         for a, b, c, d in zip(pa, pb, ah, aa)])
        assert np.array_equal(vec, scal)

    def test_draw_has_no_goal_diff_tier(self):
        # predicted 2-2 (draw), actual 1-1 (draw, GD=0 both) -> tendency 2, NOT 3
        assert int(_points_vec(np.array([2]), np.array([2]),
                               np.array([1]), np.array([1]))[0]) == 2


class TestCorrelation:
    def test_shared_outcome_correlates_opponents(self):
        # Two consensus-followers (n=0) pick identically -> perfectly correlated.
        tot = _sim_future_correlated(["o1", "o2"], "absent", 12, _empty_fm(),
                                     np.random.default_rng(1), 5000)
        assert np.corrcoef(tot["o1"], tot["o2"])[0, 1] > 0.9

    def test_me_has_no_systematic_future_edge(self):
        # Edge-free future legs: "me" is sampled like any opponent, so there is NO
        # systematic forward mean advantage (the old endogenous-edge is removed).
        fm = _scattered_fm()
        tot = _sim_future_correlated(["me", "opp"], "me", 40, fm,
                                     np.random.default_rng(2), 8000)
        # symmetric -> means within Monte-Carlo noise (<< the ~60-pt per-sim scale)
        assert abs(tot["me"].mean() - tot["opp"].mean()) < 0.5
        # shared outcome per leg -> still positively correlated (idiosyncratic
        # pick scatter lowers it vs the point-mass case; >0.9 is tested separately)
        assert np.corrcoef(tot["me"], tot["opp"])[0, 1] > 0.05


class TestUnchangedIID:
    def test_field_none_equals_legacy(self):
        mat = _matrix(1.5, 1.0)
        mm = [{"score_matrix": mat, "my_pick": (1, 0), "opponent_picks": {"a": (1, 0), "b": (0, 0)}}]
        totals = {"me": 0.0, "a": 0.0, "b": 0.0}
        r1 = simulate_rank_multi(totals, "me", mm, n_generic_future=20, n_sims=3000, seed=7)
        r2 = simulate_rank_multi(totals, "me", mm, n_generic_future=20, n_sims=3000, seed=7,
                                 field_model=None)
        assert r1.p_top == r2.p_top and r1.median_rank == r2.median_rank


class TestContrarianLever:
    def test_lever_is_exact_score_not_draw_on_lopsided(self):
        # Realistic SCATTERED field (not a monolith): on a lopsided home favourite
        # the rank-optimal lever is an alternative HOME exact score, NOT a draw -
        # a draw throws away the tendency floor. (With the edge-free future legs,
        # this holds via the decision-match decorrelation, not a fantasy edge.)
        fm = _scattered_fm((2, 0))
        mat = _matrix(2.0, 0.6)               # lopsided home favourite
        cons = ("home", (2, 0))
        cands = [(2, 0), (1, 0), (2, 1), (3, 0), (1, 1), (0, 0)]
        totals = {"me": 7.0, "o1": 8.0, "o2": 8.0, "o3": 8.0, "o4": 9.0,
                  "o5": 9.0, "o6": 6.0, "o7": 6.0}              # we trail modestly
        res = choose_pick(mat, cons, cands, fm, totals, "me",
                          horizon=12, n_sims=30000, seed=3)
        best = res[0]
        assert best.pick[0] > best.pick[1]                     # rank-optimal = a HOME win
        by_pick = {r.pick: r for r in res}
        # both draws rank strictly below the crowd's EV-max pick (2-0)
        assert by_pick[(1, 1)].p_top < by_pick[(2, 0)].p_top
        assert by_pick[(0, 0)].p_top < by_pick[(2, 0)].p_top
        assert all(r.match_ev > 0 for r in res)


class TestDesaturation:
    """The fix's calibration proof: no compounding edge -> P(top3) reflects
    standings + honest variance, not a saturating 1.0."""

    def test_p_top_from_level_is_baseline(self):
        # ALL players equal totals, symmetric scatter, long horizon -> each of 11
        # players ~ 3/11 = 0.27 top-3 (GATE 1), NOT 1.0.
        fm = _scattered_fm((1, 0))
        players = [f"q{i}" for i in range(11)]
        totals = {p: 10.0 for p in players}
        opp = {p: (1, 0) for p in players if p != "q0"}
        r = simulate_rank_multi(
            totals, "q0",
            [{"score_matrix": _matrix(1.6, 1.0), "my_pick": (1, 0), "opponent_picks": opp}],
            n_generic_future=56, field_model=fm, n_sims=20000, target=3, seed=3)
        assert 0.20 <= r.p_top <= 0.34          # ~0.27 baseline, decisively not saturated
        assert r.p_top < 0.5

    def test_trailing_player_below_baseline(self):
        # From a deficit, P(top3) is sensible and < 1 (no fantasy catch-up edge).
        fm = _scattered_fm((1, 0))
        totals = {"me": 4.0, "a": 14.0, "b": 13.0, "c": 12.0, "d": 11.0,
                  "e": 11.0, "f": 10.0, "g": 9.0, "h": 8.0, "i": 7.0, "j": 6.0}
        opp = {p: (1, 0) for p in totals if p != "me"}
        r = simulate_rank_multi(
            totals, "me",
            [{"score_matrix": _matrix(1.6, 1.0), "my_pick": (1, 0), "opponent_picks": opp}],
            n_generic_future=56, field_model=fm, n_sims=20000, target=3, seed=4)
        assert r.p_top < 0.5                    # trailing badly -> well below baseline, not 1.0


class TestEmpiricalFutureLegs:
    """The future-leg generator bootstraps real odds-derived match structures, so
    the favourite SIDE varies and the supremacy spread is realistic - replacing
    the always-strong-home-favourite synthetic stand-in. Assertions are
    RELATIONSHIPS / broad ranges, never the empirical values themselves."""

    def test_bootstrap_is_heterogeneous_not_all_home(self):
        # uses the committed snapshots' empirical (s, t) atoms
        s = (lambda L: L[:, 0] - L[:, 1])(_future_leg_lambdas(800, np.random.default_rng(0)))
        assert (s > 0).any() and (s < 0).any(), "favourite side must vary (not all home)"
        assert s.std() > 0.5, "supremacy must have a real spread (synthetic was ~0.21)"
        assert (np.abs(s) < 0.5).any(), "some near-toss-up legs must appear"

    def test_fallback_is_home_favoured_synthetic(self, monkeypatch):
        import src.rank_sim as R
        monkeypatch.setattr(R, "_empirical_match_structures", lambda: None)
        s = (lambda L: L[:, 0] - L[:, 1])(_future_leg_lambdas(200, np.random.default_rng(0)))
        assert (s > 0).all()      # graceful fallback = the old always-home stand-in

    def test_explicit_fav_dist_override_bypasses_bootstrap(self):
        # an explicit dist override must take the synthetic path, not the empirical one
        s = (lambda L: L[:, 0] - L[:, 1])(
            _future_leg_lambdas(200, np.random.default_rng(0), fav_dist=(0.5, 0.95)))
        assert (s > 0).all()


class TestHeterogeneousFieldNotResaturated:
    """After the market-modal re-anchor the field is HETEROGENEOUS (favourite-
    followers vs faders). A follower picks up a style edge on the all-favourite
    future legs, so P(top3) rises well above the symmetric ~0.27 baseline - but
    it must NOT pin at 1.0 (that would be the old saturation bug regressing). It
    stays below the health report's 0.95 saturation sentinel."""

    def _fader_heavy_fm(self):
        # half the field fades the home favourite to draws, half follows it.
        cons = {(9, i): (1, 0) for i in range(10)}
        rows = []
        for k in range(4):                                   # followers
            rows += [(f"follow{k}", 9, i, "1-0") for i in range(10)]
        for k in range(4):                                   # faders -> draws
            rows += [(f"fade{k}", 9, i, "1-1" if i < 7 else "1-0") for i in range(10)]
        return FieldModel(pd.DataFrame(rows, columns=["player", "spieltag", "match_index", "pick"]),
                          cons, follow_k=0.5, exact_k=0.5)

    def test_follower_ptop_elevated_but_not_saturated(self):
        fm = self._fader_heavy_fm()
        totals = {"me": 10.0, **{f"follow{k}": 10.0 for k in range(4)},
                  **{f"fade{k}": 10.0 for k in range(4)}}
        opp = {p: (1, 0) for p in totals if p != "me"}
        r = simulate_rank_multi(
            totals, "me",
            [{"score_matrix": _matrix(1.6, 1.0), "my_pick": (1, 0), "opponent_picks": opp}],
            n_generic_future=56, field_model=fm, n_sims=20000, target=3, seed=11)
        assert r.p_top < 0.95, f"P(top3)={r.p_top} re-saturated (sentinel is 0.95)"
        assert 0.0 < r.p_top < 1.0


class TestDiscrimination:
    """choose_pick separates picks on contested matches, not on chalk (GATE 3)."""

    @staticmethod
    def _spread_and_se(res):
        ptops = [r.p_top for r in res]
        ses = [r.diff_se for r in res if r.diff_se is not None]
        return max(ptops) - min(ptops), (max(ses) if ses else 0.0)

    def _totals(self):
        return {"me": 8.0, "o1": 9.0, "o2": 9.0, "o3": 7.0, "o4": 7.0,
                "o5": 10.0, "o6": 6.0, "o7": 8.0}

    def test_contested_discriminates(self):
        fm = _scattered_fm((1, 1))
        res = choose_pick(_matrix(1.25, 1.05), ("home", (1, 1)),
                          [(1, 0), (1, 1), (0, 1), (2, 1)], fm, self._totals(), "me",
                          horizon=10, n_sims=30000, seed=5)
        spread, se = self._spread_and_se(res)
        assert spread > 2 * se and spread > 0.02      # real rank signal

    def test_chalk_does_not_discriminate(self):
        fm = _scattered_fm((4, 0))
        res = choose_pick(_matrix(4.0, 0.22), ("home", (4, 0)),
                          [(4, 0), (3, 0), (2, 0), (5, 0)], fm, self._totals(), "me",
                          horizon=10, n_sims=30000, seed=5)
        spread, _ = self._spread_and_se(res)
        assert spread < 0.02                          # below the practical floor
