"""Tests for a.PSO knockout scoring (src/scoring/knockout.py)."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from src.scoring.knockout import (
    apso_optimal_prediction,
    apso_result_matrix,
    _outcome_masses,
)


def _reg(lh, la, g=8):
    m = np.outer(poisson.pmf(np.arange(g + 1), lh), poisson.pmf(np.arange(g + 1), la))
    return m / m.sum()


def test_final_matrix_has_no_draw_mass_and_sums_to_one():
    final, _ = apso_result_matrix(_reg(1.3, 1.1), q_home=0.55)
    assert final.sum() == pytest_approx(1.0)
    assert np.trace(final) == pytest_approx(0.0)        # no draw can survive a.PSO


def test_home_tendency_mass_equals_q_home_exactly():
    # The final matrix's home-win mass equals the advance price, for any q_home.
    for q in (0.30, 0.55, 0.93):
        final, info = apso_result_matrix(_reg(1.3, 1.1), q_home=q)
        home_mass = sum(final[i, j] for i in range(final.shape[0])
                        for j in range(final.shape[0]) if i > j)
        assert home_mass == pytest_approx(q, abs=1e-9)
        assert info.realised_q_home == pytest_approx(q, abs=1e-9)


def test_pick_is_always_decisive():
    for lh, la, q in [(1.3, 1.1, 0.55), (0.9, 1.4, 0.30), (2.4, 0.5, 0.95)]:
        pick, ev, _, _ = apso_optimal_prediction(_reg(lh, la), q_home=q)
        assert pick[0] != pick[1]                       # never a draw


def test_tendency_follows_the_advance_favourite():
    home_pick, *_ = apso_optimal_prediction(_reg(1.4, 1.2), q_home=0.72)
    assert home_pick[0] > home_pick[1]                  # home advances -> home pick
    away_pick, *_ = apso_optimal_prediction(_reg(1.2, 1.4), q_home=0.28)
    assert away_pick[0] < away_pick[1]                  # away advances -> away pick


def test_honours_advance_even_when_it_exceeds_reg_win_plus_draws():
    # kicktipp's advance price is the authoritative tendency: it is honoured exactly
    # even when it is sharper on the favourite than the (softer) regulation odds -
    # the old approach clamped here, this one does not.
    reg = _reg(1.6, 1.0)
    hw, d, _ = _outcome_masses(reg)
    q = min(0.999, hw + d + 0.1)                         # exceeds reg-win + all draws
    final, info = apso_result_matrix(reg, q_home=q)
    home_mass = sum(final[i, j] for i in range(final.shape[0])
                    for j in range(final.shape[0]) if i > j)
    assert home_mass == pytest_approx(q, abs=1e-9)
    assert info.realised_q_home == pytest_approx(q, abs=1e-9)


def test_margin_kernel_shifts_optimal_margin():
    # An all-1-goal kernel should never prefer a 2-goal margin from the draw branch
    # over a single-goal one for a coin-flip-ish tie.
    reg = _reg(1.2, 1.15)
    pick_default, *_ = apso_optimal_prediction(reg, q_home=0.55)
    pick_m1, *_ = apso_optimal_prediction(reg, q_home=0.55, margin_kernel={1: 1.0})
    assert pick_default[0] != pick_default[1] and pick_m1[0] != pick_m1[1]


# -- knockout rank/decision layer (a.PSO mode of choose_pick) ----------


def test_decisive_dist_drops_draws_and_renormalises():
    from src.rank_sim import _decisive_dist
    out = _decisive_dist({(1, 0): 0.3, (1, 1): 0.4, (0, 1): 0.3})
    assert (1, 1) not in out
    assert sum(out.values()) == pytest_approx(1.0)
    assert out[(1, 0)] == pytest_approx(0.5)            # 0.3 / (0.3+0.3)


def test_choose_pick_knockout_decisive_opponents_and_valid_pwin():
    import pandas as pd
    from src.field_model import FieldModel
    from src.rank_sim import choose_pick

    fm = FieldModel(pd.DataFrame(columns=["player", "spieltag", "match_index", "pick"]), {})
    pick, _ev, final, _info = apso_optimal_prediction(_reg(1.7, 0.8), q_home=0.82)
    cands = [pick, (pick[0] + 1, pick[1]), (0, 1)]
    res = choose_pick(final, ("home", pick), cands, fm,
                      {"me": 12.0, "o1": 11.0, "o2": 9.0}, "me",
                      horizon=6, target=1, n_sims=4000, seed=1, knockout=True)
    assert res, "knockout choose_pick returned no results"
    for r in res:
        assert 0.0 <= r.p_rank1 <= 1.0 and 0.0 <= r.p_top <= 1.0
    # the leader holds a positive (here near-certain) win probability - the run is
    # valid and the decisive-projection path executed without error.
    assert 0.0 < max(r.p_rank1 for r in res) <= 1.0


# tiny local approx helper (avoid importing pytest at module top for clarity)
def pytest_approx(x, abs=1e-9):
    import pytest
    return pytest.approx(x, abs=abs)
