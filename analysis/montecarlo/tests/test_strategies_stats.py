"""Tests for the strategy behaviours, the stats layer, and the isolation invariant
(analysis -> src is one-way; src must never import analysis)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.field_model import FieldModel

from analysis.montecarlo.engine import SimConfig, simulate
from analysis.montecarlo.strategies import (
    SIGMA_MATCH,
    Z_CHASE,
    make_strategy,
    offtendency_evmax,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def fm():
    return FieldModel.from_disk()


# -- strategies ----------------------------------------------------------


def test_sigma_match_value():
    # SD of rank_sim's baseline per-match points dist (_VALS/_BASE_DIST) ≈ 1.53
    assert SIGMA_MATCH == pytest.approx(1.526, abs=0.005)
    assert Z_CHASE == 1.0


def test_rank_relative_reduces_to_evmax_when_threshold_huge(fm):
    a = simulate(SimConfig(n_sims=4000, seed=0, regime="counterfactual", strategy="evmax"),
                 field_model=fm)
    b = simulate(SimConfig(n_sims=4000, seed=0, regime="counterfactual",
                           strategy="rank_relative", z_chase=999), field_model=fm)
    assert np.array_equal(a.finals, b.finals)          # never chase -> identical


def test_contrarian_is_dominated(fm):
    """Always-fade is structurally -EV: its max final loses to EV-max's P5; P(top3)=0."""
    ev = simulate(SimConfig(n_sims=8000, seed=0, regime="counterfactual", strategy="evmax"),
                  field_model=fm)
    co = simulate(SimConfig(n_sims=8000, seed=0, regime="counterfactual", strategy="contrarian"),
                  field_model=fm)
    assert co.finals[co.me_index].max() < np.percentile(ev.finals[ev.me_index], 5)
    assert co.p_top == 0.0


def test_offtendency_pick_is_off_consensus():
    # a strong home favourite -> the contrarian pick must NOT be a home tendency
    from src.odds.reconstruct import reconstruct_matrix
    mat = reconstruct_matrix(0.75, 0.18, 0.07, 0.5)
    pick = offtendency_evmax(mat, "home")
    assert (pick[0] > pick[1]) is False                # not a home win


def test_make_strategy_unknown_raises():
    with pytest.raises(ValueError):
        make_strategy("nope")


# -- stats layer ---------------------------------------------------------


def test_percentile_bands_monotone(fm):
    from analysis.montecarlo.stats import percentile_bands, BAND_PCTLS
    res = simulate(SimConfig(n_sims=6000, seed=0, regime="counterfactual", track="me"),
                   field_model=fm)
    b = percentile_bands(res.me_cum)
    last = [b[p][-1] for p in BAND_PCTLS]
    assert last == sorted(last)                        # P5 <= P10 <= ... <= P95


def test_rivals_ordered(fm):
    from analysis.montecarlo.stats import rival_reference_paths
    res = simulate(SimConfig(n_sims=6000, seed=0, regime="counterfactual", track="all"),
                   field_model=fm)
    riv = rival_reference_paths(res.cum, 3, res.me_index)
    assert (riv["leader"][-1] > riv["top3_boundary"][-1]
            > riv["field_median"][-1] > riv["bottom3_boundary"][-1])


# -- skill / luck decomposition + detectability --------------------------


def test_variance_decomposition_is_an_identity(fm):
    from analysis.montecarlo.stats import skill_luck_decomposition
    d = skill_luck_decomposition(n_sims=8000, seed=0, field_model=fm, bootstrap=80,
                                 noise_reps=500)
    assert d["var_skill"] + d["var_luck"] == pytest.approx(d["var_obs"], rel=1e-9)
    assert 0.0 < d["luck_share"] < 1.0
    # Mauboussin null luck (horizon·σ²) and Var(luck) agree to within a factor ~1.5
    assert 0.5 < d["var_luck"] / d["mauboussin_null_luck"] < 1.5


def test_decomposition_reproducible(fm):
    from analysis.montecarlo.stats import skill_luck_decomposition
    kw = dict(n_sims=4000, seed=0, field_model=fm, bootstrap=50, noise_reps=300)
    a, b = skill_luck_decomposition(**kw), skill_luck_decomposition(**kw)
    assert a["luck_share"] == b["luck_share"]
    assert a["detect"]["IR_tournament"] == b["detect"]["IR_tournament"]
    assert a["noise_floor"]["obs_follow_sd"] == b["noise_floor"]["obs_follow_sd"]


# -- isolation invariant: src must NEVER import analysis -----------------


def test_src_does_not_import_analysis():
    pat = re.compile(r"^\s*(?:from|import)\s+analysis", re.M)
    offenders = [p for p in (REPO / "src").rglob("*.py") if pat.search(p.read_text())]
    assert not offenders, f"src must not import analysis: {offenders}"
