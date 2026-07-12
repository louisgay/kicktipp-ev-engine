"""Integrity tests for the Monte-Carlo fan-chart engine.

Property/invariant checks that keep the engine honest and data-agnostic: seed
reproducibility, a valid rank distribution that sums to one, ranks that form a
valid permutation, and the realised path ending exactly at each player's banked
points total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.field_model import FieldModel

from analysis.montecarlo.engine import SELF, SimConfig, simulate


@pytest.fixture(scope="module")
def fm():
    return FieldModel.from_disk()


# -- reproducibility ----------------------------------------------------


def test_seed_reproducible(fm):
    cfg = SimConfig(n_sims=3000, seed=7, regime="counterfactual")
    a, b = simulate(cfg, field_model=fm), simulate(cfg, field_model=fm)
    assert np.array_equal(a.finals, b.finals)
    assert np.array_equal(a.me_cum, b.me_cum)


def test_percentile_bands_reproducible(fm):
    from analysis.montecarlo.stats import assert_percentiles_reproducible
    assert assert_percentiles_reproducible(
        SimConfig(n_sims=4000, seed=3, regime="counterfactual"), field_model=fm)


# -- distributional sanity ----------------------------------------------


def test_rank_distribution_sums_to_one(fm):
    from analysis.montecarlo.stats import rank_distribution
    res = simulate(SimConfig(n_sims=8000, seed=0, regime="counterfactual"), field_model=fm)
    rd = rank_distribution(res.me_rank, len(res.players))
    assert len(rd) == len(res.players)
    assert abs(sum(rd) - 1.0) < 1e-9
    assert abs(sum(rd[:3]) - res.p_top) < 1e-9          # P(top3) == sum of first 3


def test_ranks_are_valid_permutation(fm):
    res = simulate(SimConfig(n_sims=2000, seed=1, regime="counterfactual"), field_model=fm)
    assert res.me_rank.min() >= 1 and res.me_rank.max() <= len(res.players)


# -- realised path matches the banked results ----------------------------


def test_realised_path_matches_banked(fm):
    from analysis.montecarlo.stats import realised_path
    picks = pd.read_csv("data/opponents/picks.csv")
    banked = picks[picks["points"].notna()].groupby("player")["points"].sum()
    for player in (SELF, "p10"):
        idx, cum = realised_path(player)
        assert cum[-1] == pytest.approx(float(banked[player]))    # ends at banked total
        assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))  # monotone non-decreasing
    # hybrid engine start reproduces the banked totals deterministically
    res = simulate(SimConfig(n_sims=500, seed=0, regime="hybrid"), field_model=fm)
    assert res.realised_cum[-1] == pytest.approx(float(banked[SELF]))
