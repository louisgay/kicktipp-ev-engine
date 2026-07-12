"""Tests for src/calibration.py - metric helpers + the curated-odds join."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.calibration import (
    collect_sharp_predictions,
    draw_reliability,
    log_loss_1x2,
    multiclass_brier,
)

# The curated-odds join reads regenerable historical data that ships gitignored.
# Regenerate with:  python -m src.data.download && python -m src.data.clean
_RESULTS_CLEAN = Path(__file__).resolve().parents[1] / "data" / "processed" / "results_clean.csv"


class TestMetrics:
    def test_brier_perfect_is_zero(self):
        probs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        outcomes = np.array([0, 1, 2])
        assert multiclass_brier(probs, outcomes) == 0.0

    def test_brier_worst_is_two(self):
        # all mass on the wrong class -> per-match Brier = 1^2 + 1^2 = 2
        probs = np.array([[0, 0, 1.0]])
        outcomes = np.array([0])
        assert abs(multiclass_brier(probs, outcomes) - 2.0) < 1e-9

    def test_logloss_perfect_near_zero(self):
        probs = np.array([[1.0, 0, 0]])
        outcomes = np.array([0])
        assert log_loss_1x2(probs, outcomes) < 1e-6

    def test_logloss_penalises_confident_wrong(self):
        confident_wrong = log_loss_1x2(np.array([[0.001, 0.001, 0.998]]), np.array([0]))
        hedged = log_loss_1x2(np.array([[0.33, 0.34, 0.33]]), np.array([0]))
        assert confident_wrong > hedged

    def test_draw_reliability_columns(self):
        p_draw = np.array([0.10, 0.26, 0.27, 0.33])
        is_draw = np.array([0, 1, 0, 1])
        rel = draw_reliability(p_draw, is_draw)
        assert set(rel.columns) == {"bin", "n", "mean_predicted", "realised_freq"}
        assert rel["n"].sum() == 4


@pytest.mark.skipif(
    not _RESULTS_CLEAN.exists(),
    reason="requires regenerable data/processed/results_clean.csv "
           "(run: python -m src.data.download && python -m src.data.clean)",
)
class TestCuratedJoin:
    def test_some_matches_join_and_metrics_sane(self):
        probs, outcomes, n_total, n_matched = collect_sharp_predictions("shin")
        assert n_total >= 115                      # WC2022 (64) + Euro2024 (51)
        assert n_matched > 50                      # most should join to results_clean
        assert probs.shape == (n_matched, 3)
        # devigged probabilities sum to ~1 per match
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert 0.0 <= multiclass_brier(probs, outcomes) <= 2.0
        assert log_loss_1x2(probs, outcomes) > 0.0
