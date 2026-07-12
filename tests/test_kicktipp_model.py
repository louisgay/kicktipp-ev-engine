"""Tests for KicktippOddsModel (T2: decoupled reconstruction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.kicktipp_scrape import MatchOdds, parse_prediction_page
from src.models.odds_model import KicktippOddsModel
from src.odds.reconstruct import _1x2_from_matrix, _over_under_from_matrix
from src.scoring.kicktipp import optimal_prediction

_FIXTURES = Path(__file__).parent / "fixtures"


# -- Helper: build a model from the HTML fixture ----------------------


def _make_model(with_totals: bool = False, require_ou: bool = False) -> KicktippOddsModel:
    html = (_FIXTURES / "predict_matchday1.html").read_text(encoding="utf-8")
    matches = parse_prediction_page(html)
    model = KicktippOddsModel(rho=-0.04, max_goals=8, require_ou=require_ou)
    model.load_kicktipp_odds(matches)

    if with_totals:
        # Simulated O/U 2.5 data (bookmaker odds, will be devigged)
        totals = [
            {"home_team": "Mexico", "away_team": "South Africa",
             "totals_odds": [2.10, 1.80], "totals_line": 2.5},
            {"home_team": "Germany", "away_team": "Japan",
             "totals_odds": [1.95, 1.95], "totals_line": 2.5},
            {"home_team": "Brazil", "away_team": "Serbia",
             "totals_odds": [2.00, 1.90], "totals_line": 2.5},
        ]
        model.load_totals(totals)
    return model


# -- KicktippOddsModel loading ----------------------------------------


class TestKicktippOddsModelLoad:
    def test_loads_all_matches(self):
        model = _make_model()
        assert len(model._probs_1x2) == 8

    def test_teams_normalised(self):
        model = _make_model()
        assert "Mexico" in model.teams_
        assert "South Africa" in model.teams_

    def test_probs_sum_to_one(self):
        model = _make_model()
        for key, (ph, pd, pa) in model._probs_1x2.items():
            assert abs(ph + pd + pa - 1.0) < 1e-4, f"{key}: {ph+pd+pa}"

    def test_totals_loaded(self):
        model = _make_model(with_totals=True)
        assert len(model._p_over) == 3
        # Devigged p_over should be between 0 and 1
        for key, p in model._p_over.items():
            assert 0 < p < 1, f"{key}: p_over={p}"


# -- Score matrix reconstruction --------------------------------------


class TestKicktippReconstruction:
    def test_matrix_shape(self):
        model = _make_model()
        mat = model.predict_score_matrix("Mexico", "South Africa")
        assert mat.shape == (9, 9)  # max_goals=8 -> 9×9

    def test_matrix_sums_to_one(self):
        model = _make_model()
        mat = model.predict_score_matrix("Mexico", "South Africa")
        assert abs(mat.sum() - 1.0) < 1e-6

    def test_matrix_non_negative(self):
        model = _make_model()
        mat = model.predict_score_matrix("Mexico", "South Africa")
        assert (mat >= 0).all()

    def test_1x2_round_trip(self):
        """Reconstructed matrix should reproduce kicktipp 1X2 probs."""
        model = _make_model()
        for key, (ph, pd, pa) in model._probs_1x2.items():
            home, away = key
            mat = model.predict_score_matrix(home, away)
            rph, rpd, rpa = _1x2_from_matrix(mat)
            # Allow ≤ 2% absolute error per outcome
            assert abs(rph - ph) < 0.02, (
                f"{home} vs {away}: P(H) {ph:.3f} -> {rph:.3f}"
            )
            assert abs(rpd - pd) < 0.02, (
                f"{home} vs {away}: P(D) {pd:.3f} -> {rpd:.3f}"
            )
            assert abs(rpa - pa) < 0.02, (
                f"{home} vs {away}: P(A) {pa:.3f} -> {rpa:.3f}"
            )

    def test_heavy_favourite_matrix(self):
        """USA vs Fiji: P(H)≈0.87 -> home win should dominate the matrix."""
        model = _make_model()
        mat = model.predict_score_matrix("United States", "Fiji")
        ph, pd, pa = _1x2_from_matrix(mat)
        assert ph > 0.80
        assert pa < 0.10

    def test_with_totals_constraint(self):
        """When O/U is provided, P(over 2.5) should be closer to target."""
        model_no_ou = _make_model(with_totals=False)
        model_ou = _make_model(with_totals=True)

        # Germany vs Japan has O/U data
        mat_no = model_no_ou.predict_score_matrix("Germany", "Japan")
        mat_ou = model_ou.predict_score_matrix("Germany", "Japan")

        # Both should be valid matrices
        assert abs(mat_no.sum() - 1.0) < 1e-6
        assert abs(mat_ou.sum() - 1.0) < 1e-6

        # O/U-constrained model should produce P(over) close to the
        # devigged target (≈ 0.50 for even O/U odds 1.95/1.95)
        p_over_ou, _ = _over_under_from_matrix(mat_ou, 2.5)
        assert abs(p_over_ou - 0.50) < 0.05

    def test_missing_match_raises(self):
        model = _make_model()
        with pytest.raises(KeyError):
            model.predict_score_matrix("Atlantis", "Narnia")


# -- End-to-end: kicktipp -> matrix -> optimal prediction --------------


class TestEndToEnd:
    def test_optimal_prediction_from_kicktipp(self):
        """Full pipeline: kicktipp odds -> matrix -> optimal score."""
        model = _make_model()
        mat = model.predict_score_matrix("Mexico", "South Africa")
        (pred_h, pred_a), ev = optimal_prediction(mat)
        # Prediction should be integers
        assert isinstance(pred_h, (int, np.integer))
        assert isinstance(pred_a, (int, np.integer))
        # Mexico is favourite -> home win expected
        assert pred_h > pred_a
        # EV should be positive
        assert ev > 0

    def test_all_matches_produce_predictions(self):
        model = _make_model()
        for key in model._probs_1x2:
            home, away = key
            mat = model.predict_score_matrix(home, away)
            (pred_h, pred_a), ev = optimal_prediction(mat)
            assert ev > 0, f"{home} vs {away}: EV={ev}"

    def test_favourite_gets_more_goals(self):
        """Heavy favourites should get predicted more goals."""
        model = _make_model()
        # USA vs Fiji: USA massive favourite (normalised to "United States")
        mat = model.predict_score_matrix("United States", "Fiji")
        (pred_h, pred_a), _ = optimal_prediction(mat)
        assert pred_h >= pred_a


# -- Team name normalisation in model ---------------------------------


class TestTeamNormalisation:
    def test_normalise_on_load(self):
        """German kicktipp names should be normalised."""
        # Create MatchOdds with a German name
        m = MatchOdds(
            match_id="test", datetime_str="01/01/26 20:00",
            home_team="Bosnien-Herzegowina", away_team="Türkiye",
            odds_home=3.50, odds_draw=3.40, odds_away=2.10,
            prob_home=0.285, prob_draw=0.294, prob_away=0.421,
            overround=1.000,
        )
        model = KicktippOddsModel(require_ou=False)
        model.load_kicktipp_odds([m])
        # Should be stored under canonical names
        assert ("Bosnia and Herzegovina", "Turkey") in model._probs_1x2
        assert "Bosnia and Herzegovina" in model.teams_
        assert "Turkey" in model.teams_

    def test_predict_with_canonical_names(self):
        m = MatchOdds(
            match_id="test", datetime_str="01/01/26 20:00",
            home_team="Türkiye", away_team="USA",
            odds_home=2.50, odds_draw=3.30, odds_away=2.90,
            prob_home=0.370, prob_draw=0.280, prob_away=0.350,
            overround=1.000,
        )
        model = KicktippOddsModel(require_ou=False)
        model.load_kicktipp_odds([m])
        # Must use canonical names for lookup
        mat = model.predict_score_matrix("Turkey", "United States")
        assert mat.shape == (9, 9)
        assert abs(mat.sum() - 1.0) < 1e-6


# -- require_ou enforcement -------------------------------------------


class TestRequireOU:
    def test_require_ou_raises_when_missing(self):
        """require_ou=True should raise ValueError when O/U is missing."""
        model = _make_model(require_ou=True)
        with pytest.raises(ValueError, match="O/U data required but missing"):
            model.predict_score_matrix("Mexico", "South Africa")

    def test_require_ou_ok_with_totals(self):
        """require_ou=True should work when O/U is provided."""
        model = _make_model(with_totals=True, require_ou=True)
        # Mexico vs South Africa has O/U in the fixture
        mat = model.predict_score_matrix("Mexico", "South Africa")
        assert mat.shape == (9, 9)
        assert abs(mat.sum() - 1.0) < 1e-6

    def test_require_ou_false_allows_missing(self):
        """require_ou=False should work without O/U."""
        model = _make_model(require_ou=False)
        mat = model.predict_score_matrix("Mexico", "South Africa")
        assert mat.shape == (9, 9)
