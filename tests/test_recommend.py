"""Tests for consolidated recommendation engine (T4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.kicktipp_scrape import MatchOdds, parse_bonus_page, parse_prediction_page
from src.recommend import (
    MatchRecommendation,
    RecommendationSheet,
    build_bonus_recommendations,
    build_match_recommendations,
    build_recommendation_sheet,
    format_match_recommendations,
    format_sheet,
    main,
)

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def matches():
    html = (_FIXTURES / "predict_matchday1.html").read_text(encoding="utf-8")
    return parse_prediction_page(html)


@pytest.fixture
def questions():
    html = (_FIXTURES / "bonus.html").read_text(encoding="utf-8")
    return parse_bonus_page(html)


# Full O/U totals for all 8 matches in the fixture
_ALL_TOTALS = [
    {"home_team": "Mexico", "away_team": "South Africa",
     "totals_odds": [2.10, 1.80], "totals_line": 2.5},
    {"home_team": "USA", "away_team": "Fiji",
     "totals_odds": [1.60, 2.40], "totals_line": 2.5},
    {"home_team": "Argentina", "away_team": "Morocco",
     "totals_odds": [1.95, 1.95], "totals_line": 2.5},
    {"home_team": "France", "away_team": "Colombia",
     "totals_odds": [1.90, 2.00], "totals_line": 2.5},
    {"home_team": "Germany", "away_team": "Japan",
     "totals_odds": [1.95, 1.95], "totals_line": 2.5},
    {"home_team": "Spain", "away_team": "Ecuador",
     "totals_odds": [1.70, 2.20], "totals_line": 2.5},
    {"home_team": "England", "away_team": "Denmark",
     "totals_odds": [1.85, 2.05], "totals_line": 2.5},
    {"home_team": "Brazil", "away_team": "Serbia",
     "totals_odds": [2.00, 1.90], "totals_line": 2.5},
]


class TestBuildMatchRecommendations:
    def test_returns_list(self, matches):
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        assert isinstance(recs, list)
        assert len(recs) == len(matches)

    def test_returns_match_recommendations(self, matches):
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        assert all(isinstance(r, MatchRecommendation) for r in recs)

    def test_predictions_are_integers(self, matches):
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        for r in recs:
            assert isinstance(r.pred_home, int)
            assert isinstance(r.pred_away, int)
            assert r.pred_home >= 0
            assert r.pred_away >= 0

    def test_ev_positive(self, matches):
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        for r in recs:
            assert r.ev > 0

    def test_favourite_wins(self, matches):
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        # Mexico vs South Africa: Mexico is favourite
        r = recs[0]
        assert r.home_team == "Mexico"
        assert r.pred_home > r.pred_away

    def test_has_ou_true_with_all_totals(self, matches):
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        for r in recs:
            assert r.has_ou is True

    def test_missing_ou_raises(self, matches):
        """build_match_recommendations raises ValueError when O/U is missing."""
        with pytest.raises(ValueError, match="O/U data missing"):
            build_match_recommendations(matches)

    def test_partial_ou_raises(self, matches):
        """Even partial O/U coverage should raise."""
        partial_totals = [_ALL_TOTALS[0]]  # Only Mexico
        with pytest.raises(ValueError, match="O/U data missing"):
            build_match_recommendations(matches, totals_records=partial_totals)


class TestBuildRecommendationSheet:
    def test_returns_sheet(self, matches, questions):
        sheet = build_recommendation_sheet(
            matches, questions, totals_records=_ALL_TOTALS)
        assert isinstance(sheet, RecommendationSheet)

    def test_sheet_has_matches(self, matches, questions):
        sheet = build_recommendation_sheet(
            matches, questions, totals_records=_ALL_TOTALS)
        assert len(sheet.matches) == 8

    def test_sheet_has_bonus(self, matches, questions):
        sheet = build_recommendation_sheet(
            matches, questions, totals_records=_ALL_TOTALS)
        assert len(sheet.bonus) == 6

    def test_total_ev(self, matches, questions):
        sheet = build_recommendation_sheet(
            matches, questions, totals_records=_ALL_TOTALS)
        expected = sum(r.ev for r in sheet.matches)
        assert abs(sheet.total_ev - round(expected, 2)) < 0.01

    def test_matches_only(self, matches):
        sheet = build_recommendation_sheet(
            matches, totals_records=_ALL_TOTALS)
        assert len(sheet.matches) == 8
        assert len(sheet.bonus) == 0


class TestFormatting:
    def test_format_match_recommendations(self, matches):
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        output = format_match_recommendations(recs)
        assert "Mexico" in output
        assert "South Africa" in output
        assert "EV" in output

    def test_format_sheet(self, matches, questions):
        sheet = build_recommendation_sheet(
            matches, questions, totals_records=_ALL_TOTALS)
        output = format_sheet(sheet)
        assert "PREDICTION SHEET" in output
        assert "MATCH PREDICTIONS" in output
        assert "BONUS PREDICTIONS" in output
        assert "Mexico" in output
        assert "World Champion" in output


class TestModelTotalFallback:
    """Tests for allow_model_total=True (1X2-only fallback for missing O/U)."""

    def test_model_total_fallback_succeeds(self, matches):
        """With allow_model_total=True and no totals, should not raise."""
        recs = build_match_recommendations(
            matches, totals_records=None, allow_model_total=True,
        )
        assert len(recs) == len(matches)

    def test_model_total_ou_source_is_model(self, matches):
        """Without O/U, ou_source should be 'model'."""
        recs = build_match_recommendations(
            matches, totals_records=None, allow_model_total=True,
        )
        for r in recs:
            assert r.ou_source == "model"
            assert r.has_ou is False

    def test_partial_ou_with_fallback(self, matches):
        """Partial O/U: matched -> 'market', unmatched -> 'model'."""
        partial_totals = [_ALL_TOTALS[0]]  # Only Mexico vs South Africa
        recs = build_match_recommendations(
            matches, totals_records=partial_totals, allow_model_total=True,
        )
        # First match (Mexico vs SA) should have market O/U
        assert recs[0].ou_source == "market"
        assert recs[0].has_ou is True
        # Other matches should have model fallback
        for r in recs[1:]:
            assert r.ou_source == "model"
            assert r.has_ou is False

    def test_all_totals_ou_source_is_market(self, matches):
        """With full O/U coverage, all ou_source should be 'market'."""
        recs = build_match_recommendations(matches, totals_records=_ALL_TOTALS)
        for r in recs:
            assert r.ou_source == "market"
            assert r.has_ou is True

    def test_model_total_still_produces_valid_predictions(self, matches):
        """Model-total fallback should still produce valid integer predictions."""
        recs = build_match_recommendations(
            matches, totals_records=None, allow_model_total=True,
        )
        for r in recs:
            assert isinstance(r.pred_home, int)
            assert isinstance(r.pred_away, int)
            assert r.pred_home >= 0
            assert r.pred_away >= 0
            assert r.ev > 0


class TestDeadlinesFormatting:
    def test_format_deadlines(self):
        from src.recommend import MatchdayDeadline, format_deadlines
        deadlines = [
            MatchdayDeadline(matchday=1, first_kickoff="11.06. 18:00", n_matches=6),
            MatchdayDeadline(matchday=2, first_kickoff="12.06. 15:00", n_matches=6),
        ]
        output = format_deadlines(deadlines)
        assert "DEADLINE CALENDAR" in output
        assert "11.06. 18:00" in output
        assert "12.06. 15:00" in output
        assert "6" in output


class TestBonusSummary:
    def test_generate_bonus_summary(self):
        from src.recommend import generate_bonus_summary
        output = generate_bonus_summary()
        assert "BONUS EV-MAX ANSWER SET" in output
        # All 12 groups present
        for g in "ABCDEFGHIJKL":
            assert f"        {g}" in output
        # Key sections
        assert "GOLDEN BOOT" in output
        assert "TOURNAMENT WINNER" in output
        assert "SEMI-FINALISTS" in output
        # Reliability markers
        assert "RELIABLE" in output
        assert "AWAITING BRACKET" in output


class TestCLI:
    def test_offline_matchday_without_totals_raises(self):
        """Offline mode without --totals should raise ValueError."""
        with pytest.raises(ValueError, match="O/U data missing"):
            main([
                "--matchday", str(_FIXTURES / "predict_matchday1.html"),
            ])

    def test_offline_matchday_with_totals(self, tmp_path):
        """Offline mode with --totals should succeed."""
        import json
        totals_file = tmp_path / "totals.json"
        totals_file.write_text(json.dumps(_ALL_TOTALS))
        sheet = main([
            "--matchday", str(_FIXTURES / "predict_matchday1.html"),
            "--totals", str(totals_file),
        ])
        assert isinstance(sheet, RecommendationSheet)
        assert len(sheet.matches) == 8

    def test_offline_matchday_bonus_and_totals(self, tmp_path):
        import json
        totals_file = tmp_path / "totals.json"
        totals_file.write_text(json.dumps(_ALL_TOTALS))
        sheet = main([
            "--matchday", str(_FIXTURES / "predict_matchday1.html"),
            "--bonus", str(_FIXTURES / "bonus.html"),
            "--totals", str(totals_file),
        ])
        assert len(sheet.matches) == 8
        assert len(sheet.bonus) == 6

    def test_offline_with_allow_model_total(self):
        """--allow-model-total should allow missing O/U without raising."""
        sheet = main([
            "--matchday", str(_FIXTURES / "predict_matchday1.html"),
            "--allow-model-total",
        ])
        assert isinstance(sheet, RecommendationSheet)
        assert len(sheet.matches) == 8
        # All should be model-total since no --totals provided
        for r in sheet.matches:
            assert r.ou_source == "model"

    def test_ou_source_shown_in_format(self, matches):
        """Format output should show O/U source column."""
        recs = build_match_recommendations(
            matches, totals_records=_ALL_TOTALS,
        )
        output = format_match_recommendations(recs)
        assert "O/U" in output
        assert "market" in output
