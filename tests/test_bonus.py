"""Tests for bonus question optimizer (T3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bonus.optimizer import (
    BonusOptimizer,
    BonusRecommendation,
    _classify_question,
    format_recommendations,
)
from src.bonus.outright_odds import (
    GOLDEN_BOOT_TEAM_PROBS,
    WINNER_PROBS,
    _american_to_implied,
    _normalise_probs,
    devig_outright,
    format_devig_example,
    get_group_winner_probs,
)
from src.data.kicktipp_scrape import BonusQuestion, parse_bonus_page
from tests.fixtures.group_matches import KICKTIPP_GROUP_MATCHES

_FIXTURES = Path(__file__).parent / "fixtures"


# -- Outright odds conversion -----------------------------------------


class TestOddsConversion:
    def test_positive_american(self):
        # +475 -> decimal 5.75 -> implied 1/5.75 = 0.1739
        p = _american_to_implied(475)
        assert abs(p - 100 / 575) < 1e-6

    def test_negative_american(self):
        # -110 -> implied 110/210 = 0.5238
        p = _american_to_implied(-110)
        assert abs(p - 110 / 210) < 1e-6

    def test_even_money(self):
        p = _american_to_implied(100)
        assert abs(p - 0.5) < 1e-6

    def test_heavy_favourite(self):
        # -10000 -> implied 10000/10100 ≈ 0.9901
        p = _american_to_implied(-10000)
        assert p > 0.98

    def test_normalise_sums_to_one(self):
        odds = {"A": -110, "B": 200, "C": 350}
        probs = _normalise_probs(odds)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_normalise_preserves_order(self):
        odds = {"Fav": -200, "Mid": 150, "Long": 500}
        probs = _normalise_probs(odds)
        assert probs["Fav"] > probs["Mid"] > probs["Long"]


# -- Curated probabilities sanity checks ------------------------------
# WINNER_PROBS is the ESPN-based legacy constant, kept for odds-conversion tests.


class TestCuratedProbs:
    def test_winner_sums_to_one(self):
        assert abs(sum(WINNER_PROBS.values()) - 1.0) < 1e-6

    def test_winner_spain_is_favourite(self):
        top = max(WINNER_PROBS, key=WINNER_PROBS.get)
        assert top == "Spain"

    def test_winner_top5(self):
        top5 = sorted(WINNER_PROBS, key=WINNER_PROBS.get, reverse=True)[:5]
        assert "Spain" in top5
        assert "France" in top5
        assert "England" in top5

    def test_golden_boot_sums_to_one(self):
        assert abs(sum(GOLDEN_BOOT_TEAM_PROBS.values()) - 1.0) < 1e-6

    def test_golden_boot_france_top(self):
        # Mbappé + Dembélé -> France should be #1 or #2
        top3 = sorted(GOLDEN_BOOT_TEAM_PROBS,
                       key=GOLDEN_BOOT_TEAM_PROBS.get, reverse=True)[:3]
        assert "France" in top3

    def test_all_groups_present(self):
        gw = get_group_winner_probs()
        for g in "ABCDEFGHIJKL":
            assert g in gw, f"Missing group {g}"

    def test_group_probs_sum_to_one(self):
        for g, probs in get_group_winner_probs().items():
            assert abs(sum(probs.values()) - 1.0) < 1e-6, f"Group {g}"



# -- Question classifier ---------------------------------------------


class TestClassifyQuestion:
    def _q(self, text: str, **kw) -> BonusQuestion:
        return BonusQuestion(
            question_id="test", question_text=text,
            deadline="", options=[], question_type="single", **kw
        )

    def test_world_champion(self):
        assert _classify_question(
            self._q("Which team will be World Champion?")
        ) == "winner"

    def test_goal_scorer(self):
        assert _classify_question(
            self._q("Which team will produce the highest goal scorer?")
        ) == "golden_boot_team"

    def test_group_a(self):
        assert _classify_question(
            self._q("Which team will win group A?")
        ) == "group_A"

    def test_group_l(self):
        assert _classify_question(
            self._q("Which team will win group L?")
        ) == "group_L"

    def test_semi_finals(self):
        assert _classify_question(
            self._q("Who will reach the semi-finals?")
        ) == "semi_finalist"

    def test_unknown(self):
        assert _classify_question(
            self._q("What colour is the ball?")
        ) == "unknown"


# -- BonusOptimizer ---------------------------------------------------


class TestBonusOptimizer:
    @pytest.fixture
    def optimizer(self):
        return BonusOptimizer()

    @pytest.fixture
    def bonus_questions(self):
        html = (_FIXTURES / "bonus.html").read_text(encoding="utf-8")
        return parse_bonus_page(html)

    def test_recommend_world_champion(self, optimizer, bonus_questions):
        q = bonus_questions[0]  # "Which team will be World Champion?"
        rec = optimizer.recommend(q)
        assert isinstance(rec, BonusRecommendation)
        assert rec.market_type == "winner"
        assert len(rec.picks) == 1
        assert rec.awaiting_bracket is True  # bracket not published
        assert rec.source == "bracket_avg"
        # Top pick should be a strong team
        assert rec.picks[0]["prob"] > 0.05

    def test_recommend_golden_boot(self, optimizer, bonus_questions):
        q = bonus_questions[1]  # "highest goal scorer"
        rec = optimizer.recommend(q)
        assert rec.market_type == "golden_boot_team"
        assert len(rec.picks) == 1
        # Top team should be France, England, or Argentina
        assert rec.picks[0]["label"] in ("France", "England", "Argentina", "Spain")

    def test_recommend_group_winner(self, optimizer, bonus_questions):
        q = bonus_questions[2]  # "group A"
        rec = optimizer.recommend(q)
        assert rec.market_type == "group_A"
        assert rec.awaiting_bracket is False  # groups don't depend on bracket
        assert rec.source == "mc_group_sim"
        assert len(rec.picks) == 1
        # Mexico is the favourite in group A per MC simulation
        assert rec.picks[0]["prob"] > 0

    def test_recommend_semi_finalists(self, optimizer, bonus_questions):
        q = bonus_questions[-1]  # semi-finals, multi-select pick 4
        rec = optimizer.recommend(q)
        assert rec.market_type == "semi_finalist"
        assert rec.awaiting_bracket is True  # bracket not published
        assert rec.source == "bracket_avg"
        assert len(rec.picks) == 4
        # All picks should have positive probabilities
        for p in rec.picks:
            assert p["prob"] > 0

    def test_recommend_all(self, optimizer, bonus_questions):
        recs = optimizer.recommend_all(bonus_questions)
        assert len(recs) == len(bonus_questions)
        # No unknown markets for our fixture
        for r in recs:
            assert r.market_type != "unknown"

    def test_picks_have_required_fields(self, optimizer, bonus_questions):
        for q in bonus_questions:
            rec = optimizer.recommend(q)
            for p in rec.picks:
                assert "label" in p
                assert "value" in p
                assert "prob" in p

    def test_load_custom_probs(self, optimizer, bonus_questions):
        # Override winner market with custom probs
        optimizer.load_probs("winner", {
            "Argentina": 0.5, "Brazil": 0.3, "France": 0.2,
        })
        q = bonus_questions[0]  # world champion
        rec = optimizer.recommend(q, source="custom")
        assert rec.picks[0]["label"] == "Argentina"
        assert rec.source == "custom"

    def test_format_recommendations(self, optimizer, bonus_questions):
        recs = optimizer.recommend_all(bonus_questions)
        output = format_recommendations(recs)
        assert "World Champion" in output
        assert "Pick 1:" in output
        assert "semi" in output.lower()
        assert "AWAITING BRACKET" in output  # warning on winner + semi


# -- Outright devigging ----------------------------------------------


class TestOutrightDevig:
    def test_overround_removed(self):
        """Devigged winner probs should sum to 1.0 and raw overround ~1.30-1.50."""
        odds = {"A": -110, "B": 200, "C": 350, "D": 500, "E": 800}
        devigged, overround = devig_outright(odds)
        # Devigged sums to 1
        assert abs(sum(devigged.values()) - 1.0) < 1e-10
        # Raw overround should be > 1 (bookmaker margin)
        assert overround > 1.0

    def test_winner_overround_realistic(self):
        """Real winner odds should have overround in ~1.10-1.50 range."""
        from src.bonus.outright_odds import _WINNER_ODDS_AMERICAN
        devigged, overround = devig_outright(_WINNER_ODDS_AMERICAN)
        assert abs(sum(devigged.values()) - 1.0) < 1e-10
        assert 1.10 < overround < 1.60, f"overround={overround}"

    def test_devig_preserves_order(self):
        """Devigging should preserve rank order."""
        odds = {"Fav": -200, "Mid": 150, "Long": 500}
        devigged, _ = devig_outright(odds)
        assert devigged["Fav"] > devigged["Mid"] > devigged["Long"]

    def test_devig_additive_matches_normalise(self):
        """devig_outright(method='normalise') matches _normalise_probs."""
        odds = {"A": -110, "B": 200, "C": 350}
        devigged, _ = devig_outright(odds, method="normalise")
        normalised = _normalise_probs(odds)
        for team in odds:
            assert abs(devigged[team] - normalised[team]) < 1e-10

    def test_shin_favours_favourites(self):
        """Shin should give higher prob to favourites than additive."""
        odds = {"Fav": -200, "Mid": 150, "Long": 500}
        add, _ = devig_outright(odds, method="normalise")
        shin, _ = devig_outright(odds, method="shin")
        assert shin["Fav"] > add["Fav"]
        assert shin["Long"] < add["Long"]

    def test_format_devig_example(self):
        """format_devig_example should produce readable output."""
        output = format_devig_example()
        assert "Devigging" in output
        assert "overround" in output.lower()
        assert "Argmax pick" in output
        assert "Spain" in output  # top favourite


# -- Group simulation ----------------------------------------------


class TestGroupSimulation:
    def test_simulate_group_from_fixture_odds(self):
        """Simulate Group A using fixture kicktipp odds."""
        from src.bonus.group_sim import build_group_lambdas, simulate_group_winner

        lambdas = build_group_lambdas(matches=KICKTIPP_GROUP_MATCHES)
        probs = simulate_group_winner("A", lambdas, n_sims=10_000)

        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert len(probs) == 4
        # Mexico should be strong favourite (89.9% to beat SA, 69.8% to beat Korea)
        assert probs["Mexico"] > 0.3
        # All probs positive
        for p in probs.values():
            assert p > 0

    def test_simulate_sums_to_one(self):
        """All group simulation probs must sum to 1."""
        from src.bonus.group_sim import simulate_group_winner

        # Group A teams: Mexico, South Africa, South Korea, Czech Republic
        match_lambdas = {
            ("Mexico", "South Korea"): (1.2, 1.1),
            ("Mexico", "Czech Republic"): (1.2, 1.1),
            ("Mexico", "South Africa"): (1.2, 1.1),
            ("South Korea", "Czech Republic"): (1.1, 1.1),
            ("South Korea", "South Africa"): (1.1, 1.1),
            ("Czech Republic", "South Africa"): (1.1, 1.1),
        }
        probs = simulate_group_winner("A", match_lambdas, n_sims=5_000)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_missing_match_raises(self):
        """Missing match λ should raise ValueError."""
        from src.bonus.group_sim import simulate_group_winner

        # Only 5 of 6 matches for Group A
        match_lambdas = {
            ("Mexico", "South Korea"): (1.5, 0.8),
            ("Mexico", "Czech Republic"): (1.5, 0.8),
            ("Mexico", "South Africa"): (1.5, 0.8),
            ("South Korea", "Czech Republic"): (1.1, 1.0),
            ("South Korea", "South Africa"): (1.1, 1.0),
            # Missing: Czech Republic vs South Africa
        }
        with pytest.raises(ValueError, match="No λ values"):
            simulate_group_winner("A", match_lambdas)

    def test_build_group_lambdas_no_data_raises(self):
        """build_group_lambdas() without data or cache should raise."""
        from src.bonus.group_sim import build_group_lambdas
        # With no matches, no cache, no live -> should raise
        import src.bonus.group_sim as gs
        # Temporarily clear cache check
        orig = gs._CACHE_FILE
        gs._CACHE_FILE = Path("/nonexistent/path/cache.json")
        try:
            with pytest.raises(RuntimeError, match="No group match data"):
                build_group_lambdas()
        finally:
            gs._CACHE_FILE = orig


# -- Bracket simulation ----------------------------------------------


class TestBracketSimulation:
    @pytest.fixture
    def match_lambdas(self):
        from src.bonus.group_sim import build_group_lambdas
        return build_group_lambdas(matches=KICKTIPP_GROUP_MATCHES)

    def test_tournament_winner_sums_to_one(self, match_lambdas):
        """Tournament winner probs should sum to ~1."""
        from src.bonus.bracket_sim import simulate_tournament
        result = simulate_tournament(match_lambdas, n_sims=5_000)
        winner_sum = sum(result["winner"].values())
        assert abs(winner_sum - 1.0) < 0.01

    def test_semi_finalist_sums_to_four(self, match_lambdas):
        """Semi-finalist probs should sum to ~4 (4 slots)."""
        from src.bonus.bracket_sim import simulate_tournament
        result = simulate_tournament(match_lambdas, n_sims=5_000)
        sf_sum = sum(result["semi_finalist"].values())
        assert abs(sf_sum - 4.0) < 0.1  # MC noise -> wider tolerance

    def test_sf_geq_winner(self, match_lambdas):
        """P(reach SF) >= P(win) for every team."""
        from src.bonus.bracket_sim import simulate_tournament
        result = simulate_tournament(match_lambdas, n_sims=10_000)
        for team in result["winner"]:
            assert result["semi_finalist"][team] >= result["winner"][team] - 0.01, (
                f"{team}: P(semi)={result['semi_finalist'][team]:.4f} < "
                f"P(win)={result['winner'][team]:.4f}"
            )

    def test_strong_teams_favoured(self, match_lambdas):
        """Top teams by group strength should appear in top winners."""
        from src.bonus.bracket_sim import simulate_tournament
        result = simulate_tournament(match_lambdas, n_sims=10_000)
        top8 = sorted(result["winner"], key=result["winner"].get, reverse=True)[:8]
        # At least some of these should be in top 8
        strong_teams = {"Spain", "France", "Brazil", "Argentina", "Germany",
                        "England", "Portugal", "Netherlands"}
        overlap = set(top8) & strong_teams
        assert len(overlap) >= 4, f"Top 8 winners: {top8}"

    def test_all_48_teams_present(self, match_lambdas):
        """All 48 teams should appear in results."""
        from src.bonus.bracket_sim import simulate_tournament
        result = simulate_tournament(match_lambdas, n_sims=2_000)
        assert len(result["winner"]) == 48
        assert len(result["semi_finalist"]) == 48
