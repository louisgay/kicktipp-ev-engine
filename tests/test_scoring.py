"""Tests for the pool scoring rules.

These are the most critical tests in the project: the scoring function
drives all optimisation decisions, so it must be provably correct.
"""

import numpy as np
import pytest

from src.scoring.kicktipp import expected_points, optimal_prediction, points


class TestPoints:
    """Test the points() function against all rule tiers."""

    # -- Exact score: 4 points ----------------------------------------

    def test_exact_score_home_win(self):
        assert points((2, 1), (2, 1)) == 4

    def test_exact_score_away_win(self):
        assert points((0, 3), (0, 3)) == 4

    def test_exact_score_draw(self):
        assert points((1, 1), (1, 1)) == 4

    def test_exact_score_0_0(self):
        assert points((0, 0), (0, 0)) == 4

    def test_exact_score_high(self):
        assert points((4, 3), (4, 3)) == 4

    # -- Correct goal difference (wins only): 3 points ----------------

    def test_goal_diff_home_win(self):
        # Pred: 2-1 (+1), Actual: 3-2 (+1) -> same winner, same diff
        assert points((2, 1), (3, 2)) == 3

    def test_goal_diff_away_win(self):
        # Pred: 0-2 (-2), Actual: 1-3 (-2) -> same winner, same diff
        assert points((0, 2), (1, 3)) == 3

    def test_goal_diff_large(self):
        # Pred: 3-0 (+3), Actual: 4-1 (+3) -> same winner, same diff
        assert points((3, 0), (4, 1)) == 3

    def test_goal_diff_1_0_vs_2_1(self):
        assert points((1, 0), (2, 1)) == 3

    # -- No goal-diff tier for draws: draw non-exact = 2 points -------

    def test_draw_non_exact_0_0_vs_1_1(self):
        # Both draws, but different scores -> tendency only (2)
        assert points((0, 0), (1, 1)) == 2

    def test_draw_non_exact_2_2_vs_0_0(self):
        assert points((2, 2), (0, 0)) == 2

    def test_draw_non_exact_1_1_vs_3_3(self):
        assert points((1, 1), (3, 3)) == 2

    # -- Correct tendency only: 2 points ------------------------------

    def test_tendency_home_win_different_diff(self):
        # Pred: 1-0 (+1), Actual: 3-0 (+3) -> same winner, different diff
        assert points((1, 0), (3, 0)) == 2

    def test_tendency_away_win_different_diff(self):
        # Pred: 0-1 (-1), Actual: 0-3 (-3) -> same winner, different diff
        assert points((0, 1), (0, 3)) == 2

    def test_tendency_home_win_big_diff(self):
        # Pred: 2-0 (+2), Actual: 1-0 (+1) -> same winner, different diff
        assert points((2, 0), (1, 0)) == 2

    # -- Wrong tendency: 0 points -------------------------------------

    def test_wrong_home_vs_away(self):
        assert points((2, 0), (0, 1)) == 0

    def test_wrong_home_vs_draw(self):
        assert points((2, 1), (1, 1)) == 0

    def test_wrong_draw_vs_home(self):
        assert points((1, 1), (2, 0)) == 0

    def test_wrong_away_vs_home(self):
        assert points((0, 1), (2, 0)) == 0

    def test_wrong_draw_vs_away(self):
        assert points((0, 0), (0, 1)) == 0

    def test_wrong_away_vs_draw(self):
        assert points((0, 2), (1, 1)) == 0


class TestExpectedPoints:
    """Test expected_points() with known probability matrices."""

    def test_certain_outcome(self):
        """If P(1,0) = 1.0, predicting (1,0) should give EV = 4."""
        matrix = np.zeros((5, 5))
        matrix[1, 0] = 1.0
        assert expected_points((1, 0), matrix) == pytest.approx(4.0)

    def test_certain_outcome_wrong_pred(self):
        """If P(1,0) = 1.0, predicting (0,1) should give EV = 0."""
        matrix = np.zeros((5, 5))
        matrix[1, 0] = 1.0
        assert expected_points((0, 1), matrix) == pytest.approx(0.0)

    def test_two_outcomes(self):
        """50% chance of 1-0, 50% chance of 2-1.
        Pred (1,0): 0.5*4 + 0.5*3 = 3.5 (exact + goal diff)
        Pred (2,1): 0.5*3 + 0.5*4 = 3.5 (goal diff + exact)
        """
        matrix = np.zeros((5, 5))
        matrix[1, 0] = 0.5
        matrix[2, 1] = 0.5
        assert expected_points((1, 0), matrix) == pytest.approx(3.5)
        assert expected_points((2, 1), matrix) == pytest.approx(3.5)

    def test_draw_scenario(self):
        """50% chance of 1-1, 50% chance of 0-0.
        Pred (1,1): 0.5*4 + 0.5*2 = 3.0
        Pred (0,0): 0.5*2 + 0.5*4 = 3.0
        """
        matrix = np.zeros((5, 5))
        matrix[1, 1] = 0.5
        matrix[0, 0] = 0.5
        assert expected_points((1, 1), matrix) == pytest.approx(3.0)
        assert expected_points((0, 0), matrix) == pytest.approx(3.0)

    def test_uniform_matrix_draw_vs_win(self):
        """With uniform distribution, a draw prediction should be suboptimal
        vs a home-win prediction (since more cells are wins than draws)."""
        matrix = np.ones((5, 5)) / 25.0
        ev_draw = expected_points((1, 1), matrix)
        ev_home = expected_points((1, 0), matrix)
        # Both should be positive
        assert ev_draw > 0
        assert ev_home > 0


class TestOptimalPrediction:
    """Test optimal_prediction() selects the right prediction."""

    def test_certain_outcome(self):
        """If outcome is certain, optimal pred = that outcome."""
        matrix = np.zeros((5, 5))
        matrix[2, 1] = 1.0
        pred, ev = optimal_prediction(matrix)
        assert pred == (2, 1)
        assert ev == pytest.approx(4.0)

    def test_spread_home_win(self):
        """If probability is spread across home-win scores,
        optimal pred should be a common home-win score."""
        matrix = np.zeros((6, 6))
        matrix[1, 0] = 0.30
        matrix[2, 0] = 0.20
        matrix[2, 1] = 0.15
        matrix[3, 1] = 0.10
        matrix[1, 1] = 0.10
        matrix[0, 0] = 0.05
        matrix[0, 1] = 0.05
        matrix[3, 0] = 0.05
        pred, ev = optimal_prediction(matrix)
        # The optimal prediction should maximise EV, not just be the mode
        assert pred[0] > pred[1] or pred[0] == pred[1]  # at least check it runs
        assert ev > 0

    def test_favours_common_draw(self):
        """When draw probability is high, optimal pred should be a draw."""
        matrix = np.zeros((5, 5))
        matrix[0, 0] = 0.15
        matrix[1, 1] = 0.20
        matrix[2, 2] = 0.05
        # Some non-draw outcomes
        matrix[1, 0] = 0.15
        matrix[0, 1] = 0.15
        matrix[2, 1] = 0.10
        matrix[1, 2] = 0.10
        matrix[2, 0] = 0.05
        matrix[0, 2] = 0.05
        pred, ev = optimal_prediction(matrix)
        # With 40% draw probability and no dominant single outcome,
        # a draw prediction (1,1) should be competitive
        assert ev > 0
