"""Tests for odds devigging and score matrix reconstruction."""

import numpy as np
import pytest

from src.odds.devig import (
    devig_1x2,
    devig_normalise,
    devig_over_under,
    devig_shin,
    implied_probabilities,
    overround,
)
from src.odds.reconstruct import (
    _1x2_from_matrix,
    _over_under_from_matrix,
    _score_matrix_from_lambdas,
    extract_odds_for_event,
    reconstruct_lambdas,
    reconstruct_matrix,
)


def _h2h(home, away):
    return {"key": "h2h", "outcomes": [
        {"name": home, "price": 1.55}, {"name": "Draw", "price": 4.2},
        {"name": away, "price": 6.0}]}


def _totals(*outcomes):
    """outcomes: tuples of (name, point, price)."""
    return {"key": "totals", "outcomes": [
        {"name": n, "point": pt, "price": pr} for (n, pt, pr) in outcomes]}


class TestExtractOddsOU:
    """O/U extraction must never pair Over and Under from different books/markets."""

    def test_cross_book_leak_is_gone(self):
        # Pinnacle quotes only Under@2.5; Betfair only Over@2.5. The Over and the
        # Under live in different books -> NO valid 2.5 line -> totals absent.
        ev = {"home_team": "Mexico", "away_team": "South Africa", "commence_time": "",
              "bookmakers": [
                  {"key": "pinnacle", "markets": [_h2h("Mexico", "South Africa"),
                                                  _totals(("Under", 2.5, 1.50))]},
                  {"key": "betfair_ex_eu", "markets": [_totals(("Over", 2.5, 1.95))]},
              ]}
        rec = extract_odds_for_event(ev)
        assert rec is not None and rec["h2h_odds"] is not None   # h2h still found
        assert rec["totals_odds"] is None                        # leak gone
        assert rec["totals_line"] is None

    def test_clean_single_book_unchanged(self):
        ev = {"home_team": "France", "away_team": "Colombia", "commence_time": "",
              "bookmakers": [
                  {"key": "pinnacle", "markets": [_h2h("France", "Colombia"),
                      _totals(("Over", 2.5, 1.95), ("Under", 2.5, 1.90))]},
              ]}
        rec = extract_odds_for_event(ev)
        assert rec["totals_odds"] == [1.95, 1.90]
        assert rec["totals_line"] == 2.5

    def test_preferred_book_priority(self):
        # Both books quote a full 2.5 line; pinnacle (preferred) wins.
        ev = {"home_team": "Spain", "away_team": "Ecuador", "commence_time": "",
              "bookmakers": [
                  {"key": "betclic", "markets": [
                      _totals(("Over", 2.5, 2.10), ("Under", 2.5, 1.75))]},
                  {"key": "pinnacle", "markets": [_h2h("Spain", "Ecuador"),
                      _totals(("Over", 2.5, 1.90), ("Under", 2.5, 1.92))]},
              ]}
        rec = extract_odds_for_event(ev)
        assert rec["totals_odds"] == [1.90, 1.92]   # pinnacle, not betclic

    def test_consensus_fallback_averages_only_complete_pairs(self):
        # No preferred book; two non-preferred books each quote BOTH sides ->
        # consensus averages them. A third book quotes only Over (orphan) and a
        # fourth only Under -> those fragments are NOT paired into the average.
        ev = {"home_team": "Brazil", "away_team": "Serbia", "commence_time": "",
              "bookmakers": [
                  {"key": "h2h_only", "markets": [_h2h("Brazil", "Serbia")]},
                  {"key": "book_a", "markets": [
                      _totals(("Over", 2.5, 1.80), ("Under", 2.5, 2.00))]},
                  {"key": "book_b", "markets": [
                      _totals(("Over", 2.5, 1.90), ("Under", 2.5, 1.95))]},
                  {"key": "orphan_over", "markets": [_totals(("Over", 2.5, 5.00))]},
                  {"key": "orphan_under", "markets": [_totals(("Under", 2.5, 5.00))]},
              ]}
        rec = extract_odds_for_event(ev)
        # mean of complete pairs only: over (1.80+1.90)/2=1.85, under (2.00+1.95)/2=1.975
        assert rec["totals_odds"][0] == pytest.approx(1.85)
        assert rec["totals_odds"][1] == pytest.approx(1.975)

    def test_no_totals_at_all_is_none(self):
        ev = {"home_team": "Qatar", "away_team": "Switzerland", "commence_time": "",
              "bookmakers": [{"key": "pinnacle", "markets": [_h2h("Qatar", "Switzerland")]}]}
        rec = extract_odds_for_event(ev)
        assert rec["totals_odds"] is None


class TestImpliedProbabilities:
    def test_even_odds(self):
        """Even odds (2.0, 2.0) -> 50/50."""
        probs = implied_probabilities([2.0, 2.0])
        assert probs == pytest.approx([0.5, 0.5])

    def test_three_way(self):
        """Typical 1X2 odds with vig."""
        probs = implied_probabilities([2.10, 3.40, 3.50])
        assert sum(probs) > 1.0  # vig present

    def test_overround(self):
        """Overround should be positive for real odds."""
        ovr = overround([2.10, 3.40, 3.50])
        assert ovr > 0
        assert ovr < 0.15  # typical range


class TestDevigNormalise:
    def test_sums_to_one(self):
        fair = devig_normalise([2.10, 3.40, 3.50])
        assert sum(fair) == pytest.approx(1.0, abs=1e-10)

    def test_fair_odds(self):
        """If odds already have no vig, normalisation is identity."""
        # Odds implying exact 50/30/20
        odds = [1 / 0.5, 1 / 0.3, 1 / 0.2]  # [2.0, 3.33, 5.0]
        fair = devig_normalise(odds)
        assert fair == pytest.approx([0.5, 0.3, 0.2], abs=1e-6)

    def test_preserves_order(self):
        """Favourite should remain favourite after devig."""
        fair = devig_normalise([1.50, 4.00, 6.00])
        assert fair[0] > fair[1] > fair[2]

    def test_two_way(self):
        """Works for 2-outcome markets (O/U)."""
        fair = devig_normalise([1.90, 2.00])
        assert sum(fair) == pytest.approx(1.0)


class TestDevigShin:
    def test_sums_to_one(self):
        fair = devig_shin([2.10, 3.40, 3.50])
        assert sum(fair) == pytest.approx(1.0, abs=1e-6)

    def test_preserves_order(self):
        fair = devig_shin([1.50, 4.00, 6.00])
        assert fair[0] > fair[1] > fair[2]

    def test_differs_from_normalise(self):
        """Shin and normalisation should give different results for typical odds."""
        norm = devig_normalise([1.50, 4.00, 6.00])
        shin = devig_shin([1.50, 4.00, 6.00])
        # Shin should give higher prob to favourite (less bias)
        # Both sum to 1, so they can't be identical
        assert not all(abs(n - s) < 1e-6 for n, s in zip(norm, shin))

    def test_two_way(self):
        fair = devig_shin([1.90, 2.00])
        assert sum(fair) == pytest.approx(1.0, abs=1e-6)

    def test_low_vig_close_to_normalise(self):
        """With very low vig, both methods should converge."""
        # Almost fair odds
        odds = [2.01, 3.34, 5.01]
        norm = devig_normalise(odds)
        shin = devig_shin(odds)
        for n, s in zip(norm, shin):
            assert abs(n - s) < 0.02


class TestDevig1X2:
    def test_returns_three(self):
        p_h, p_d, p_a = devig_1x2(2.10, 3.40, 3.50)
        assert p_h + p_d + p_a == pytest.approx(1.0)
        assert p_h > p_d  # home favourite
        assert p_h > p_a


class TestDevigOU:
    def test_returns_two(self):
        p_over, p_under = devig_over_under(1.95, 1.95)
        assert p_over + p_under == pytest.approx(1.0)
        assert p_over == pytest.approx(0.5, abs=0.02)


class TestScoreMatrixFromLambdas:
    def test_shape(self):
        m = _score_matrix_from_lambdas(1.5, 1.2, rho=0.0, max_goals=8)
        assert m.shape == (9, 9)

    def test_sums_to_one(self):
        m = _score_matrix_from_lambdas(1.5, 1.2, rho=-0.04, max_goals=8)
        assert m.sum() == pytest.approx(1.0, abs=1e-6)

    def test_nonnegative(self):
        m = _score_matrix_from_lambdas(1.5, 1.2, rho=-0.04, max_goals=8)
        assert (m >= 0).all()

    def test_rho_increases_draws(self):
        """Negative rho should increase 0-0 and 1-1 probabilities."""
        m_no_rho = _score_matrix_from_lambdas(1.3, 1.1, rho=0.0)
        m_with_rho = _score_matrix_from_lambdas(1.3, 1.1, rho=-0.1)
        # 0-0 should be higher with negative rho
        assert m_with_rho[0, 0] > m_no_rho[0, 0]
        # 1-1 should be higher with negative rho
        assert m_with_rho[1, 1] > m_no_rho[1, 1]


class Test1X2FromMatrix:
    def test_sums_to_one(self):
        m = _score_matrix_from_lambdas(1.5, 1.0, rho=0.0)
        h, d, a = _1x2_from_matrix(m)
        assert h + d + a == pytest.approx(1.0, abs=1e-6)

    def test_higher_lambda_home_favours_home(self):
        m = _score_matrix_from_lambdas(2.0, 0.8, rho=0.0)
        h, d, a = _1x2_from_matrix(m)
        assert h > a


class TestOverUnderFromMatrix:
    def test_sums_to_one(self):
        m = _score_matrix_from_lambdas(1.5, 1.2, rho=0.0)
        over, under = _over_under_from_matrix(m, 2.5)
        assert over + under == pytest.approx(1.0, abs=1e-6)

    def test_high_lambda_favours_over(self):
        m = _score_matrix_from_lambdas(2.5, 2.0, rho=0.0)
        over, _ = _over_under_from_matrix(m, 2.5)
        assert over > 0.5


class TestReconstructLambdas:
    def test_round_trip(self):
        """Reconstruct lambdas from a known matrix's 1X2 probabilities."""
        lam_h_true, lam_a_true = 1.6, 1.1
        m = _score_matrix_from_lambdas(lam_h_true, lam_a_true, rho=-0.04)
        p_h, p_d, p_a = _1x2_from_matrix(m)
        p_over, _ = _over_under_from_matrix(m, 2.5)

        lam_h, lam_a, err = reconstruct_lambdas(
            p_h, p_d, p_a, p_over, rho=-0.04
        )
        assert lam_h == pytest.approx(lam_h_true, abs=0.05)
        assert lam_a == pytest.approx(lam_a_true, abs=0.05)
        assert err < 0.001

    def test_1x2_only(self):
        """Should work with 1X2 only (no O/U)."""
        m = _score_matrix_from_lambdas(1.3, 0.9, rho=-0.04)
        p_h, p_d, p_a = _1x2_from_matrix(m)
        lam_h, lam_a, err = reconstruct_lambdas(p_h, p_d, p_a, rho=-0.04)
        assert lam_h > 0
        assert lam_a > 0
        assert err < 0.01

    def test_symmetric_match(self):
        """Equal teams should produce similar lambdas."""
        lam_h, lam_a, _ = reconstruct_lambdas(
            p_home=0.35, p_draw=0.30, p_away=0.35, rho=-0.04
        )
        assert abs(lam_h - lam_a) < 0.15


class TestReconstructMatrix:
    def test_shape_and_sum(self):
        m = reconstruct_matrix(0.45, 0.25, 0.30)
        assert m.shape == (9, 9)
        assert m.sum() == pytest.approx(1.0, abs=1e-6)

    def test_nonnegative(self):
        m = reconstruct_matrix(0.45, 0.25, 0.30)
        assert (m >= 0).all()

    def test_reproduces_1x2(self):
        """The reconstructed matrix should closely match the input 1X2."""
        p_h, p_d, p_a = 0.50, 0.25, 0.25
        m = reconstruct_matrix(p_h, p_d, p_a, rho=-0.04)
        r_h, r_d, r_a = _1x2_from_matrix(m)
        assert r_h == pytest.approx(p_h, abs=0.02)
        assert r_d == pytest.approx(p_d, abs=0.02)
        assert r_a == pytest.approx(p_a, abs=0.02)


class TestLambdaFloor:
    """Verify lambda floor of 0.15 is respected."""

    def test_lambda_floor_respected(self):
        """Extreme 1X2 (95-2.5-2.5) should still produce λ ≥ 0.15."""
        lam_h, lam_a, err = reconstruct_lambdas(
            p_home=0.95, p_draw=0.025, p_away=0.025, rho=-0.04
        )
        assert lam_h >= 0.15, f"λ_home={lam_h} below floor"
        assert lam_a >= 0.15, f"λ_away={lam_a} below floor"

    def test_extreme_odds_top10(self):
        """Test the 10 most extreme real kicktipp odds produce valid results.

        All should reconstruct to valid matrices with λ ≥ 0.15.
        """
        # Real extreme kicktipp odds (approximate 1X2 probs):
        # Heavily one-sided matches from the fixture data
        extreme_cases = [
            # (p_home, p_draw, p_away, label)
            (0.87, 0.08, 0.05, "USA vs Fiji-like"),
            (0.90, 0.06, 0.04, "Mexico-like (1.11)"),
            (0.92, 0.05, 0.03, "Switzerland-like (1.06)"),
            (0.91, 0.06, 0.03, "Scotland-like (1.08)"),
            (0.75, 0.15, 0.10, "Turkey-like (1.28)"),
            (0.85, 0.09, 0.06, "Brazil-like (1.16)"),
            (0.80, 0.12, 0.08, "Moderate favourite"),
            (0.70, 0.18, 0.12, "Mild favourite"),
            (0.95, 0.03, 0.02, "Near-certain home"),
            (0.03, 0.05, 0.92, "Near-certain away"),
        ]

        for p_h, p_d, p_a, label in extreme_cases:
            lam_h, lam_a, err = reconstruct_lambdas(
                p_h, p_d, p_a, rho=-0.04
            )
            assert lam_h >= 0.15, f"{label}: λ_home={lam_h} below floor"
            assert lam_a >= 0.15, f"{label}: λ_away={lam_a} below floor"

            # Matrix should be valid
            m = reconstruct_matrix(p_h, p_d, p_a, rho=-0.04)
            assert m.shape == (9, 9), f"{label}: wrong shape"
            assert (m >= 0).all(), f"{label}: negative probabilities"
            assert m.sum() == pytest.approx(1.0, abs=1e-6), (
                f"{label}: doesn't sum to 1"
            )
