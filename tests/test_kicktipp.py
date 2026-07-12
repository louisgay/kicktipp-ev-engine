"""Tests for kicktipp scraper/parser (offline, using HTML fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.kicktipp_scrape import (
    BonusQuestion,
    MatchOdds,
    format_bonus_table,
    format_odds_table,
    parse_bonus_page,
    parse_prediction_page,
)

_FIXTURES = Path(__file__).parent / "fixtures"


# -- Prediction page tests --------------------------------------------


class TestParsePredictionPage:
    @pytest.fixture(autouse=True)
    def _load(self):
        html = (_FIXTURES / "predict_matchday1.html").read_text(encoding="utf-8")
        self.matches = parse_prediction_page(html)

    def test_match_count(self):
        assert len(self.matches) == 8

    def test_returns_match_odds(self):
        assert all(isinstance(m, MatchOdds) for m in self.matches)

    def test_first_match_teams(self):
        m = self.matches[0]
        assert m.home_team == "Mexico"
        assert m.away_team == "South Africa"

    def test_first_match_odds(self):
        m = self.matches[0]
        assert m.odds_home == 1.53
        assert m.odds_draw == 4.30
        assert m.odds_away == 6.10

    def test_probabilities_sum_to_one(self):
        for m in self.matches:
            total = m.prob_home + m.prob_draw + m.prob_away
            assert abs(total - 1.0) < 1e-5, (
                f"{m.home_team} vs {m.away_team}: probs sum to {total}"
            )

    def test_overround_near_one(self):
        """Kicktipp odds are pre-devigged, overround should be close to 1.0."""
        for m in self.matches:
            assert 0.95 < m.overround < 1.10, (
                f"{m.home_team} vs {m.away_team}: overround = {m.overround}"
            )

    def test_match_ids(self):
        assert self.matches[0].match_id == "match_1234001"
        assert self.matches[4].match_id == "match_1234005"

    def test_datetime_extracted(self):
        assert self.matches[0].datetime_str == "11/06/26 20:00"

    def test_result_is_none_for_unplayed(self):
        """All fixture matches are unplayed (---)."""
        for m in self.matches:
            assert m.result is None

    def test_germany_japan_odds(self):
        m = self.matches[4]  # Germany vs Japan
        assert m.home_team == "Germany"
        assert m.away_team == "Japan"
        assert m.odds_home == 1.60
        assert m.odds_draw == 4.00
        assert m.odds_away == 5.50

    def test_heavy_favourite_usa_fiji(self):
        m = self.matches[1]  # USA vs Fiji
        assert m.home_team == "USA"
        assert m.away_team == "Fiji"
        assert m.prob_home > 0.85  # very heavy favourite


# -- Bonus page tests -------------------------------------------------


class TestParseKnockoutPredictionPage:
    """Knockout (a.PSO) layout: 2-way 'advance' odds inline in a single
    td.quoten cell as '1 <oh> X 0.00 2 <oa>', with an a.PSO spielabschnitt cell."""

    _HTML = """
    <div id="kicktipp-content"><table><tbody>
      <tr><td>Dead line</td><td></td><td></td><td></td><td class="kicktipp-quote"></td></tr>
      <tr>
        <td class="nw kicktipp-time">28/06/26 20:00</td>
        <td class="nw">South Africa</td>
        <td class="nw">Canada</td>
        <td class="kicktipp-tippabgabe spielabschnitt">a.PSO</td>
        <td class="nw quoten">1 6.86 X 0.00 2 1.17</td>
      </tr>
      <tr>
        <td class="nw kicktipp-time">29/06/26 18:00</td>
        <td class="nw">Brazil</td>
        <td class="nw">Japan</td>
        <td class="kicktipp-tippabgabe spielabschnitt">a.PSO</td>
        <td class="nw quoten">1 1.08 X 0.00 2 13.8</td>
      </tr>
    </tbody></table></div>
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        self.matches = parse_prediction_page(self._HTML)

    def test_both_knockout_matches_parsed(self):
        assert len(self.matches) == 2
        assert all(m.a_pso for m in self.matches)

    def test_teams_and_two_way_probs(self):
        m = self.matches[0]
        assert (m.home_team, m.away_team) == ("South Africa", "Canada")
        assert m.prob_draw == 0.0                       # no draw in a.PSO
        # 2-way devig of 6.86 / 1.17
        assert m.prob_home == pytest.approx((1 / 6.86) / (1 / 6.86 + 1 / 1.17), abs=1e-4)
        assert m.prob_home + m.prob_away == pytest.approx(1.0, abs=1e-6)

    def test_favourite_side(self):
        brazil = self.matches[1]
        assert brazil.prob_home > 0.9 and brazil.prob_away < 0.1


class TestParseBonusPage:
    @pytest.fixture(autouse=True)
    def _load(self):
        html = (_FIXTURES / "bonus.html").read_text(encoding="utf-8")
        self.questions = parse_bonus_page(html)

    def test_question_count(self):
        # 1 world champion + 1 top scorer + 3 group winners + 1 semi-finalists = 6
        assert len(self.questions) == 6

    def test_returns_bonus_questions(self):
        assert all(isinstance(q, BonusQuestion) for q in self.questions)

    def test_world_champion_question(self):
        q = self.questions[0]
        assert "World Champion" in q.question_text
        assert q.question_type == "single"
        assert q.question_id == "bonus_100001"

    def test_world_champion_options(self):
        q = self.questions[0]
        labels = [o["label"] for o in q.options]
        assert "Argentina" in labels
        assert "France" in labels
        assert "Germany" in labels
        assert len(q.options) >= 48  # 48 WC teams

    def test_group_winner_has_4_options(self):
        # Group A winner question
        q = self.questions[2]  # "Which team will win group A?"
        assert "group A" in q.question_text
        assert len(q.options) == 4
        labels = [o["label"] for o in q.options]
        assert "Mexico" in labels
        assert "Argentina" in labels

    def test_semi_finalists_multi_select(self):
        q = self.questions[-1]  # Semi-finalists
        assert "semi-finals" in q.question_text.lower()
        assert q.question_type == "multi"
        assert q.select_count == 4

    def test_deadlines_parsed(self):
        for q in self.questions:
            assert q.deadline  # not empty

    def test_select_placeholder_excluded(self):
        """'-- Select --' should not appear in options."""
        for q in self.questions:
            labels = [o["label"] for o in q.options]
            assert "-- Select --" not in labels


# -- Format helpers ----------------------------------------------------


class TestFormatHelpers:
    def test_format_odds_table_runs(self):
        html = (_FIXTURES / "predict_matchday1.html").read_text(encoding="utf-8")
        matches = parse_prediction_page(html)
        table = format_odds_table(matches)
        assert "Mexico" in table
        assert "South Africa" in table
        assert len(table.splitlines()) == len(matches) + 2  # header + separator + rows

    def test_format_bonus_table_runs(self):
        html = (_FIXTURES / "bonus.html").read_text(encoding="utf-8")
        questions = parse_bonus_page(html)
        table = format_bonus_table(questions)
        assert "World Champion" in table
        assert "semi-finals" in table.lower()
