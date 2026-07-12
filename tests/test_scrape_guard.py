"""Defensive guard: parse_prediction_page must not divide by zero on
odds-less matchdays (the scrape_deadlines ZeroDivisionError)."""

from __future__ import annotations

from src.data.kicktipp_scrape import parse_prediction_page

# Legacy-layout table: one real match + one with 0 quotes (no market posted).
_HTML = """
<table class="tippabgabe"><tbody>
  <tr>
    <td class="col1">14/06/26 20:00</td>
    <td class="col3">TeamA</td>
    <td class="col5">TeamB</td>
    <td class="kicktipp-wettquote">2.00</td>
    <td class="kicktipp-wettquote">3.00</td>
    <td class="kicktipp-wettquote">4.00</td>
  </tr>
  <tr>
    <td class="col1">15/06/26 20:00</td>
    <td class="col3">TeamC</td>
    <td class="col5">TeamD</td>
    <td class="kicktipp-wettquote">0</td>
    <td class="kicktipp-wettquote">0</td>
    <td class="kicktipp-wettquote">0</td>
  </tr>
</tbody></table>
"""


def test_zero_odds_row_skipped_without_crash():
    matches = parse_prediction_page(_HTML)          # must NOT raise ZeroDivisionError
    homes = {m.home_team for m in matches}
    assert "TeamA" in homes                          # real match parsed
    assert "TeamC" not in homes                       # zero-odds match skipped


def test_valid_row_probabilities_normalise():
    matches = parse_prediction_page(_HTML)
    m = next(x for x in matches if x.home_team == "TeamA")
    assert abs(m.prob_home + m.prob_draw + m.prob_away - 1.0) < 1e-6
    assert m.prob_home > m.prob_away                  # 2.00 < 4.00 -> home favoured
