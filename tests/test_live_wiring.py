"""Tests for the live-path wiring in recommend.py (DIFF 3):
fetch_live_sharp_1x2 (Pinnacle-preferred) and the blend pass-through."""

from __future__ import annotations

from src.data.kicktipp_scrape import MatchOdds
from src.recommend import build_match_recommendations, fetch_live_sharp_1x2


def _mo(home: str, away: str, ph: float, pd: float, pa: float) -> MatchOdds:
    return MatchOdds(
        match_id="1", datetime_str="14/06/26", home_team=home, away_team=away,
        odds_home=1 / ph, odds_draw=1 / pd, odds_away=1 / pa,
        prob_home=ph, prob_draw=pd, prob_away=pa, overround=1.0, result=None,
    )


def test_fetch_live_sharp_prefers_pinnacle(monkeypatch):
    import src.odds.client as client
    event = {
        "home_team": "Brazil", "away_team": "Morocco",
        "bookmakers": [
            {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Brazil", "price": 1.70},
                {"name": "Draw", "price": 3.80},
                {"name": "Morocco", "price": 5.50}]}]},
            {"key": "sport888", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Brazil", "price": 1.50},  # softer book - must NOT be chosen
                {"name": "Draw", "price": 4.50},
                {"name": "Morocco", "price": 7.00}]}]},
        ],
    }
    monkeypatch.setattr(client, "get_odds", lambda **kw: [event])
    recs = fetch_live_sharp_1x2([_mo("Brazil", "Morocco", 0.84, 0.10, 0.06)])
    assert len(recs) == 1
    assert recs[0]["h2h_odds"] == [1.70, 3.80, 5.50]   # Pinnacle, not the mean/soft book


def test_build_recommendations_accepts_blend_passthrough():
    matches = [_mo("Brazil", "Morocco", 0.84, 0.10, 0.06)]
    totals = [{"home_team": "Brazil", "away_team": "Morocco",
               "totals_odds": [2.10, 1.70], "totals_line": 2.5}]
    sharp = [{"home_team": "Brazil", "away_team": "Morocco", "h2h_odds": [1.70, 3.80, 5.50]}]
    # blend_weight=0 (default behaviour) and 0.65 both produce a valid recommendation
    base = build_match_recommendations(matches, totals, sharp_records=sharp, blend_weight=0.0)
    blended = build_match_recommendations(matches, totals, sharp_records=sharp, blend_weight=0.65)
    assert len(base) == 1 and len(blended) == 1
    assert base[0].pred_home >= base[0].pred_away      # Brazil favoured either way
