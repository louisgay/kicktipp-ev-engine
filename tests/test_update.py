"""Tests for src/update.py - deadline/lead window, quota guard, login detection,
crontab, and the get_odds force_refresh enabler. No live calls (all mocked)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import update
from src.data.kicktipp_scrape import MatchOdds

_NOW = datetime(2026, 6, 14, 18, 0, tzinfo=timezone.utc)


def _mo(home, away, ph=0.5, pd=0.3, pa=0.2):
    return MatchOdds(match_id="1", datetime_str="x", home_team=home, away_team=away,
                     odds_home=1 / ph, odds_draw=1 / pd, odds_away=1 / pa,
                     prob_home=ph, prob_draw=pd, prob_away=pa, overround=1.0, result=None)


def _event(home, away, commence_iso, over=1.9):
    return {"home_team": home, "away_team": away, "commence_time": commence_iso,
            "bookmakers": [{"key": "pinnacle", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": home, "price": 1.5}, {"name": "Draw", "price": 4.0},
                    {"name": away, "price": 6.0}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": over, "point": 2.5},
                    {"name": "Under", "price": 1.9, "point": 2.5}]}]}]}


class TestWindow:
    def _commence(self):
        return {
            ("A", "B"): _NOW + timedelta(minutes=10),    # in window (<=45)
            ("C", "D"): _NOW + timedelta(minutes=120),   # too early (>45)
            ("E", "F"): _NOW - timedelta(minutes=5),      # already kicked off
        }

    def test_matches_in_window(self):
        win = update.matches_in_window(self._commence(), _NOW, 45)
        assert win == {("A", "B")}

    def test_next_refresh_now_when_in_window(self):
        assert update.next_refresh_time(self._commence(), _NOW, 45) == _NOW

    def test_next_refresh_future_when_none_in_window(self):
        commence = {("C", "D"): _NOW + timedelta(minutes=120)}
        nxt = update.next_refresh_time(commence, _NOW, 45)
        assert nxt == _NOW + timedelta(minutes=120 - 45)

    def test_next_refresh_none_when_all_started(self):
        commence = {("E", "F"): _NOW - timedelta(minutes=5)}
        assert update.next_refresh_time(commence, _NOW, 45) is None


class TestQuotaGuard:
    def test_no_snapshot_fetches(self):
        assert update.should_refetch(None, 30) is True

    def test_recent_snapshot_skips(self):
        assert update.should_refetch(5.0, 30) is False

    def test_old_snapshot_fetches(self):
        assert update.should_refetch(45.0, 30) is True
        assert update.should_refetch(30.0, 30) is True   # exactly at the interval


class TestLoginDetection:
    def test_login_page_detected(self):
        assert update.looks_like_login('<form><input type="password" name="kennwort"></form>')

    def test_prediction_page_not_login(self):
        assert not update.looks_like_login('<table class="tippabgabe">...</table>')


class TestParseCommence:
    def test_iso_z(self):
        dt = update.parse_commence("2026-06-14T20:00:00Z")
        assert dt == datetime(2026, 6, 14, 20, 0, tzinfo=timezone.utc)

    def test_bad_input(self):
        assert update.parse_commence("not-a-date") is None


class TestCrontab:
    def test_line_shape(self):
        line = update.crontab_line(3, lead_minutes=45, min_interval=30, every_minutes=5,
                                   project_dir="/x", python="/x/.venv/bin/python")
        assert line.startswith("*/5 * * * * cd /x && /x/.venv/bin/python -m src.update")
        assert "--spieltag 3" in line and "--lead-minutes 45" in line and "--min-interval 30" in line

    def test_line_carries_autonomous_optin(self):
        # The generated cron command must self-opt-in, else the CLI guard refuses.
        assert "--i-know-this-is-autonomous" in update.crontab_line(2)


class TestAutonomousGuard:
    """src.update refuses to run autonomously without the explicit opt-in flag."""

    def test_cli_refuses_default_refresh_without_flag(self, monkeypatch, capsys):
        called = {"refresh": False, "watch": False}
        monkeypatch.setattr(update, "refresh", lambda *a, **k: called.__setitem__("refresh", True))
        monkeypatch.setattr(update, "watch", lambda *a, **k: called.__setitem__("watch", True))
        update.main(["--spieltag", "2"])
        out = capsys.readouterr().out
        assert "REFUSING" in out and "src.pipeline" in out
        assert called == {"refresh": False, "watch": False}   # nothing ran

    def test_cli_refuses_watch_without_flag(self, monkeypatch, capsys):
        ran = {"watch": False}
        monkeypatch.setattr(update, "watch", lambda *a, **k: ran.__setitem__("watch", True))
        update.main(["--spieltag", "2", "--watch"])
        assert "REFUSING" in capsys.readouterr().out and ran["watch"] is False

    def test_cli_runs_refresh_with_flag(self, monkeypatch):
        called = {}
        monkeypatch.setattr(update, "refresh", lambda st, **k: called.update(spieltag=st))
        update.main(["--spieltag", "5", "--i-know-this-is-autonomous"])
        assert called.get("spieltag") == 5

    def test_watch_function_refuses_without_optin(self):
        import pytest
        with pytest.raises(RuntimeError, match="i-know-this-is-autonomous"):
            update.watch(2)

    def test_crontab_prints_without_running(self, monkeypatch, capsys):
        # --crontab just prints a line (incl. the opt-in) and never runs refresh.
        monkeypatch.setattr(update, "refresh",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("refresh ran")))
        update.main(["--spieltag", "2", "--crontab"])
        out = capsys.readouterr().out
        assert "src.update" in out and "--i-know-this-is-autonomous" in out


class TestRefreshGating:
    """refresh() gates the credited fetch on window + quota, with mocked get_odds."""

    def _wire(self, monkeypatch, commence_min, snapshot_age):
        monkeypatch.setattr(update, "_now", lambda: _NOW)
        monkeypatch.setattr(update, "_scrape_with_auth_check", lambda md: [_mo("A", "B")])
        monkeypatch.setattr(update, "_snapshot_age_minutes", lambda md, now: snapshot_age)
        monkeypatch.setattr(update.snapshot, "record_match", lambda *a, **k: None)
        monkeypatch.setattr(update, "_write_exports", lambda *a, **k: (Path("x.csv"), Path("y.md")))
        calls = {"fresh": 0}
        ev = [_event("A", "B", (_NOW + timedelta(minutes=commence_min)).isoformat())]

        def fake_get_odds(*a, force_refresh=False, **k):
            if force_refresh:
                calls["fresh"] += 1
            return ev
        monkeypatch.setattr(update, "get_odds", fake_get_odds)
        return calls

    def test_skips_when_no_match_in_window(self, monkeypatch):
        calls = self._wire(monkeypatch, commence_min=120, snapshot_age=999)  # kickoff far away
        assert update.refresh(1, lead_minutes=45, min_interval=30) is None
        assert calls["fresh"] == 0                       # no credited fetch

    def test_skips_when_snapshot_recent(self, monkeypatch):
        calls = self._wire(monkeypatch, commence_min=10, snapshot_age=5)      # in window but fresh
        assert update.refresh(1, lead_minutes=45, min_interval=30) is None
        assert calls["fresh"] == 0                       # quota guard blocks the fetch

    def test_fetches_when_in_window_and_stale(self, monkeypatch):
        calls = self._wire(monkeypatch, commence_min=10, snapshot_age=60)     # in window + stale
        rows = update.refresh(1, lead_minutes=45, min_interval=30)
        assert rows is not None and len(rows) == 1
        assert calls["fresh"] == 1                       # exactly one credited fetch


class TestGetOddsForceRefresh:
    def test_force_refresh_bypasses_cache(self, monkeypatch, tmp_path):
        import src.odds.client as client
        monkeypatch.setattr(client, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(client, "_get_api_key", lambda: "KEY")
        calls = {"n": 0}

        class Resp:
            headers = {"x-requests-remaining": "100", "x-requests-used": "1", "x-requests-last": "1"}

            def raise_for_status(self):
                pass

            def json(self):
                return [{"id": "x"}]

        monkeypatch.setattr(client.requests, "get", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or Resp()))
        client.get_odds("soccer_x", markets="h2h")                       # fetch #1 (caches)
        client.get_odds("soccer_x", markets="h2h")                       # cache hit
        client.get_odds("soccer_x", markets="h2h", force_refresh=True)   # forced fetch #2
        assert calls["n"] == 2
