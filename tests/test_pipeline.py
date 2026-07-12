"""Tests for src/pipeline.py - the single on-demand refresh + health-check.

All network/IO is mocked: kicktipp scrape, get_odds (via _extract_events),
leaderboard, picks/snapshot CSVs. No live calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import pipeline
from src.update import CardRow


# -- lightweight fakes -------------------------------------------------


@dataclass
class FakeMatch:
    home_team: str
    away_team: str
    datetime_str: str = ""
    prob_home: float = 0.6
    prob_draw: float = 0.24
    prob_away: float = 0.16
    overround: float = 1.00


@dataclass
class FakeFixture:
    index: int
    home: str
    away: str
    result: tuple[int, int] | None = None
    group: str = "Group A"
    date: str = ""


@dataclass
class FakePlayer:
    name: str
    total: float


def _now():
    return datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


def _card_row(idx, home, away, *, sharp=(0.58, 0.25, 0.17), ou=0.52,
              ev_pick=(1, 0), ou_source="market", kt=(0.62, 0.22, 0.16),
              blended=(0.60, 0.24, 0.16)):
    return CardRow(index=idx, home=home, away=away, kickoff="2026-06-15T18:00:00+00:00",
                   kt=kt, sharp=sharp, blended=blended, ou_over=ou,
                   ev_pick=ev_pick, ev=1.6, decorr_pick=(2, 1), delta_ev=-0.02,
                   ou_source=ou_source, matrix=None)


# -- PART 3a: pipeline order with everything mocked --------------------


class TestPipelineOrder:
    def test_run_order_and_outputs(self, monkeypatch, tmp_path, capsys):
        calls = []

        matches = [FakeMatch("Mexico", "South Africa"), FakeMatch("France", "Colombia")]
        commence = {
            ("mexico", "south africa"): _now() + timedelta(hours=6),   # upcoming
            ("france", "colombia"): _now() - timedelta(hours=2),       # already kicked off
        }
        # Mexico = upcoming (no leaderboard result yet) -> included; France has
        # started (result present) -> excluded by the not-started selector.
        fixtures = [FakeFixture(0, "Mexico", "South Africa", result=None),
                    FakeFixture(1, "France", "Colombia", result=(2, 2))]
        players = [FakePlayer("self", 4.0), FakePlayer("p07", 6.0)]

        def fake_update_log(sts):
            calls.append("collect.update_log"); return 3

        class FakeLB:
            def __init__(self): self.fixtures = fixtures; self.players = players

        def fake_scrape_lb(*a, **k):
            calls.append("collect.leaderboard"); return FakeLB()

        def fake_backfill(st, fx, **k):
            calls.append("collect.backfill")

        def fake_scrape_auth(st):
            calls.append("refresh.kicktipp"); return matches

        def fake_extract(force_refresh):
            calls.append(f"refresh.get_odds(force={force_refresh})")
            return ([], [], commence)

        def fake_build_card(sel, sharp, totals, com, bw):
            calls.append("recompute.build_card")
            return [_card_row(i, m.home_team, m.away_team) for i, m in sel]

        def fake_attach(rows, st, totals, me, horizon, fm, *, target=1, knockout=False):
            calls.append("recompute.relative_ev")
            for r in rows:
                r.rel_pick, r.rel_p_top, r.rel_delta_ev = (1, 0), 0.27, -0.01
                r.rel_p_win, r.rel_target, r.rel_gate_z = 0.68, target, 2.0
                r.rel_discriminating = False

        def fake_write(st, rows, now):
            calls.append("export.write")
            return (tmp_path / "x.csv", tmp_path / "x.md")

        def fake_key(h, a):
            return (h.lower(), a.lower())

        recorded = []

        def fake_record(*a, **k):
            recorded.append(k)

        monkeypatch.setattr(pipeline, "update_log", fake_update_log, raising=False)
        monkeypatch.setattr("src.opponents.update_log", fake_update_log)
        monkeypatch.setattr("src.data.leaderboard.scrape_leaderboard", fake_scrape_lb)
        monkeypatch.setattr(pipeline.snapshot, "backfill_results", fake_backfill)
        monkeypatch.setattr(pipeline.snapshot, "load_history",
                            lambda *a, **k: pd.DataFrame(columns=["spieltag", "match_index", "result"]))
        monkeypatch.setattr(pipeline.snapshot, "record_match", fake_record)
        monkeypatch.setattr("src.opponents.load_picks",
                            lambda: pd.DataFrame(columns=["spieltag", "match_index"]))
        monkeypatch.setattr(pipeline, "_scrape_with_auth_check", fake_scrape_auth)
        monkeypatch.setattr(pipeline, "_extract_events", fake_extract)
        monkeypatch.setattr(pipeline, "_build_card", fake_build_card)
        monkeypatch.setattr(pipeline, "_attach_relative_ev", fake_attach)
        monkeypatch.setattr(pipeline, "_write_exports", fake_write)
        monkeypatch.setattr(pipeline, "_key", fake_key)
        monkeypatch.setattr(pipeline, "_config_blend_weight", lambda: 0.65)
        monkeypatch.setattr(pipeline, "_find_prior_export", lambda st: None)
        # field model + horizon are heavy/networked -> stub
        monkeypatch.setattr("src.field_model.FieldModel.from_disk", classmethod(lambda cls: object()))
        monkeypatch.setattr("src.rank_sim.remaining_match_count", lambda *a, **k: {"remaining": 60})

        rows, report = pipeline.run(1, now=_now())

        # COLLECT precedes REFRESH precedes RECOMPUTE precedes EXPORT.
        assert calls.index("collect.update_log") < calls.index("refresh.kicktipp")
        assert calls.index("refresh.kicktipp") < calls.index("recompute.build_card")
        assert calls.index("recompute.build_card") < calls.index("export.write")
        # force_refresh=True fires on the credited fetch.
        assert "refresh.get_odds(force=True)" in calls
        # Only the not-yet-kicked-off match survives selection.
        assert len(rows) == 1 and rows[0].home == "Mexico"
        # SNAPSHOT carried the lead-minutes (capture distance) for the upcoming match.
        assert recorded and recorded[0]["lead_minutes_to_kickoff"] == pytest.approx(360.0, abs=1.0)
        # Both cards printed.
        out = capsys.readouterr().out
        assert "EV-max card" in out and "Decision card" in out
        assert "Health / staleness" in out


# -- Fix 1: not-started selector (upcoming vs live vs finished) --------


class TestUpcomingSelector:
    def test_predicate_three_states_and_fallback(self):
        now = _now()
        fut, past = now + timedelta(hours=3), now - timedelta(hours=2)
        # upcoming: no result, future kickoff -> predict
        assert pipeline._is_upcoming(fut, None, now) is True
        # live: result present (recent kickoff) -> excluded, NOT re-predicted
        assert pipeline._is_upcoming(past, (0, 1), now) is False
        # finished: result present, commence already dropped off board (None) -> excluded
        assert pipeline._is_upcoming(None, (7, 1), now) is False
        # empty/failed leaderboard (no result row) -> commence fallback unchanged
        assert pipeline._is_upcoming(fut, None, now) is True       # future -> included
        assert pipeline._is_upcoming(past, None, now) is False     # past   -> excluded
        assert pipeline._is_upcoming(None, None, now) is True      # unknown kickoff -> included

    def test_run_excludes_live_and_finished_emits_only_upcoming(self, monkeypatch, tmp_path):
        matches = [FakeMatch("Spain", "Cape Verde"),      # upcoming
                   FakeMatch("Saudi Arabia", "Uruguay"),  # LIVE (result present, recent KO)
                   FakeMatch("Germany", "Curaçao")]       # finished (off board)
        commence = {
            ("spain", "cape verde"): _now() + timedelta(hours=4),     # future
            ("saudi arabia", "uruguay"): _now() - timedelta(minutes=30),  # recent KO (live)
            # Germany absent from commence -> dropped off the odds board
        }
        fixtures = [FakeFixture(0, "Spain", "Cape Verde", result=None),       # not started
                    FakeFixture(1, "Saudi Arabia", "Uruguay", result=(0, 1)),  # LIVE provisional
                    FakeFixture(2, "Germany", "Curaçao", result=(7, 1))]       # finished
        players = [FakePlayer("self", 4.0)]

        class FakeLB:
            def __init__(self): self.fixtures = fixtures; self.players = players

        emitted = {}

        def fake_build_card(sel, sharp, totals, com, bw):
            emitted["homes"] = [m.home_team for _, m in sel]
            return [_card_row(i, m.home_team, m.away_team) for i, m in sel]

        monkeypatch.setattr("src.opponents.update_log", lambda sts: 0)
        monkeypatch.setattr("src.opponents.load_picks",
                            lambda: pd.DataFrame(columns=["spieltag", "match_index"]))
        monkeypatch.setattr("src.data.leaderboard.scrape_leaderboard", lambda *a, **k: FakeLB())
        monkeypatch.setattr(pipeline.snapshot, "backfill_results", lambda *a, **k: None)
        monkeypatch.setattr(pipeline.snapshot, "load_history",
                            lambda *a, **k: pd.DataFrame(columns=["spieltag", "match_index", "result"]))
        monkeypatch.setattr(pipeline.snapshot, "record_match", lambda *a, **k: None)
        monkeypatch.setattr(pipeline, "_scrape_with_auth_check", lambda st: matches)
        monkeypatch.setattr(pipeline, "_extract_events", lambda force_refresh: ([], [], commence))
        monkeypatch.setattr(pipeline, "_build_card", fake_build_card)
        monkeypatch.setattr(pipeline, "_attach_relative_ev", lambda *a, **k: None)
        monkeypatch.setattr(pipeline, "_write_exports", lambda st, rows, now: (tmp_path / "x.csv", tmp_path / "x.md"))
        monkeypatch.setattr(pipeline, "_key", lambda h, a: (h.lower(), a.lower()))
        monkeypatch.setattr(pipeline, "_find_prior_export", lambda st: None)

        rows, _ = pipeline.run(2, now=_now())
        homes = [r.home for r in rows]
        assert homes == ["Spain"]                       # only the not-started match
        assert "Saudi Arabia" not in homes              # live -> NOT re-predicted
        assert "Germany" not in homes                   # finished -> excluded
        assert emitted["homes"] == ["Spain"]            # never even built for live/finished


# -- PART 3b: staleness / health report --------------------------------


def _prior(idx, kt, sharp, ou, ev_pick):
    d = {"match_index": str(idx),
         "kt_h": f"{kt[0]:.4f}", "kt_d": f"{kt[1]:.4f}", "kt_a": f"{kt[2]:.4f}",
         "ou_over_2_5": f"{ou:.4f}", "ev_pick": ev_pick}
    if sharp is None:
        d.update(sharp_h="", sharp_d="", sharp_a="")
    else:
        d.update(sharp_h=f"{sharp[0]:.4f}", sharp_d=f"{sharp[1]:.4f}", sharp_a=f"{sharp[2]:.4f}")
    return d


class TestHealthReport:
    def _empty(self):
        return pd.DataFrame(columns=["spieltag", "match_index", "result"])

    def test_unchanged_match_flagged(self):
        kt, sharp, ou = (0.62, 0.22, 0.16), (0.58, 0.25, 0.17), 0.52
        row = _card_row(0, "Mexico", "South Africa", kt=kt, sharp=sharp, ou=ou, ev_pick=(1, 0))
        prior = {0: _prior(0, kt, sharp, ou, "1-0")}
        rep = pipeline.health_report(
            1, [row], [(0, FakeMatch("Mexico", "South Africa"))],
            n_scraped=1, sharp_matched=1, fixtures=[], picks_df=self._empty(),
            snap_df=self._empty(), prior_rows=prior, prior_existed=True,
            picks_added=0, results_backfilled=0)
        unchanged = [r for r in rep if r.category == "unchanged-from-last"]
        assert len(unchanged) == 1
        assert "possibly stale" in unchanged[0].detail and "unmoved market" in unchanged[0].detail

    def test_moved_market_not_flagged_unchanged(self):
        prior = {0: _prior(0, (0.62, 0.22, 0.16), (0.58, 0.25, 0.17), 0.52, "1-0")}
        row = _card_row(0, "Mexico", "South Africa",
                        kt=(0.70, 0.18, 0.12), sharp=(0.66, 0.20, 0.14), ou=0.55)  # all moved
        rep = pipeline.health_report(
            1, [row], [(0, FakeMatch("Mexico", "South Africa"))],
            n_scraped=1, sharp_matched=1, fixtures=[], picks_df=self._empty(),
            snap_df=self._empty(), prior_rows=prior, prior_existed=True,
            picks_added=0, results_backfilled=0)
        assert not [r for r in rep if r.category == "unchanged-from-last"]

    def test_missing_ou_degraded(self):
        row = _card_row(0, "France", "Colombia", ou=None, ou_source="model")
        rep = pipeline.health_report(
            1, [row], [(0, FakeMatch("France", "Colombia"))],
            n_scraped=1, sharp_matched=1, fixtures=[], picks_df=self._empty(),
            snap_df=self._empty(), prior_rows={}, prior_existed=False,
            picks_added=0, results_backfilled=0)
        deg = [r for r in rep if r.category == "degraded-input"]
        assert any("ou_source=model" in r.detail for r in deg)

    def test_sharp_unmatched_and_overround_flagged(self):
        row = _card_row(0, "X", "Y", sharp=None)
        rep = pipeline.health_report(
            1, [row], [(0, FakeMatch("X", "Y", overround=1.12))],   # out of band
            n_scraped=1, sharp_matched=0, fixtures=[], picks_df=self._empty(),
            snap_df=self._empty(), prior_rows={}, prior_existed=False,
            picks_added=0, results_backfilled=0)
        details = " ".join(r.detail for r in rep)
        assert "sharp 1X2 unmatched" in details
        assert "overround" in details
        assert "matched 0/" in details   # run-level sharp 0/N warning

    def test_not_updated_when_resolved_match_has_no_picks(self):
        fixtures = [FakeFixture(0, "Mexico", "South Africa", result=(2, 0))]
        rep = pipeline.health_report(
            1, [], [], n_scraped=2, sharp_matched=0, fixtures=fixtures,
            picks_df=self._empty(),                        # no picks logged at all
            snap_df=self._empty(),                         # no result backfilled
            prior_rows={}, prior_existed=False, picks_added=0, results_backfilled=0)
        cats = {(r.category, r.scope) for r in rep if r.status == "WARN"}
        assert ("not-updated", "picks.csv") in cats
        assert ("not-updated", "snapshots.csv") in cats

    def test_saturated_relative_ev_warns(self):
        # Regression sentinel: a chosen relative-EV pick at P(top)>=0.95 must WARN.
        row = _card_row(0, "Mexico", "South Africa")
        row.rel_p_top = 0.97
        rep = pipeline.health_report(
            1, [row], [(0, FakeMatch("Mexico", "South Africa"))],
            n_scraped=1, sharp_matched=1, fixtures=[], picks_df=self._empty(),
            snap_df=self._empty(), prior_rows={}, prior_existed=False,
            picks_added=0, results_backfilled=0)
        sat = [r for r in rep if r.category == "relative-EV"]
        assert sat and "saturation" in sat[0].detail

    def test_clean_run_all_pass(self):
        kt, sharp, ou = (0.62, 0.22, 0.16), (0.58, 0.25, 0.17), 0.52
        row = _card_row(0, "Mexico", "South Africa", kt=kt, sharp=sharp, ou=ou)
        fixtures = [FakeFixture(0, "Mexico", "South Africa", result=(2, 0))]
        picks = pd.DataFrame([{"spieltag": 1, "match_index": 0}])
        snap = pd.DataFrame([{"spieltag": 1, "match_index": 0, "result": "2-0"}])
        rep = pipeline.health_report(
            1, [row], [(0, FakeMatch("Mexico", "South Africa"))],
            n_scraped=1, sharp_matched=1, fixtures=fixtures, picks_df=picks,
            snap_df=snap, prior_rows={}, prior_existed=False,
            picks_added=1, results_backfilled=1)
        assert not [r for r in rep if r.status == "WARN"]


class TestPriorExportDiscovery:
    def test_find_and_load_prior(self, tmp_path, monkeypatch):
        d = tmp_path / "exports"
        d.mkdir()
        (d / "md1_20260614T100000Z.csv").write_text(
            "match_index,kt_h,kt_d,kt_a,sharp_h,sharp_d,sharp_a,ou_over_2_5,ev_pick\n"
            "0,0.6200,0.2200,0.1600,0.5800,0.2500,0.1700,0.5200,1-0\n")
        (d / "md1_20260614T120000Z.csv").write_text(   # newer -> should win
            "match_index,kt_h,kt_d,kt_a,sharp_h,sharp_d,sharp_a,ou_over_2_5,ev_pick\n"
            "0,0.7000,0.1800,0.1200,0.6600,0.2000,0.1400,0.5500,2-0\n")
        monkeypatch.setattr(pipeline, "_EXPORT_DIR", d)
        prior = pipeline._find_prior_export(1)
        assert prior.name == "md1_20260614T120000Z.csv"
        rows = pipeline._load_prior_rows(prior)
        assert rows[0]["ev_pick"] == "2-0"
