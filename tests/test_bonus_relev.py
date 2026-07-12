"""Tests for src/bonus_relev.py - contested-question flagging + deviation gate."""

from __future__ import annotations

from src.bonus_relev import analyse_bonus, analyse_question, format_report


class TestAnalyseQuestion:
    def test_contested_overconcentrated_deviates(self):
        # Group-D-like: weak favourite (55%); field over-concentrates -> DEVIATE.
        a = analyse_question({"Turkey": 0.55, "USA": 0.25, "Australia": 0.13, "Paraguay": 0.07})
        assert a["contested"] is True
        assert a["over_concentrated"] is True
        assert a["recommend"] == "DEVIATE"
        assert a["target"] == "USA"                 # the runner-up
        assert a["ev_cost"] == round(4 * (0.55 - 0.25), 3)
        assert a["separation_upside"] > 0

    def test_dominant_favourite_takes_favourite(self):
        a = analyse_question({"Spain": 0.97, "Cape Verde": 0.02, "Uruguay": 0.01})
        assert a["contested"] is False
        assert a["recommend"] == "favourite"
        assert a["target"] == "Spain" and a["ev_cost"] == 0.0

    def test_uncontested_above_threshold_takes_favourite(self):
        a = analyse_question({"Brazil": 0.70, "Morocco": 0.18, "Scotland": 0.12})
        assert a["contested"] is False
        assert a["recommend"] == "favourite"

    def test_single_option_no_deviation(self):
        a = analyse_question({"OnlyTeam": 1.0})
        assert a["recommend"] == "favourite"


class TestReport:
    def test_report_flags_only_contested(self):
        sources = {
            "Group D winner": {"Turkey": 0.55, "USA": 0.25, "AUS": 0.13, "PAR": 0.07},
            "Group H winner": {"Spain": 0.97, "CPV": 0.02, "URU": 0.01},
        }
        res = analyse_bonus(sources)
        assert res["Group D winner"]["recommend"] == "DEVIATE"
        assert res["Group H winner"]["recommend"] == "favourite"
        report = format_report(res)
        assert "Group D winner" in report and "DEVIATE" in report
