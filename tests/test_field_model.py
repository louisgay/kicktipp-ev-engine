"""Tests for src/field_model.py - graceful degradation + type recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.field_model import FieldModel


def _picks(rows):
    return pd.DataFrame(rows, columns=["player", "spieltag", "match_index", "pick"])


class TestGracefulDegradation:
    def test_n0_equals_consensus(self):
        # No data: must collapse to the consensus pick (= field_picks_consensus).
        fm = FieldModel(_picks([]), {(1, 0): (0, 2)})        # consensus = away 0-2
        assert fm.pick_distribution("Anyone", 1, 0) == {(0, 2): 1.0}

    def test_for_consensus_n0_point_mass(self):
        fm = FieldModel(_picks([]), {})
        assert fm.pick_distribution_for_consensus("X", "home", (1, 0)) == {(1, 0): 1.0}
        assert fm.pick_distribution_for_consensus("X", "draw", (1, 1)) == {(1, 1): 1.0}

    def test_unknown_match_returns_none(self):
        assert FieldModel(_picks([]), {}).pick_distribution("X", 9, 9) is None

    def test_pop_follow_defaults_to_one(self):
        assert FieldModel(_picks([]), {}).pop_follow == 1.0


class TestTypeRecovery:
    def _model(self):
        rows = []
        for i in range(20):
            rows.append(("DrawLover", 1, i, "1-1"))   # always picks the draw 1-1
            rows.append(("Follower", 1, i, "1-0"))     # always picks the consensus
        cons = {(1, i): (1, 0) for i in range(20)}      # consensus = home win 1-0
        return FieldModel(_picks(rows), cons, follow_k=5, exact_k=3)

    def test_drawlover_recovered(self):
        d = self._model().pick_distribution_for_consensus("DrawLover", "home", (1, 0))
        assert max(d, key=d.get) == (1, 1) and d[(1, 1)] > 0.5

    def test_follower_recovered(self):
        d = self._model().pick_distribution_for_consensus("Follower", "home", (1, 0))
        assert max(d, key=d.get) == (1, 0) and d[(1, 0)] > 0.6

    def test_follow_rate_shrinkage_direction(self):
        fm = self._model()
        assert fm.follow_rate("DrawLover") < 0.3     # never matched consensus
        assert fm.follow_rate("Follower") > 0.7      # always matched
        # a player with NO history sits at the population mean (between the two)
        assert fm.follow_rate("Newcomer") == fm.pop_follow


class TestDrawShareShrinkage:
    """Empirical-Bayes draw_k: weakly-supported (noisy) draw_share estimates collapse
    toward pop_draw_share. Relationship/range assertions only - no hardcoded values."""

    def _noisy_field(self):
        # Many players, each deviating only a few times -> small n_dev per player, so
        # their raw draw-share is a noisy estimate the EB rule should shrink hard.
        rows, cons = [], {}
        for i in range(12):
            cons[(1, i)] = (1, 0)                              # consensus = home win
        # Build a field whose RAW draw-shares scatter widely but on tiny n_dev each.
        for k in range(12):
            for i in range(12):
                # player k deviates on match (k % 12); draws iff k is even
                if i == k:
                    rows.append((f"p{k}", 1, i, "1-1" if k % 2 == 0 else "0-1"))
                else:
                    rows.append((f"p{k}", 1, i, "1-0"))
        return FieldModel(_picks(rows), cons)

    def test_draw_k_is_empirical_bayes_by_default(self):
        fm = self._noisy_field()
        assert fm.draw_k > 0 and fm._draw_k_override is None    # EB-computed, finite, positive

    def test_explicit_override_respected(self):
        fm = FieldModel(_picks([("p", 1, 0, "1-1")]), {(1, 0): (1, 0)}, draw_k=99.0)
        assert fm.draw_k == 99.0

    def test_weakly_supported_draw_share_shrinks_toward_pop(self):
        fm = self._noisy_field()
        pop = fm.pop_draw_share
        # an even player drew on their one deviation -> raw draw_share = 1.0; the EB
        # shrinkage must pull the estimate strictly between its raw value and pop.
        raw = 1.0
        shrunk = fm.draw_share("p0")
        assert pop < shrunk < raw                              # collapsed toward pop, not at raw

    def test_fallback_to_follow_k_when_insufficient(self):
        # No deviations at all -> τ² inestimable -> draw_k falls back to follow_k.
        fm = FieldModel(_picks([("p", 1, i, "1-0") for i in range(5)]),
                        {(1, i): (1, 0) for i in range(5)}, follow_k=5.0)
        assert fm.draw_k == fm.follow_k


class TestSample:
    def test_sample_deterministic_at_n0(self):
        fm = FieldModel(_picks([]), {(1, 0): (0, 2)})
        rng = np.random.default_rng(0)
        assert fm.sample_pick("X", 1, 0, rng) == (0, 2)

    def test_sample_distribution_sums_to_one(self):
        d = self._modeldist()
        assert abs(sum(d.values()) - 1.0) < 1e-9

    def _modeldist(self):
        rows = [("P", 1, i, "2-1") for i in range(10)]
        fm = FieldModel(_picks(rows), {(1, i): (1, 0) for i in range(10)})
        return fm.pick_distribution_for_consensus("P", "home", (1, 0))


def _snaps(rows):
    from src.snapshot import COLUMNS
    return pd.DataFrame(rows, columns=COLUMNS)


class TestMarketModalAnchor:
    """from_disk must anchor follow/deviation on the MARKET-MODAL favourite for
    EVERY snapshotted matchday (oracle override where it exists) - so the model
    learns from all banked picks, not just the MD1 matches the oracle covers."""

    def test_market_modal_picks_favourite_scoreline(self, monkeypatch):
        from src import field_model as F
        # one strong home favourite + one strong away favourite
        snaps = _snaps([
            (1, 0, "H", "A", 0.80, 0.13, 0.07, 0.74, 0.16, 0.10, 0.50, None, None, "t"),
            (1, 1, "X", "Y", 0.08, 0.14, 0.78, 0.10, 0.18, 0.72, 0.48, None, None, "t"),
        ])
        monkeypatch.setattr("src.snapshot.load_history", lambda *a, **k: snaps)
        cons = F._market_modal_consensus()
        assert cons[(1, 0)][0] > cons[(1, 0)][1]      # home favourite -> home-win modal
        assert cons[(1, 1)][0] < cons[(1, 1)][1]      # away favourite -> away-win modal

    def test_market_modal_empty_on_no_snapshots(self, monkeypatch):
        from src import field_model as F
        monkeypatch.setattr("src.snapshot.load_history", lambda *a, **k: _snaps([]))
        assert F._market_modal_consensus() == {}

    def test_from_disk_ingests_all_matchdays(self):
        # REGRESSION for the MD1-only bug: with oracle covering only MD1, the model
        # used to drop every MD2+ pick (max n_follow == 8). The market-modal anchor
        # must lift coverage past one matchday and past 8 picks/player.
        fm = FieldModel.from_disk()
        matchdays = {st for st, _ in fm._consensus}
        assert len(matchdays) >= 2, f"anchor still covers only {matchdays}"
        assert max(fm._n_follow.values()) > 8, "model still capped at the MD1 fit"

    def test_oracle_overrides_market_where_present(self, monkeypatch):
        from src import field_model as F
        # market anchor says (1,0) everywhere; oracle (MD1 only) says a draw (1,1).
        monkeypatch.setattr(F, "_market_modal_consensus",
                            lambda: {(1, 0): (1, 0), (2, 0): (1, 0)})
        monkeypatch.setattr("src.oracle.consensus",
                            lambda st, *a, **k: (pd.DataFrame([{"match_index": 0}])
                                                 if st == 1 else pd.DataFrame()))
        monkeypatch.setattr("src.oracle.consensus_pick",
                            lambda st, mi: (1, 1) if st == 1 else None)
        monkeypatch.setattr("src.oracle.load_oracle", lambda: pd.DataFrame(
            columns=["source", "spieltag", "match_index", "pred_home", "pred_away", "kind"]))
        monkeypatch.setattr("src.oracle.correlate_per_player", lambda *a, **k: pd.DataFrame())
        picks = _picks([("P", 1, 0, "1-1"), ("P", 2, 0, "1-0")])
        monkeypatch.setattr("pandas.read_csv", lambda *a, **k: picks)
        fm = FieldModel.from_disk()
        assert fm._consensus[(1, 0)] == (1, 1)    # oracle overrode the market on MD1
        assert fm._consensus[(2, 0)] == (1, 0)    # market filled MD2 (no oracle there)
