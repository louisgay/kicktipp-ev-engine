"""Tests for the relative-EV card wiring in update.py (Prompt 3 Part C)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.field_model import FieldModel
from src.scoring.kicktipp import optimal_prediction
from src.update import CardRow, _attach_relative_ev, _candidate_picks, _format_card_md


def _matrix(lh, la, g=8):
    m = np.outer(poisson.pmf(np.arange(g + 1), lh), poisson.pmf(np.arange(g + 1), la))
    return m / m.sum()


def _row():
    mat = _matrix(1.8, 0.8)
    ev, evv = optimal_prediction(mat)
    return CardRow(index=0, home="A", away="B", kickoff="", kt=(0.7, 0.2, 0.1),
                   sharp=None, blended=(0.7, 0.2, 0.1), ou_over=None,
                   ev_pick=ev, ev=round(evv, 3), decorr_pick=(2, 1), delta_ev=-0.05,
                   ou_source="market", matrix=mat)


def test_candidate_picks_dedup_includes_ev_and_decorr():
    r = _row()
    cands = _candidate_picks(r)
    assert tuple(r.ev_pick) in cands and (2, 1) in cands
    assert len(cands) == len(set(cands))      # de-duplicated


def test_attach_relative_ev_sets_fields(monkeypatch):
    from src import oracle
    monkeypatch.setattr(oracle, "consensus_pick", lambda st, mi: (1, 0))   # consensus home win
    fm = FieldModel(pd.DataFrame(columns=["player", "spieltag", "match_index", "pick"]), {})
    rows = [_row()]
    _attach_relative_ev(rows, 1, {"self": 4.0, "o1": 6.0, "o2": 6.0}, "self",
                        horizon=10, field_model=fm)
    r = rows[0]
    assert r.rel_pick is not None and 0.0 <= r.rel_p_top <= 1.0
    # rel pick's match_ev cannot exceed the EV-max ev, so the EV cost is <= 0
    assert r.rel_delta_ev <= 1e-9
    # default target is P(win); P-of-1st and the gate constant are recorded
    assert r.rel_target == 1 and 0.0 <= r.rel_p_win <= 1.0
    assert r.rel_gate_z == 2.0


def test_attach_relative_ev_target_threads(monkeypatch):
    """--target 3 reaches choose_pick and is recorded on the row."""
    from src import oracle
    import src.rank_sim as rs
    monkeypatch.setattr(oracle, "consensus_pick", lambda st, mi: (1, 0))
    seen = {}
    real = rs.choose_pick

    def _spy(*a, **k):
        seen["target"] = k.get("target")
        return real(*a, **k)
    monkeypatch.setattr(rs, "choose_pick", _spy)
    fm = FieldModel(pd.DataFrame(columns=["player", "spieltag", "match_index", "pick"]), {})
    r = _row()
    _attach_relative_ev([r], 1, {"self": 4.0, "o1": 6.0, "o2": 6.0}, "self",
                        horizon=10, field_model=fm, target=3)
    assert seen["target"] == 3 and r.rel_target == 3


def test_format_md_shows_both_cards():
    r = _row()
    r.rel_pick, r.rel_p_top, r.rel_p_win, r.rel_delta_ev = (1, 0), 0.99, 0.68, -0.02
    r.rel_target, r.rel_gate_z, r.rel_discriminating = 1, 2.0, False
    md = _format_card_md(1, [r], datetime(2026, 6, 14, tzinfo=timezone.utc))
    assert "## EV-max card" in md and "## Decision card - target: top-1 (win)" in md
    assert "P(win)" in md and "68.0%" in md            # win odds surfaced

    # target=3 relabels the header
    r.rel_target = 3
    md3 = _format_card_md(1, [r], datetime(2026, 6, 14, tzinfo=timezone.utc))
    assert "target: top-3" in md3

    # without rel data the decision section is omitted
    r2 = _row()
    md2 = _format_card_md(1, [r2], datetime(2026, 6, 14, tzinfo=timezone.utc))
    assert "## EV-max card" in md2 and "Decision card" not in md2


# -- Part C: honest discrimination (no rank separation on chalk) -------


def _matrix2(lh, la, g=8):
    m = np.outer(poisson.pmf(np.arange(g + 1), lh), poisson.pmf(np.arange(g + 1), la))
    return m / m.sum()


def _scattered_fm(cs):
    cons = {(9, i): cs for i in range(10)}
    s = f"{cs[0]}-{cs[1]}"
    rows = [(f"sd{k}", 9, i, s if i < 6 else ("1-1" if i < 8 else "0-1"))
            for k in range(5) for i in range(10)]
    return FieldModel(pd.DataFrame(rows, columns=["player", "spieltag", "match_index", "pick"]),
                      cons, follow_k=0.01, exact_k=0.01)


def _cardrow(idx, mat):
    from src.decorrelate import cheapest_decorrelation
    ev, evv = optimal_prediction(mat)
    dp, dev, _ = cheapest_decorrelation(mat, ev)
    return CardRow(index=idx, home="A", away="B", kickoff="", kt=(.5, .3, .2), sharp=None,
                   blended=(.5, .3, .2), ou_over=None, ev_pick=ev, ev=round(evv, 3),
                   decorr_pick=dp, delta_ev=round(dev - evv, 3), ou_source="model", matrix=mat)


_BUBBLE = {"self": 8., "o1": 9., "o2": 9., "o3": 7., "o4": 7., "o5": 10., "o6": 6., "o7": 8.}


def test_chalk_non_discriminating_labels_evmax(monkeypatch):
    from src import oracle
    monkeypatch.setattr(oracle, "consensus_pick", lambda st, mi: (4, 0))
    r = _cardrow(0, _matrix2(4.0, 0.22))                       # ~95% favourite
    _attach_relative_ev([r], 1, _BUBBLE, "self", horizon=10, field_model=_scattered_fm((4, 0)))
    assert r.rel_discriminating is False
    assert tuple(r.rel_pick) == tuple(r.ev_pick) and r.rel_delta_ev == 0.0
    md = _format_card_md(1, [r], datetime(2026, 6, 14, tzinfo=timezone.utc))
    assert "HOLD · EV-max is rank-optimal" in md
    # a HOLD still reports how close the best challenger came (gate math on the row)
    assert "fails 2.0·SE gate" in md


def test_contested_midpack_does_not_spuriously_deviate(monkeypatch):
    # Hardened gate: a contested match from a MID-PACK position has a flat P(top3)
    # surface - no candidate beats EV-max beyond the CRN-paired noise floor - so the
    # gate must NOT flag a DEVIATE (the old global-spread test wrongly did). This is
    # the exact MD5 failure mode (sub-noise argmax at negative EV).
    from src import oracle
    monkeypatch.setattr(oracle, "consensus_pick", lambda st, mi: (1, 1))
    r = _cardrow(0, _matrix2(1.25, 1.05))                      # ~55%, contested
    _attach_relative_ev([r], 1, _BUBBLE, "self", horizon=10, field_model=_scattered_fm((1, 1)))
    assert r.rel_discriminating is False                      # flat surface -> no flag
    assert tuple(r.rel_pick) == tuple(r.ev_pick) and r.rel_delta_ev == 0.0


def test_no_deviate_when_dominated(monkeypatch):
    # No candidate with ΔP(top3) <= 0 may ever fire (the inequality diff>z·se>0 also
    # rules out strictly-dominated picks at negative EV). On any chalk match the
    # non-EV-max candidates lose P(top3); none may be flagged.
    from src import oracle
    monkeypatch.setattr(oracle, "consensus_pick", lambda st, mi: (3, 0))
    r = _cardrow(0, _matrix2(3.0, 0.4))
    _attach_relative_ev([r], 1, _BUBBLE, "self", horizon=10, field_model=_scattered_fm((3, 0)))
    assert tuple(r.rel_pick) == tuple(r.ev_pick)              # EV-max held; nothing dominated it


def _stub_choose_pick(ev_pick, dev_pick, *, dev_diff, dev_se, ev_p_top=0.40):
    """Return a choose_pick stub whose results encode a controlled rank gradient:
    a deviating candidate with paired Δp_top = dev_diff ± dev_se vs the EV-max pick.
    Robust (no RNG): isolates the gate's eligibility rule from Monte-Carlo tail noise."""
    from src.rank_sim import RankResult

    def _cp(matrix, consensus, candidates, *a, **k):
        ev = RankResult(pick=ev_pick, match_ev=2.0, p_rank1=0.1, p_top=ev_p_top,
                        median_rank=3.0, diff_vs_evmax=None, diff_se=None)
        dev = RankResult(pick=dev_pick, match_ev=1.9, p_rank1=0.1, p_top=ev_p_top + dev_diff,
                         median_rank=3.0, diff_vs_evmax=dev_diff, diff_se=dev_se)
        return [dev, ev]
    return _cp


def test_genuine_deviation_fires_above_floor(monkeypatch):
    # The hardening removes ONLY sub-noise flags: a real rank gradient (Δp_top far
    # beyond the CRN-paired floor, as far-behind-late bold play produces) must still
    # fire a DEVIATE away from EV-max.
    from src import oracle
    import src.rank_sim as rs
    monkeypatch.setattr(oracle, "consensus_pick", lambda st, mi: (1, 1))
    r = _cardrow(0, _matrix2(1.3, 1.1))
    ev = tuple(r.ev_pick); dev = (ev[0] + 1, ev[1] + 1)
    monkeypatch.setattr(rs, "choose_pick",
                        _stub_choose_pick(ev, dev, dev_diff=0.06, dev_se=0.01))  # 6σ
    _attach_relative_ev([r], 1, _BUBBLE, "self", horizon=2, field_model=_scattered_fm((1, 1)))
    assert r.rel_discriminating is True and tuple(r.rel_pick) == dev
    md = _format_card_md(1, [r], datetime(2026, 6, 14, tzinfo=timezone.utc))
    assert "DEVIATE" in md


def test_sub_noise_gradient_does_not_fire(monkeypatch):
    # A candidate whose Δp_top is within the paired floor (here 0.5σ) must NOT fire,
    # even at a positive point estimate - the exact MD5 spurious-flag case.
    from src import oracle
    import src.rank_sim as rs
    monkeypatch.setattr(oracle, "consensus_pick", lambda st, mi: (1, 1))
    r = _cardrow(0, _matrix2(1.3, 1.1))
    ev = tuple(r.ev_pick); dev = (ev[0] + 1, ev[1] + 1)
    monkeypatch.setattr(rs, "choose_pick",
                        _stub_choose_pick(ev, dev, dev_diff=0.005, dev_se=0.01))  # 0.5σ
    _attach_relative_ev([r], 1, _BUBBLE, "self", horizon=2, field_model=_scattered_fm((1, 1)))
    assert r.rel_discriminating is False and tuple(r.rel_pick) == ev
