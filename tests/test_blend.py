"""Tests for the kicktipp->sharp 1X2 blend (KicktippOddsModel, DIFF 2)."""

from __future__ import annotations

from src.models.odds_model import KicktippOddsModel

# Brazil-Morocco-shaped: kicktipp crushes the draw (0.10); sharp odds give it more.
_KT = (0.84, 0.10, 0.06)
_SHARP_H2H = {"home_team": "Brazil", "away_team": "Morocco", "h2h_odds": [1.70, 3.80, 5.50]}
_KEY = ("Brazil", "Morocco")


def _model(blend: float) -> KicktippOddsModel:
    m = KicktippOddsModel(blend_weight=blend, require_ou=False)
    m._probs_1x2[_KEY] = _KT
    m.load_sharp_1x2([_SHARP_H2H])
    return m


def test_blend_zero_is_identity():
    # Default behaviour must be unchanged: blend_weight=0 => pure kicktipp.
    assert _model(0.0)._blended_1x2(_KEY) == _KT


def test_sharp_stored_devigged_shin():
    s = _model(0.0)._sharp_1x2[_KEY]
    assert abs(sum(s) - 1.0) < 1e-6
    assert s[0] > s[1] > s[2]            # home > draw > away ordering preserved
    assert s[1] > _KT[1]                 # sharp rates the draw higher than kicktipp


def test_blend_raises_draw_toward_sharp():
    m = _model(0.65)
    b = m._blended_1x2(_KEY)
    s = m._sharp_1x2[_KEY]
    assert abs(sum(b) - 1.0) < 1e-9      # renormalised
    assert _KT[1] < b[1] < s[1]          # blended draw sits between kicktipp and sharp


def test_missing_sharp_falls_back_to_kicktipp():
    m = KicktippOddsModel(blend_weight=0.65, require_ou=False)
    m._probs_1x2[("X", "Y")] = (0.5, 0.3, 0.2)
    assert m._blended_1x2(("X", "Y")) == (0.5, 0.3, 0.2)


def test_predict_matrix_draw_rises_with_blend():
    # End-to-end: the reconstructed score matrix's P(draw) should rise when blending.
    base = _model(0.0).predict_score_matrix(*_KEY)
    blended = _model(0.65).predict_score_matrix(*_KEY)
    import numpy as np
    assert float(np.trace(blended)) > float(np.trace(base))
