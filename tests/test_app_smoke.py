"""Smoke tests for the Streamlit demo.

The `_compute` layer is pure and always tested. The Streamlit UI import is tested
only when the `[app]` extra is installed (it is in CI). This is the "deployed app
still imports on a fresh clone" guarantee that keeps a live link from 500-ing.
"""
import pytest


def test_compute_score_bundle_shapes():
    from app import _compute as C

    snaps = C.load_snapshots()
    assert len(snaps) > 0 and "label" in snaps

    row = snaps.iloc[0]
    b = C.score_bundle(row.kt_home, row.kt_draw, row.kt_away, row.ou_over_2_5)
    assert b["prob"].shape == (C.GRID + 1, C.GRID + 1)
    assert b["ev"].shape == (C.GRID + 1, C.GRID + 1)
    assert 0.0 <= b["ev_value"] <= 4.0
    # picking the likeliest score never beats the EV-max pick, by construction
    assert b["ev_value"] >= b["ev_of_most_likely"] - 1e-9


def test_compute_devig_normalises():
    from app import _compute as C

    ph, pd_, pa = C.devig_odds(1.90, 3.50, 4.20)
    assert abs(ph + pd_ + pa - 1.0) < 1e-6
    assert all(0 < p < 1 for p in (ph, pd_, pa))


def test_streamlit_app_imports():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    import app.streamlit_app  # noqa: F401  (importing runs the script body)
