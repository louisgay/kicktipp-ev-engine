"""kicktipp-ev-engine interactive demo.

A live window onto the engine: change the market inputs and watch the Dixon-Coles
matrix, the expected-points table and the EV-max pick recompute through the real
src/ code.

Run locally:   streamlit run app/streamlit_app.py
(install with:  pip install -e ".[app]")
"""
from __future__ import annotations

import pathlib
import sys

# Streamlit puts the script's own dir on sys.path, not the repo root; add it so
# app (and src underneath) resolve when launched from anywhere.
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app import _compute as C  # noqa: E402

st.set_page_config(page_title="kicktipp-ev-engine", layout="wide")


@st.cache_data
def _snapshots():
    return C.load_snapshots()


def _heatmap(z, title, colorscale, highlight, texttemplate, home, away):
    """A goals-by-goals heatmap with the argmax cell outlined."""
    g = z.shape[0] - 1
    labels = [str(i) for i in range(g + 1)]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels, y=labels, colorscale=colorscale,
            text=z, texttemplate=texttemplate, textfont={"size": 11},
            hovertemplate=f"{home} %{{y}} - %{{x}} {away}<br>%{{z:.3f}}<extra></extra>",
            showscale=True, colorbar={"thickness": 12},
        )
    )
    hi_home, hi_away = highlight
    fig.add_shape(
        type="rect", x0=hi_away - 0.5, x1=hi_away + 0.5,
        y0=hi_home - 0.5, y1=hi_home + 0.5,
        line={"color": "#111", "width": 3},
    )
    fig.update_layout(
        title=title, xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
        yaxis={"autorange": "reversed"}, height=430,
        margin={"l": 60, "r": 10, "t": 50, "b": 40},
    )
    return fig


st.title("kicktipp-ev-engine")
st.markdown(
    "An expected-value-optimal score-prediction engine. The scoring is asymmetric "
    "(4 exact, 3 goal-difference, 2 winner, 0 wrong, and no goal-difference tier "
    "for draws), so the score that maximises expected points is usually not the "
    "most likely score. Everything below runs through the real engine."
)

st.subheader("From market odds to the EV-max pick")
snaps = _snapshots()

mode = st.radio(
    "Input", ["Pick a real match", "Enter your own odds"],
    horizontal=True, label_visibility="collapsed",
)

if mode == "Pick a real match":
    label = st.selectbox("Match (banked pre-kickoff market snapshot)", snaps["label"])
    row = snaps[snaps["label"] == label].iloc[0]
    p_home, p_draw, p_away = row.kt_home, row.kt_draw, row.kt_away
    p_over = row.ou_over_2_5
    home, away = row.home, row.away
else:
    c1, c2, c3, c4 = st.columns(4)
    oh = c1.number_input("Home odds", 1.05, 30.0, 1.90, 0.05)
    od = c2.number_input("Draw odds", 1.05, 30.0, 3.50, 0.05)
    oa = c3.number_input("Away odds", 1.05, 30.0, 4.20, 0.05)
    p_over = c4.slider("P(over 2.5 goals)", 0.10, 0.90, 0.52, 0.01)
    p_home, p_draw, p_away = C.devig_odds(oh, od, oa, method="shin")
    home, away = "Home", "Away"
    st.caption(
        f"De-vigged (Shin) fair probabilities: home {p_home:.1%}, "
        f"draw {p_draw:.1%}, away {p_away:.1%}"
    )

b = C.score_bundle(p_home, p_draw, p_away, p_over)
ml, pick = b["most_likely"], b["ev_pick"]

m1, m2, m3 = st.columns(3)
m1.metric("Most likely score", f"{ml[0]}-{ml[1]}", f"{b['most_likely_p']:.1%} probable")
m2.metric("EV-max pick", f"{pick[0]}-{pick[1]}", f"E[pts] = {b['ev_value']:.2f}")
m3.metric(
    "Cost of picking the likeliest score",
    f"-{b['ev_value'] - b['ev_of_most_likely']:.2f} pts",
    "they agree here" if b["agree"] else "they disagree",
    delta_color="off",
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        _heatmap(b["prob"], "P(scoreline): most likely", "Blues",
                 ml, "%{z:.1%}", home, away),
        width="stretch",
    )
with right:
    st.plotly_chart(
        _heatmap(b["ev"], "E[points]: what we pick", "YlOrRd",
                 pick, "%{z:.2f}", home, away),
        width="stretch",
    )

st.markdown(
    "The outlined cell is each panel's argmax. When they differ, the engine plays "
    "the right panel (maximum expected points), not the left (most likely). For "
    "example it prefers a decisive 1-0 over a likelier 1-1, because a non-exact "
    "draw scores only 2 with no goal-difference upside."
)

st.divider()
st.caption(
    "The score matrix is driven only by market odds. "
    "Source: github.com/louisgay/kicktipp-ev-engine"
)
