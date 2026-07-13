"""kicktipp-ev-engine — interactive demo.

A live, clickable window onto the real engine: change the market inputs and watch
the Dixon-Coles matrix, the expected-points table and the EV-max pick recompute
through the actual `src/` code. Three tabs:

  1. Pick explorer   — odds → fair probs → score matrix → EV-max pick, live.
  2. Field model     — how the rest of the pool picks (pseudonymised).
  3. The honest edge — the behavioral edge, and why it's real yet hard to prove.

Run locally:   streamlit run app/streamlit_app.py
(install with:  pip install -e ".[app]")
"""
from __future__ import annotations

import pathlib
import sys

# Streamlit puts the script's own dir on sys.path, not the repo root — add it so
# `app` (and `src`/`analysis` underneath) resolve when launched from anywhere.
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app import _compute as C  # noqa: E402

st.set_page_config(page_title="kicktipp-ev-engine", page_icon="⚽", layout="wide")


# ── cached wrappers ──────────────────────────────────────────────────────
@st.cache_data
def _snapshots():
    return C.load_snapshots()


@st.cache_data
def _field():
    return C.field_table()


@st.cache_data
def _edge():
    return C.edge_bundle()


def _heatmap(z, title, colorscale, highlight, texttemplate, home, away):
    """A goals×goals heatmap with the argmax cell outlined."""
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


# ── header ───────────────────────────────────────────────────────────────
st.title("⚽ kicktipp-ev-engine")
st.markdown(
    "**An expected-value-optimal score-prediction engine.** The scoring is "
    "asymmetric — 4 exact / 3 goal-difference / 2 winner / 0 wrong, and *no "
    "goal-difference tier for draws* — so the score that maximises expected points "
    "is usually **not** the most likely score. Everything below runs through the "
    "real engine. All player data is pseudonymised."
)

tab1, tab2, tab3 = st.tabs(
    ["🎯 Pick explorer", "👥 Field model", "🔬 The honest edge"]
)

# ── Tab 1 — pick explorer ────────────────────────────────────────────────
with tab1:
    st.subheader("From market odds to the EV-max pick — live")
    snaps = _snapshots()

    mode = st.radio(
        "Input", ["Pick a real match", "Enter your own odds"],
        horizontal=True, label_visibility="collapsed",
    )

    if mode == "Pick a real match":
        label = st.selectbox("Match (banked pre-kickoff market snapshot)",
                             snaps["label"])
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
            f"De-vigged (Shin) fair probs — home **{p_home:.1%}** · "
            f"draw **{p_draw:.1%}** · away **{p_away:.1%}**"
        )

    b = C.score_bundle(p_home, p_draw, p_away, p_over)
    ml, pick = b["most_likely"], b["ev_pick"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Most likely score", f"{ml[0]}-{ml[1]}", f"{b['most_likely_p']:.1%} probable")
    m2.metric("EV-max pick", f"{pick[0]}-{pick[1]}", f"E[pts] = {b['ev_value']:.2f}")
    m3.metric(
        "Cost of picking the likeliest score",
        f"−{b['ev_value'] - b['ev_of_most_likely']:.2f} pts",
        "they agree here" if b["agree"] else "they disagree → pick the right panel",
        delta_color="off",
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _heatmap(b["prob"], "P(scoreline) — most likely", "Blues",
                     ml, "%{z:.1%}", home, away),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            _heatmap(b["ev"], "E[points] — what we pick", "YlOrRd",
                     pick, "%{z:.2f}", home, away),
            width="stretch",
        )
    st.info(
        "The **outlined cell** is each panel's argmax. When they differ, the engine "
        "plays the **right** panel (max expected points), not the left (most likely) "
        "— e.g. it prefers a decisive 1-0 over a likelier 1-1, because a non-exact "
        "draw scores only 2 with no goal-difference upside.",
        icon="💡",
    )

# ── Tab 2 — field model ──────────────────────────────────────────────────
with tab2:
    st.subheader("How the rest of the pool picks")
    st.markdown(
        "A hierarchical (partial-pooling) model of each player's tendencies, shrunk "
        "toward the pool average by `w(n) = n / (n + k)` — so a player with few "
        "observed picks looks like the crowd until they earn their own estimate. "
        "The **decision layer** uses this to choose *when to follow the field and "
        "when to deviate*. Players are pseudonymised `p01…p11`; `self` is the engine."
    )
    df, pop = _field()
    st.caption(
        f"Pool baselines — follow-rate **{pop['pop_follow']:.1%}**, "
        f"draw-share **{pop['pop_draw_share']:.1%}**."
    )

    fig = go.Figure()
    fig.add_bar(x=df["player"], y=df["follow_rate"], name="follow-rate",
                marker_color="#3b6ea5")
    fig.add_bar(x=df["player"], y=df["draw_share"], name="draw-share",
                marker_color="#e08a3c")
    fig.add_hline(y=pop["pop_follow"], line_dash="dot", line_color="#3b6ea5",
                  annotation_text="pop follow")
    fig.update_layout(barmode="group", height=380, yaxis_tickformat=".0%",
                      margin={"t": 20, "b": 30}, legend={"orientation": "h"})
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        df.style.format({"follow_rate": "{:.1%}", "draw_share": "{:.1%}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        "**follow-rate** = how often the player takes the market-consensus scoreline. "
        "**draw-share** = of their deviations, how many are draws. High draw-share is "
        "the behavioral leak dissected in the next tab."
    )

# ── Tab 3 — the honest edge ──────────────────────────────────────────────
with tab3:
    st.subheader("The edge is real — and honest about its own limits")
    e = _edge()
    reg = e["reg"]

    st.markdown(
        "This is the part a reviewer should trust *because* it doesn't overclaim. "
        "The **behavioral edge** is measured without looking at results: for each "
        "opponent, how many expected points they *leak* by deviating from the "
        "EV-max pick (mostly by picking draws). If the mechanism is real, a player's "
        "leak should predict their realised under-performance — and it does:"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Leak → points regression slope", f"{reg['slope']:.2f}",
              f"95% CI [{reg['ci_lo']:.2f}, {reg['ci_hi']:.2f}]", delta_color="off")
    c2.metric("R²", f"{reg['r2']:.2f}", f"p = {reg['p_value']:.3f}", delta_color="off")
    c3.metric("Players", f"{reg['n']}", "one point each", delta_color="off")

    st.markdown(
        "A slope near **−1** means: *each expected point leaked ≈ one realised point "
        "lost.* The mechanism holds. **But** look at the per-opponent paired "
        "t-statistics — the honest catch:"
    )

    paired = e["paired"]
    fig = go.Figure()
    fig.add_bar(
        x=paired["opponent"], y=paired["t_now"],
        marker_color=["#2e7d32" if d else "#b0bec5" for d in paired["detectable_now"]],
        name="t now",
    )
    fig.add_hline(y=2.0, line_dash="dash", line_color="#c62828",
                  annotation_text="t = 2 (detectable)")
    fig.update_layout(height=360, margin={"t": 20, "b": 30},
                      yaxis_title="paired t-stat (self vs opponent)")
    st.plotly_chart(fig, width="stretch")

    n_detect = int(paired["detectable_now"].sum())
    st.warning(
        f"Only **{n_detect} of {len(paired)}** opponents show a statistically "
        "detectable edge at today's sample size (t ≥ 2, green bars). The effect is "
        "real and points the right way in aggregate, but a single tournament is too "
        "few matches to *prove* it opponent-by-opponent. Stating that plainly — "
        "**mechanism real, edge not yet statistically detectable at this n** — is the "
        "point, not a hedge.",
        icon="⚖️",
    )

    with st.expander("Per-player expected leak (no outcomes used)"):
        st.dataframe(
            e["leaks"].style.format({
                "leak_per_match": "{:.3f}", "from_draws": "{:.1f}",
                "from_other": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )

st.divider()
st.caption(
    "Invariant: the score matrix is driven **only** by market odds; opponents' picks "
    "and standings feed the decision layer, never the scoreline. "
    "Source: github.com/louisgay/kicktipp-ev-engine"
)
