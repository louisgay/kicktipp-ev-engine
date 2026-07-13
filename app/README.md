# Interactive demo (Streamlit)

A live window onto the engine — change the market inputs and watch the
Dixon-Coles matrix, the expected-points table and the EV-max pick recompute
through the real `src/` code.

## Run locally

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501.

## What's inside

| Tab | What it shows |
|-----|----------------|
| **🎯 Pick explorer** | Odds → de-vigged probabilities → Dixon-Coles score matrix → EV table → EV-max pick, live. Side-by-side `P(scoreline)` vs `E[points]` heatmaps make the whole thesis clickable: the likeliest score usually isn't the pick. Works off a banked market snapshot **or** your own odds. |
| **👥 Field model** | The hierarchical, shrinkage-based model of how the (pseudonymised) pool picks — the input to the follow/deviate decision layer. |
| **🔬 The honest edge** | The behavioral-edge anchor: the leak→points regression (slope ≈ −1) *and* the per-opponent t-stats showing the effect isn't yet statistically detectable at this sample size. Deliberately honest. |

## Architecture

- **`app/_compute.py`** — pure compute, no Streamlit. Reuses the production
  primitives (`src/odds`, `src/scoring`, `src/field_model`,
  `analysis/behavioral_edge`); never forks a parallel model. Unit-tested in
  `tests/test_app_smoke.py`.
- **`app/streamlit_app.py`** — widgets and Plotly charts on top of `_compute`.

## Deploy (Streamlit Community Cloud)

1. Point a new app at this repo, main file `app/streamlit_app.py`.
2. It installs from the repo-root `requirements.txt` (`.[app]`).
3. No secrets needed — the demo runs entirely offline from the banked CSVs.
