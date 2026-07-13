# Interactive demo (Streamlit)

A live window onto the engine: change the market inputs and watch the Dixon-Coles
matrix, the expected-points table and the EV-max pick recompute through the real
`src/` code.

Live: https://kicktipp-ev-engine.streamlit.app/

## Run locally

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501.

## What it shows

The pick explorer: odds to de-vigged probabilities to a Dixon-Coles score matrix
to the EV table to the EV-max pick, live. Side-by-side `P(scoreline)` and
`E[points]` heatmaps make the core idea concrete: the likeliest score usually is
not the pick. Works off a banked market snapshot or your own odds.

## Architecture

- `app/_compute.py`: pure compute, no Streamlit. Reuses the production primitives
  (`src/odds`, `src/scoring`); never forks a parallel model. Unit-tested in
  `tests/test_app_smoke.py`.
- `app/streamlit_app.py`: widgets and Plotly charts on top of `_compute`.

## Deploy (Streamlit Community Cloud)

1. Point a new app at this repo, main file `app/streamlit_app.py`.
2. It installs from the repo-root `requirements.txt` (`.[app]`).
3. No secrets needed: the demo runs entirely offline from the banked CSVs.
