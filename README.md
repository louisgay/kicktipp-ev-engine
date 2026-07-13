# kicktipp-ev-engine

[![CI](https://github.com/louisgay/kicktipp-ev-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/louisgay/kicktipp-ev-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An expected-value-optimal score-prediction engine for a World Cup 2026 prediction
pool (Kicktipp), with a behavioral-edge study of opponents' picks.**

Quantitative, market-driven, and deliberately honest about what it can and cannot
prove. All personal data has been pseudonymised; no pool member is identifiable.

---

## What is this?

In a Kicktipp pool you predict the exact scoreline of each match. The scoring is
**asymmetric**: 4 points for the exact score, 3 for the right winner *and* goal
difference, 2 for just the right winner, 0 otherwise - and, crucially, **there is
no goal-difference tier for draws**. Because of that skew, the score that
*maximises expected points* is usually **not** the single most likely score: safe,
low-variance lines (1-0, 2-1, 1-1) win points more reliably than a "most probable
score" model would suggest.

This engine turns **market odds into fair probabilities** (removing the bookmaker
margin), builds a full **Dixon-Coles score matrix** for each match, and picks the
scoreline that **maximises expected points under the 4/3/2/0 rules**. On top of the
score engine sits a **decision layer** that models how the rest of the field will
pick, so it can decide *when to follow the crowd and when to deviate* - the lever
that actually moves you up the rankings.

## Key ideas

- **De-vig -> probabilities.** Strip the overround from 1X2 odds (normalisation or
  Shin's method) to recover fair win/draw/loss probabilities.
- **Score matrix.** Reconstruct `(λ_home, λ_away)` and build a Dixon-Coles
  double-Poisson matrix (low-score correlation correction).
- **EV-max under 4/3/2/0.** Choose the scoreline with the highest expected points -
  not the modal score. Draws are ordinary outcomes in the group stage; knockouts are
  scored after extra time + penalties, so a knockout pick is **never** a draw.
- **Field model.** A hierarchical, partial-pooling model of each opponent's pick
  tendencies, with shrinkage `w(n) = n/(n+k)` toward population priors as data
  accrues.
- **Decision layer.** Monte-Carlo `P(finish ≤ target rank)` under common random
  numbers, with a HOLD/DEVIATE gate hardened by empirical-Bayes shrinkage - chase
  when behind, protect/correlate when ahead.
- **Invariant:** the score matrix is driven by **market odds only**. Opponents,
  standings, and consensus feed the *decision* layer, never the scoreline model.

## Findings - stated honestly

> **The behavioral edge is a real mechanism, but it is not statistically detectable
> over a single tournament.** Both halves of that sentence matter.
>
> - **Mechanism (measurable now, no outcomes needed):** picking a draw forfeits the
>   3-point goal-difference tier, so on any match that isn't very even and low-scoring
>   a draw pick is *rules-based* -EV - a leak against the EV-max pick. Opponents leak
>   this systematically; the engine's own picks sit at ≈0 leak. One pool member is a
>   pronounced favourite-fader (deviates ~40% of the time, binomial p ≈ 0.0003 vs the
>   pool, and earns ~0.97 pts/pick on deviations vs ~1.81 on follows).
> - **Edge (what you can prove):** over one 104-match tournament the *predictive* edge
>   (forecasting scores better than the market) is ≈ 0 - everyone reads the same odds.
>   The *behavioral* differentials are suggestive but small; realised paired t-stats
>   are borderline and the confidence interval on any single-tournament edge is not
>   cleanly separable from luck.
>
> **Mechanisms can be real while the edge is statistically unmeasurable at small n -
> this repo quantifies that honestly rather than selling a backtest.**

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                       # ~300 tests, fully offline (~40s)

# teaching notebooks (offline; outputs are committed so they render on GitHub)
jupyter lab notebooks/01_how_a_pick_is_made.ipynb
jupyter lab notebooks/02_behavioral_edge.ipynb

# offline diagnostics
python -m analysis.behavioral_edge          # the EV-leak edge anchor
python -m analysis.montecarlo.generate      # rank fan-chart (writes gitignored HTML)

# regenerate historical match data (public sources) if you want the calibration test
python -m src.data.download && python -m src.data.clean
```

**Live mode is optional.** Odds refresh and read-only pool scraping need credentials -
copy `.env.example` to `.env` and fill in `ODDS_API_KEY`, `KICKTIPP_SESSION`,
`KICKTIPP_POOL`. Without them, live commands stop with a clear message; everything
else runs offline. No code path ever submits picks to Kicktipp.

## Repository map

```
src/odds/         de-vig + reconstruct market odds into a score matrix
src/scoring/      the 4/3/2/0 rules + EV-max pick (group + a.PSO knockout)
src/models/       Dixon-Coles / Poisson score models (+ odds-driven wrapper)
src/field_model   hierarchical partial-pooling model of opponents' picks
src/rank_sim      Monte-Carlo rank simulation + HOLD/DEVIATE decision gate
src/pipeline      manual end-to-end refresh (collect -> odds -> cards -> export)
src/bonus/        outright/bonus-question optimiser and tournament sims
analysis/         isolated viz/diagnostics (behavioral edge, fan chart, standings)
docs/SYSTEM_MAP.md   authoritative, anti-black-box architecture map - read this
```

See **`docs/SYSTEM_MAP.md`** for the full data flow and function inventory, and
**`docs/KNOCKOUT_APSO_ENGINE.md`** for the after-penalties knockout design.

## Data & licensing

Small, **pseudonymised** datasets ship so the tests and notebooks run offline;
third-party bookmaker odds are **not** redistributed. Provenance, licensing, and
regeneration steps are in **`data/README.md`**.

## About

Part of my quant portfolio. Questions or feedback welcome -
**Louis Gay**, [louis.gay.ch@gmail.com](mailto:louis.gay.ch@gmail.com).

## License

[MIT](LICENSE) © 2026 Louis Gay.
