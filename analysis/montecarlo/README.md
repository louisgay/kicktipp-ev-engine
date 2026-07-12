# Monte-Carlo fan chart - `analysis/montecarlo/`

A quant-finance-style **fan chart** for the Kicktipp race: re-simulate the whole
WC2026 tournament (104 matches) thousands of times, plot the cloud of cumulative-
points paths with percentile bands, overlay the **actually-realised** path, and show
that even a rational EV-max strategy has a wide spread. A teaching tool about
**variance dominating edge** over 104 matches - honest, not flattering.

> **Isolation invariant.** This package is analysis/viz, *not* the production
> pipeline. The dependency arrow is one-way: `analysis.montecarlo -> src` (we REUSE
> the production primitives; we never reimplement them). Nothing under `src/` imports
> this package - enforced by `tests/test_strategies_stats.py::test_src_does_not_import_analysis`.

## What it shows

- **Variance ≫ edge (~8:1).** EV-max's median edge over the field is ~**+4 pts**, but
  the relative-to-field band is ~**32 pts** (P10 **-12** -> P90 +20): even a rational
  player finishes *below* the field median >10% of the time.
- **Reality drew a bottom-decile path.** self's realised path (14 pts after 16
  matches) sits *below the counterfactual P10* (23) - a poor draw of a good strategy,
  not bad play.
- **Leader regression.** p10 leads at 24 (rank 1) on an always-fade strategy, which
  projects to a median final of ~54 pts and **rank 12** - a P99 path reverting to a
  -EV median. The tracked-player toggle plots its leader path regressing.

## Skill vs luck - the detectability claim (and why there is no "% luck" headline)

`stats.skill_luck_decomposition()` (CLI: `python -m analysis.montecarlo.stats`) leads
with the **Fundamental Law of Active Management** (Grinold), which uses self's
*measured* EV-max edge and is robust:

- relative edge **+4 pts/tournament**, relative band (P90-P10) ≈ **32** -> σ_rel ≈ 12.5
- **information ratio ≈ 0.32** per tournament - i.e. one World Cup's outperformance is
  only about **1σ** (a t-stat of ~0.3, not significant)
- implied **IC ≈ 0.031** (IR = IC·√breadth, breadth = 104 matches)
- **≈ 40 World Cups (relative) / ≈ 60 (absolute) for a t = 2 detection**

The headline is that number: **the edge is real but unmeasurable over a single
tournament.** This is a *breadth wall* - 104 matches simply is not enough independent
information for an IR of 0.32 to clear significance.

> **There is deliberately no "% luck" headline.** A law-of-total-variance split
> (`Var(obs)=Var(luck)+Var(skill)`) is computed, but **only as an internal diagnostic**,
> because no luck-share number is reliable here - and for the *same* breadth reason.
> The field model's per-player signal rests on only **~8 discriminating MD1 picks per
> player** (oracle consensus covers MD1; shrinkage w≈0.62), far too few to estimate any
> player's true skill. The function proves it with a **noise floor**: the observed
> follow-rate spread (0.079) sits *inside* the band 12 *truly identical* players would
> show from 8-pick estimates ([0.053, 0.118]). So the apparent skill spread is
> statistically indistinguishable from sampling noise, and any luck-share % (the raw
> figure is ~55%) is an artefact of that 8-pick fit, not a measurement. The same
> small-sample wall that makes the edge unmeasurable makes the split unquotable -
> which is the honest claim, so we quote the detectability numbers instead.
>
> The raw split and the per-player ranking are kept in the CLI as a *diagnostic only*,
> with the MD1-fit caveat: the ranking reflects **MD1 follow rates, not archetypes**
> (p10 ranks high because it *followed* consensus 7/8 in MD1; its fade is an MD2
> behaviour the field model never sees).

## Files

| File | Role |
|---|---|
| `engine.py` | Vectorised MC engine. Re-simulates 104 matches; odds matches sample from the Dixon-Coles matrix, far-future matches from `rank_sim._favourite_strengths`. REUSES `_favourite_strengths`, `_points_vec`, `reconstruct_matrix`, `FieldModel`, `oracle.consensus_pick`. Counterfactual / hybrid regimes; configurable tracked player. |
| `strategies.py` | The "me" strategies: **EV-max** (edge-free on generative legs, reconciles with `choose_pick`), **rank-relative** (a documented *qualitative proxy* of `choose_pick`, not its MC), **contrarian** (always-fade "p10"). |
| `stats.py` | Percentile bands, rank distribution, rival reference paths, the option-pricing readout, and the leader-regression projection. |
| `generate.py` | Sweeps the control matrix, writes `fanchart_data.json`, and injects it into the standalone `fanchart.html`. |
| `_fanchart_template.html` | Dark-mode quant-notebook template (Canvas + KaTeX); `/*__FANCHART_DATA__*/` placeholder. |
| `tests/` | Reproducibility, rank-dist, realised-path, **`choose_pick` bracketing reconciliation**, and **`_sim_future_correlated` equivalence** guards. |

## Run

```bash
# unit tests
python -m pytest analysis/montecarlo/tests/ -q

# quick numeric checks
python -m analysis.montecarlo.engine 8000        # smoke: all strategies × regimes
python -m analysis.montecarlo.stats  40000       # option-pricing + leader regression

# build the interactive visualiser (≈3 min at 80k sims)
python -m analysis.montecarlo.generate 80000
open analysis/montecarlo/fanchart.html           # generated, self-contained, opens with file://
```

`fanchart.html` / `fanchart_data.json` are **generated** (gitignored - the embedded
data goes stale each matchday); regenerate after banking new results.

## Reconciliation with production (`src.rank_sim.choose_pick`)

The engine **brackets** `choose_pick` monotonically in the number of edge-bearing
odds matches (hybrid, from today's standings), all at median rank 7:

| edge-bearing odds matches | P(top3) | source |
|---|---|---|
| 0 (fully edge-free knob) | ~5.6% | engine |
| 1 (lone decision match) | ~6.7% | **`choose_pick`** |
| 8 (all upcoming MD3) | ~9.3% | engine default |

## Modelling caveats (honest)

- **EV-max is edge-free on generative future legs** (me drawn from field-model-self,
  like every opponent) - a deterministic-optimal me vs a scattered field on 80 unseen
  matchups would manufacture a fantasy compounding edge (P(top3)->99%; the SYSTEM_MAP
  §6 flag-#4 saturation). Reproduces `_sim_future_correlated`'s treatment.
- **Knockout legs drop the 90' draw** (knockouts resolve) - an approximation; and
  `_favourite_strengths` may need recentering for knockouts (not addressed, per
  `rank_sim`'s `TODO(knockout)`).
- **Rank-relative is a qualitative proxy** of `choose_pick`'s regime (bold-when-behind),
  NOT its Monte Carlo. `z_chase` is a documented, tunable threshold (Phase-4 slider).
- Generative far-future matches use the synthetic `_favourite_strengths` stand-in
  (the softest assumption), and oracle consensus only covers MD1 (later matches fall
  back to the market matrix modal, as `update._attach_relative_ev` does).
