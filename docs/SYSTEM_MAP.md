# SYSTEM_MAP - kicktipp-ev-engine

A complete, anti-black-box map of the repo and how predictions are computed.
Engine numbers in the worked examples are real outputs of the code, captured at
the matchday noted beside each. The banked data spans the group stage and the
early knockout rounds.

> Quant prediction framework for a FIFA World Cup 2026 Kicktipp pool (pool
> scoring, 11 players, `self` = SELF). Two engines: a **score engine** (market
> odds -> EV-max scoreline pick) and a **rank/relative-EV decision layer** (field
> model -> P(finish in target rank)).

---

## 1. REPO MAP

```
kicktipp-ev-engine/
├── README.md · pyproject.toml · .gitignore
├── config/config.yaml
├── docs/SYSTEM_MAP.md                 <- this file
├── data/
│   ├── opponents/{oracle.csv, picks.csv}        # tracked (pseudonymised)
│   └── history/{snapshots.csv, behavioral_edge_log.csv}   # tracked
├── src/
│   ├── pipeline.py        ── single on-demand refresh-everything + health-check (MANUAL entry)
│   ├── recommend.py       ── EV-max sheet (offline + --live run_matchday)
│   ├── update.py          ── AUTONOMOUS windowed/quota refresh + --watch/cron (opt-in guarded)
│   ├── rank_sim.py        ── Monte-Carlo rank simulator + choose_pick (relative-EV)
│   ├── field_model.py     ── hierarchical per-player pick model + gated source mixture
│   ├── oracle.py          ── German-media prediction store + consensus + field-tracking
│   ├── snapshot.py        ── append-only per-match history recorder
│   ├── calibration.py     ── sharp-odds calibration validation
│   ├── decorrelate.py     ── cheapest decorrelating pick at minimal EV cost
│   ├── bonus_relev.py     ── contested-bonus-question relative-EV scan (validation/next-edition)
│   ├── opponents.py       ── leaderboard -> picks.csv logger + profiling
│   ├── models/{base,poisson,dixon_coles,odds_model}.py
│   ├── odds/{client,devig,reconstruct,historical}.py
│   ├── scoring/kicktipp.py    ── pool scoring + optimal_prediction (EV-max)
│   ├── data/{clean,download,kicktipp_scrape,leaderboard}.py
│   └── bonus/{outright_odds,group_sim,bracket_sim,optimizer}.py
├── strategy_research/     ── dated journal entries + README index
└── tests/                 ── src suite (297 tests) + analysis suites
```

### `src/` one-liners

| File | Purpose |
|---|---|
| `pipeline.py` | **Manual** one-shot: collect picks/results -> refresh odds -> recompute cards -> snapshot -> export -> health report. No window/loop/cron. |
| `recommend.py` | `run_matchday` (live EV-max sheet); offline HTML mode; bonus summary; deadlines. |
| `update.py` | Autonomous windowed/quota refresh + `--watch`/cron. **Guarded** - refuses without `--i-know-this-is-autonomous`. |
| `rank_sim.py` | `simulate_rank(_multi)`, `choose_pick` (relative-EV under CRN), `remaining_match_count`. |
| `field_model.py` | `FieldModel` partial-pooling pick predictor; Tier-2 + gated source mixture. |
| `oracle.py` | oracle.csv store, `consensus`/`consensus_pick`, `correlate_per_player`. |
| `snapshot.py` | `record_match`/`backfill_results` -> snapshots.csv (full odds+result tuple). |
| `calibration.py` | Brier/log-loss/draw-reliability of sharp odds (validation only). |
| `decorrelate.py` | `cheapest_decorrelation`, matchday decorrelation report. |
| `bonus_relev.py` | Contested-bonus scan (no live caller; validation/next-edition). |
| `opponents.py` | `update_log`/`log_snapshot` -> picks.csv; profiling, readiness. |
| `scoring/kicktipp.py` | `points`, `expected_points`, `optimal_prediction` (EV-max). |
| `odds/devig.py` | `devig_normalise`, `devig_shin` (bisection), `devig_1x2`, `devig_over_under`. |
| `odds/reconstruct.py` | `reconstruct_lambdas`/`reconstruct_matrix`, `extract_odds_for_event`. |
| `odds/client.py` | `get_odds(... force_refresh=False)` (disk-cached, no TTL). |
| `models/odds_model.py` | `KicktippOddsModel` (live engine, blend); `OddsModel`/`EnsembleModel` (backtest/unused). |
| `models/{dixon_coles,poisson,base}.py` | DC/DP models (fitting = backtest-only); `_dc_tau` reused live; `ScorePredictor` ABC. |
| `data/kicktipp_scrape.py` | GET-only kicktipp scraping -> `MatchOdds`/`BonusQuestion`. |
| `data/leaderboard.py` | `scrape_leaderboard` -> `Fixture`/`PlayerRow`/`Leaderboard`. |
| `data/{clean,download}.py` | team-name normalisation, source downloads (backtest data). |
| `bonus/*` | outright odds, group/bracket MC sims, bonus optimizer. |

### `data/` layout

| Path | Tracked? | Holds |
|---|---|---|
| `data/opponents/oracle.csv` | tracked | German-media/tipster/AI per-match predicted scorelines. Tipster given names pseudonymised. |
| `data/opponents/picks.csv` | tracked | Player picks, one row per (matchday, player, match). Display names pseudonymised to `p01`-`p11` / `self`. |
| `data/history/snapshots.csv` | tracked | Per-match `(kicktipp, sharp, O/U, result, lead_min, updated_at)`. The durable market record. |
| `data/history/behavioral_edge_log.csv` | tracked | Appended t-stat log (per-player columns use the pseudonyms). |
| `data/raw/odds_cache/*.json` | not shipped | Raw bookmaker odds (The Odds API), not redistributed (see `data/README.md`). |
| `data/opponents/snapshots/*.html` | gitignored | Raw leaderboard HTML (embeds member IDs). |
| `data/raw/*` (else), `data/processed/` | gitignored | Regenerable via `src.data.download`. |
| `data/exports/` | gitignored | Dated refresh cards + `update.log` (runtime). |

---

## 2. MODULE + FUNCTION INVENTORY (highlights)

- **`scoring/kicktipp.py`** - `points(pred, actual)` 4/3/2/0 (no GD tier for draws); `expected_points`; `optimal_prediction(prob_matrix)` -> EV-max pick.
- **`odds/devig.py`** - `devig_normalise` (live kicktipp/O-U), `devig_shin` (sharp + backtest), `devig_1x2`, `devig_over_under`.
- **`odds/reconstruct.py`** - `reconstruct_lambdas(p_h,p_d,p_a,p_over_2_5=None,rho=-0.04,max_goals=8)`; `reconstruct_matrix`; `extract_odds_for_event(event, preferred_bookmakers=None)` (Pinnacle-led; O/U paired per (book,market,point)).
- **`models/odds_model.py`** - `KicktippOddsModel(rho,max_goals,ou_devig_method,require_ou,blend_weight=0)`: `load_kicktipp_odds`/`load_totals`/`load_sharp_1x2`/`_blended_1x2`/`predict_score_matrix`/`predict_lambdas`. `OddsModel`/`EnsembleModel`/`DoublePoissonModel`/`DixonColesModel` fitting = backtest-only.
- **`oracle.py`** - `consensus(spieltag,sources)` (empty-with-columns when no rows), `consensus_pick`, `field_picks_consensus`, `correlate_per_player`, `correlate_sources`.
- **`field_model.py`** - `FieldModel(...)`/`from_disk`; `follow_rate`/`draw_share`/`_exact_dist`; `pick_distribution_for_consensus`; `discriminating_count(s)`/`source_weights`/`_mixture_dist`; gated `pick_distribution`.
- **`rank_sim.py`** - `simulate_rank`, `simulate_rank_multi(...,field_model=None,fav_dist=None)`, `compare_picks`, `choose_pick(...,horizon,target=3,n_sims,...)` (library default `target=3`; the pipeline passes `target=1`), `remaining_match_count`, `_sim_future_correlated` (endogenous edge), `_favourite_strengths` (generative stand-in).
- **`pipeline.py`** - `run(spieltag, now=None)`, `health_report(...)`, `format_health`, `_find_prior_export`/`_load_prior_rows`, `_lead_minutes`, `_not_kicked_off`. `HealthRow`.
- **`update.py`** - `refresh`, `watch(...,i_know_autonomous=False)`, `crontab_line` (emits opt-in), pure helpers, `_build_card`/`_attach_relative_ev`/`_write_exports`/`_format_card_md`. `_AUTONOMOUS_REFUSAL` guard.
- **`snapshot.py`** - `record_match(..., lead_minutes_to_kickoff=None, ...)`, `backfill_results`, `load_history`. 14-col schema.

---

## 3. PREDICTION ENGINES - EXACT DATA FLOW

### 3a. Score engine (EV-max pick)
1. **kicktipp 1X2** (`MatchOdds`, parser-normalised) + **O/U 2.5** (`load_totals`, devig `normalise`).
2. **Sharp blend** - Pinnacle-led sharp h2h via `extract_odds_for_event` -> `load_sharp_1x2` (Shin) -> `_blended_1x2`:
   `blended = normalise(w·sharp_shin + (1-w)·kicktipp)`, `w=blend_weight=0.65`. `w≤0` or no sharp => pure kicktipp. Applied at 1X2 level only.
3. **`reconstruct_lambdas`** - Nelder-Mead fit of (supremacy, total) minimising squared error vs blended 1X2 + (when present) `(p_over-model_over)²`. **P(draw) is whatever a DC double-Poisson with the fitted (λ,μ,ρ) yields** - so the blend fixes draws *upstream* at the 1X2 level.
4. **Score matrix** - double-Poisson + DC τ correction on (0,0),(1,0),(0,1),(1,1), **ρ=-0.04 fixed** (live). O/U enters only via the fitted total.
5. **`optimal_prediction`** - argmax E[pool points] (Bayes under 4/3/2/0). Favourites -> 2-0/2-1/1-0 (tendency + goal-diff mass).
6. **Decorrelation** - highest-EV scoreline differing from EV-max, same tendency (2-pt floor). Post-processing; never touches probabilities.

### 3b. Rank / relative-EV decision layer
- **oracle** -> consensus tendency + modal scoreline; `correlate_per_player` = player×source tracking.
- **field_model** -> consensus prior + shrunk per-player signals (follow rate, draw-lean, exact scatter; `w(n)=n/(n+k)`); gated source mixture activates at `discriminating_count ≥ 8` (currently all players = 2 -> **Tier-2**).
- **choose_pick** -> samples decision-match outcome (our matrix) + opponents' picks (field model) + correlated future legs (`_sim_future_correlated`, shared outcome, **endogenous edge**: we EV-max, field follows consensus). CRN -> tight paired SE. Returns `RankResult`s by `p_top`. Each carries `p_rank1` (P-of-1st, target-independent) alongside `p_top` (P(rank≤target)).
- **Decision card** (`_attach_relative_ev`, formerly "relative-EV card") -> `rel_target` (the gate's rank target), `rel_pick`, `rel_p_top` (P(rank≤target)), `rel_p_win` (P-of-1st, always shown), `rel_delta_ev = match_ev(rel) - ev`, plus the top challenger's paired gate math (`rel_challenger`, `rel_challenger_gain`, `rel_challenger_se`, `rel_gate_z`). Rendered as a 3-column card: **EV-max** (always) | **Rank-opt** | **P(win)/ΔEV + HOLD/DEVIATE verdict with the gate math**.
- **Decision target (default = 1 = P(win)).** `python -m src.pipeline --spieltag N` defaults to `--target 1` (optimise the gate for *finishing 1st* - the live objective once top-3 is secured; `--target 3` selects the legacy P(rank≤3) band). The **library** defaults (`choose_pick`/`simulate_rank`) stay `target=3` so other callers/tests are unchanged; only the decision-card path defaults to 1. The gate itself is unchanged - switching the target only moves which boundary `diff_vs_evmax` is measured against; EV-max remains the action default and the two coincide at parity.

### 3c. The three inputs, separated

| Input | Engine/layer | Function | Influences |
|---|---|---|---|
| kicktipp 1X2 (+ sharp, O/U) | **Score engine** | `_blended_1x2` -> `reconstruct_*` -> matrix | **Match-outcome probabilities** |
| German-media (oracle) | Decision layer **only** | `consensus*`/`correlate_per_player` -> field model / `choose_pick` | opponents' picks; rank-optimal deviation |
| players' picks + standings | Decision layer **only** | `FieldModel`, `current_totals` | field signals; rank-sim starting totals |

**Verified by reading the code: the score matrix is driven ONLY by market odds (kicktipp ⊕ sharp + O/U). Oracle/picks/standings never reach `predict_score_matrix`/`reconstruct_*`.** The one benign cross-over: `_attach_relative_ev` uses the matrix's modal scoreline as a consensus proxy when the oracle has no pick.

### 3d. Worked example - Germany v Curaçao (MD2, first live run, 2026-06-14)
| Stage | Value |
|---|---|
| kicktipp odds H/D/A | 1.01 / 359 / 176 (overround 0.9986) |
| kicktipp probs | 0.9915 / 0.0028 / 0.0057 |
| sharp (Shin) | 0.9441 / 0.0371 / 0.0187 |
| **blended (w=0.65)** | **0.9631 / 0.0320 / 0.0049** |
| reconstructed λ | λ_home=4.022, λ_away=0.221 (fit_err 4.4e-10; p_over feed 0.791) |
| top scorelines | 4-0 16.0%, 3-0 15.9%, 5-0 12.9%, 2-0 11.9%, 6-0 8.6%, 1-0 5.8% |
| **EV-max** | **3-0**, EV 2.283, P(exact) 15.9% |
| decorrelated | 4-0, ΔEV -0.006 |
| relative-EV | 3-0, P(top3) **100% (SATURATED - see §6)**, ΔEV +0.00 |
| O/U source | market |

### 3e. Flow diagram
```
kicktipp 1X2 ─┐                                                      EV-max ─► decorr
sharp 1X2  ───┼─►_blended_1x2(w=.65)─►reconstruct_lambdas─►DC matrix(ρ=-.04)─►optimal_prediction
O/U 2.5    ───┘            (matrix passed in, never modified) │
                                                              ▼
oracle ───►consensus/correlate ─┐                      choose_pick ─► rel_pick, P(top3), ΔEV
picks  ───►FieldModel ──────────┼─► (decision layer)        ▲
standings ─►current_totals ─────┘            _sim_future_correlated (endogenous edge)
```

### 3f. Knockout (a.PSO) score path - `src/scoring/knockout.py`

From **MD11 (Round of 32)** kicktipp scores the **final result incl. extra time +
penalty shoot-out** (`a.PSO`): the result is **never a draw**, and shoot-out goals
inflate the score (2:2 a.e.t. + 5:4 pens = 7:6). Two pipeline consequences:

- **Scraper.** Knockout fixtures post **2-way "advance" odds** inline in one
  `td.quoten` cell as `1 <oh> X 0.00 2 <oa>` (draw price 0). `parse_prediction_page`
  reads this layout and sets **`MatchOdds.a_pso=True`**, `prob_draw=0`, with
  `prob_home`/`prob_away` devigged 2-way. (Group-stage 1X2 path unchanged.)
- **Score engine.** `pipeline.run` auto-detects `a_pso` and calls
  `_build_card_knockout` -> `apso_optimal_prediction`. The 90-min **regulation**
  matrix is built the normal way from the **3-way Odds-API h2h + O/U** (invariant
  preserved - kicktipp's 2-way never touches the regulation matrix). The a.PSO
  transform then factorises cleanly: **tendency = who advances = kicktipp 2-way
  price `q_home`** (honoured exactly), **conditional scoreline shape = regulation
  matrix** (a draw `(k,k)` resolves to `(k+m,k)`/`(k,k+m)` via `DEFAULT_MARGIN_KERNEL`;
  ET/PSO deciders are mostly 1-2 goals). `final = q_home·H + (1-q_home)·A`, then the
  standard `optimal_prediction`. Picks are always decisive (a draw can never be the
  a.PSO result). Documented approximation: shoot-out scores are collapsed to the
  deciding margin, so the *exact* tier is slightly over-credited on 1-2 goal
  margins; tendency + GD (which dominate EV) are exact.
- **Decision layer in knockout mode.** `_attach_relative_ev(..., knockout=True)` ->
  `choose_pick(..., knockout=True)`: the a.PSO (draw-free) matrix is the decision
  match, and opponents' decision picks + future legs are projected onto decisive
  scorelines (`rank_sim._decisive_dist`; future legs also resolve any level shared
  outcome to the favourite). So the `## Decision card` (P(win)) **does** render for
  knockouts (header `## EV-max card - a.PSO knockout` + `## Decision card`). **Caveat
  (`TODO(knockout)` partly open):** opponents have made no knockout picks yet, so the
  knockout opponent model reuses group-stage tendencies minus draws - P(win) is
  *indicative* (MD11 ≈ 0.73-0.76, all HOLD), to recalibrate once real knockout picks
  land.

---

## 4. OPERATIONAL / TIMING SURFACE

| Entry point | Trigger | Writes |
|---|---|---|
| **`python -m src.pipeline --spieltag N`** | **Manual, one-shot, NOW** (no window/loop/cron) | snapshots.csv, `data/exports/mdN_*.{csv,md}`, stdout cards + health |
| `recommend.run_matchday` / `--live` | Manual | snapshots.csv; stdout sheet |
| `update.refresh` / `--watch` / cron | **Autonomous, time-triggered** - **guarded** (`--i-know-this-is-autonomous`) | exports, snapshots.csv |
| `update --crontab` | prints cron line (carries the opt-in flag) | stdout |
| `opponents.update_log` | Manual | picks.csv |

- **(a) No code path submits to kicktipp.** All kicktipp access is read-only GET (`_fetch_page`). Picks are entered by hand.
- **(b) Autonomous/time-triggered paths:** `update.watch` (loop) and the cron line, plus the 45-min lead window + 30-min quota guard inside `refresh`. **All now refuse to run without `--i-know-this-is-autonomous`.** The manual flow is `src.pipeline`, which has no window/loop/cron dependency.

---

## 5. STATUS LEDGER

| Component | State | Why |
|---|---|---|
| Draw blend (w=0.65) | **ACTIVE** | Wired through pipeline/recommend/update; confirmed live (Germany 99.2%->96.3%). |
| Snapshot logging | **ACTIVE / populated** | `snapshots.csv` banked across the group stage and early knockouts (kicktipp + sharp + O/U + results). |
| `field_model` Tier-2 | **ACTIVE (fallback)** | All players discriminating-count = 2 < 8. |
| `field_model` gated mixture | **DORMANT** | Below the 8-match gate; flag #6 TODO left, not built. |
| `choose_pick` / Decision card | **ACTIVE** | Default `target=1`: the live metric is **P(win)** (≈0.67-0.69 mid-tournament - discriminating, not saturated). The old P(top3) band saturated near 1.0 once top-3 was secured (that's why the default moved to `target=1`); `--target 3` still reproduces it. EV-max stays the action default; honor a DEVIATE only when gated *and* positionally legible. See §6. |
| `bonus_relev` | **DORMANT** | No live caller; validation/next-edition. Bonus picks locked this edition. |

**Tests:** 297 passed, 1 skipped (~40s), plus the separate `analysis/` suites.

---

## 6. FLAGS - findings, surprises, latent bugs

1. **[FIXED 2026-06-14] Cross-book O/U leak in `extract_odds_for_event`** - `over_price`/`under_price` were never reset between books, so a stale Over could pair with another book's Under. Now scoped per (book, market, point); both must come from the same market or O/U is treated absent. (`odds/reconstruct.py`.)
2. **[FIXED 2026-06-14] `oracle.consensus` KeyError on unseen spieltag** - empty filter -> `pd.DataFrame([]).sort_values(["spieltag",...])` raised. Now returns an empty frame with columns; `consensus_pick` yields None and callers fall back. Found on the first live MD2 run (oracle has only MD1 data) - had silently skipped the relative-EV card.
3. **[FIXED 2026-06-14] snapshot first-insert FutureWarning** - string cols now object-dtype on first concat too.
4. **Relative-EV elevated P(top3) (managed).** P(top3) can sit high (~0.9) when the field is heterogeneous - disciplined followers genuinely out-rank draw-heavy faders over the long horizon. This is **legitimate, not the old saturation** (which pinned every candidate at 1.0 via a compounding endogenous edge - fixed by edge-free future legs + the empirical-odds future-leg bootstrap). Two further hardenings make the DEVIATE flag robust to it: (a) `field_model.draw_share` is shrunk with an **empirical-Bayes `draw_k`** (weakly-supported fades collapse toward `pop_draw_share`); (b) the gate in `update._attach_relative_ev` now fires a DEVIATE only when a candidate beats EV-max by more than the **CRN-paired noise floor** (`z·diff_se`, z=2), with a dominance guard - so a flat surface can no longer leak sub-noise argmax flags at negative EV. Net: invariant "trust the EV-max card when the surface is flat" is now **belt-and-suspenders** (the gate self-suppresses spurious flags) rather than load-bearing. The health report still WARNs at P(top)≥0.95 as a tripwire - now target-aware: at the default `target=1` a ≥0.95 reading means a near-locked first place (heads-up, not a regression); at `target=3` it keeps the old saturation-regression framing. (Open, separate: the fades remain individually Bonferroni-insignificant at this n - a real-but-small edge measured generator-independently by `analysis/behavioral_edge`.)
5. **`require_ou=False` on the live path** - missing O/U silently degrades to `ou_source="model"` (1X2-only), labelled but easy to miss; the health report now WARNs on it.
6. **`ρ=-0.04` hardcoded live** (not fitted) - intentional, but a constant assumption.
7. **Flag #6 (deferred):** the gated mixture lives in `field_model.pick_distribution` but `choose_pick`'s decision match still samples Tier-2; wiring it through is a TODO, dormant until a player crosses the 8-match gate.

The health report (`src.pipeline`) operationalises several of these: it WARNs on missing/empty scrape, sharp 0/N, O/U fallback, high `fit_err`, raw overround outside [0.98,1.06], unchanged-from-last (stale vs unmoved market), and resolved matches with no logged picks / no backfilled result.
