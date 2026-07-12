# Knockout (a.PSO) scoring - engine change log & version map

> **TL;DR.** From the Round of 32 (MD11) kicktipp scores the **final result
> including extra time and the penalty shoot-out** (`a.PSO`). A match can therefore
> **never end in a draw**, and shoot-out goals inflate the score (2:2 a.e.t. + 5:4
> on penalties is recorded **7:6**). The group-stage engine assumed draws are
> ordinary outcomes, so it would have produced wrong picks for knockouts. We added a
> **second, knockout-only score path** and route to it automatically - the
> group-stage engine is **untouched and still live** for MD1-10.

Author note: this file documents *what changed in the scoring engine and why*, and
how the two versions coexist so we can show how scoring differed across the stages
of the tournament.

---

## 1. Both engines are preserved (nothing was overwritten)

The change is **additive**. There are now two score paths in one codebase, picked
automatically per matchday:

| Stage | Matchdays | Engine | Entry point | Draws? |
|---|---|---|---|---|
| **Group stage** | MD1-MD10 (72 matches) | original | `update._build_card` -> `scoring.kicktipp.optimal_prediction` | **yes** (1-1 etc. are valid, score 2 or 4) |
| **Knockouts (a.PSO)** | MD11+ (R32 -> final, 32 matches) | new | `update._build_card_knockout` -> `scoring.knockout.apso_optimal_prediction` | **no** (result is never a draw) |

Routing: `pipeline.run` sets `is_knockout = any(m.a_pso for ... in sel)` and calls
the matching builder. `MatchOdds.a_pso` is set by the scraper from the page layout
(see §3), so detection is data-driven - no hand-set matchday cutoff.

**Version history.**
- `src/scoring/kicktipp.py` (group-stage scorer) is **unchanged** - verify with
  `git diff e871ff3 -- src/scoring/kicktipp.py` (empty).
- The last **pure group-stage** repository state is commit **`e871ff3`** (and its
  ancestors). Checking it out reproduces the MD1-10 engine exactly as it scored
  those matchdays. Everything in §3 below is layered on top of that.
- The knockout path lives in the **new** file `src/scoring/knockout.py`, so the two
  scorers sit side by side and can be diffed/run independently.

To run a specific stage's engine: `python -m src.pipeline --spieltag N` (N≤10 ->
group engine; N≥11 -> a.PSO engine). The exported card header states which one ran
(`## EV-max card` vs `## EV-max card - a.PSO knockout`).

---

## 2. Why the group-stage engine is wrong for knockouts

Kicktipp tariff (unchanged): exact **4**, goal-difference **3**, tendency **2**,
wrong **0**; there is no GD tier for draws.

Under a.PSO the **recorded result is never a draw**. Consequences:

1. **A draw scoreline (1-1, 0-0, ...) scores 0, always** - its tendency can never
   match a decisive result. In the group stage a draw pick is often EV-max (it
   banks the draw-tendency 2 and can hit the exact 4); in knockouts it is strictly
   dominated. The old engine, run on a knockout, would still happily pick 1-1 on a
   coin-flip tie and score 0.
2. **The result's tendency = who advances.** Kicktipp posts a **2-way "advance"**
   price (no draw), and its devigged `q_home` is exactly *P(the result is a home
   win)*. This is a *sharper* and *different* number than the 90-minute home-win
   probability (e.g. Brazil v Japan: **93%** to advance vs **57%** to win in 90').
3. **Shoot-out scores are inflated** (e.g. 6:5), so exact-score prediction is
   near-hopeless; only tendency and goal-difference are realistically hittable.

---

## 3. What changed (file by file)

### 3.1 Scraper - `src/data/kicktipp_scrape.py`
- Knockout fixtures render odds inline in a **single** `td.quoten` cell as
  `1 <oh> X 0.00 2 <oa>` (draw price `0.00`), with an `a.PSO` `spielabschnitt`
  cell. The previous parser only understood the group-stage layouts
  (`td.kicktipp-wettquote` ×3 or `div.tippabgabe-quoten`), so knockout pages parsed
  to **0 matches** (which is why MD11 first looked "empty").
- Added: an inline `1 ... X ... 2 ...` regex branch; **a.PSO detection** -> new field
  **`MatchOdds.a_pso`**; and **2-way devig** when `a_pso and odds_draw == 0`
  (`prob_draw = 0`, `prob_home`/`prob_away` normalised over the two prices). The
  group-stage 3-way path is unchanged.

### 3.2 New scorer - `src/scoring/knockout.py`
Turns a 90-minute **regulation** score matrix into the **a.PSO final-result**
matrix, then reuses the existing, tested `optimal_prediction`. Clean factorisation:

- **Tendency = `q_home`** (kicktipp 2-way advance price), honoured **exactly** -
  the final matrix's home-win mass equals `q_home`.
- **Conditional scoreline shape = the regulation matrix.** Conditional on a side
  advancing, the scoreline is a regulation win for that side *or* a former draw
  resolved its way: a draw `(k,k)` becomes `(k+m, k)` / `(k, k+m)` with the deciding
  margin `m` from `DEFAULT_MARGIN_KERNEL` (`{1:.60, 2:.27, 3:.09, 4:.04}` - ET /
  shoot-out deciders are mostly 1-2 goals). Each side's shape is normalised, then
  mixed `final = q_home·H + (1-q_home)·A`.
- Result: `optimal_prediction(final)` always returns a **decisive** pick.

**Score-engine invariant preserved.** The regulation matrix is built the normal way
from the **3-way market odds** (Odds-API h2h + O/U -> `reconstruct_matrix`). The
kicktipp 2-way price only sets the *tendency*; it never reaches the regulation
matrix. So "the score matrix is driven only by market odds" still holds; the two
markets are used for two distinct, non-overlapping jobs (tendency vs scoreline
shape).

**Documented approximation.** A shoot-out really records an inflated score (6:5),
but for kicktipp points only tendency and goal-difference matter (plus the tiny,
near-unhittable exact term). We place a resolved tie at `(k+m, k)` - correct
tendency, correct GD `m` - which slightly **over-credits the *exact* tier** on 1-2
goal margins. The effect is small and nudges toward 1-goal-margin picks, which is
already where knockout EV-max lands. Tendency and GD (which dominate EV) are exact.

> Earlier draft used a different calibration (split the draw mass with a clamped
> `r_home` so total advance = `q_home`). It clamped pervasively because the Odds-API
> regulation odds are *softer* on favourites than kicktipp's direct advance market,
> leaving that sharper signal partly unused. The shipped factorisation (tendency
> from the advance market, shape from the regulation market) honours `q_home`
> exactly and removed the clamp.

### 3.3 Pipeline - `src/pipeline.py`, card builder - `src/update.py`
- `pipeline.run` auto-detects `a_pso` and calls `_build_card_knockout` (new, in
  `update.py`) instead of `_build_card`.
- `_build_card_knockout` builds the regulation matrix from sharp h2h + O/U, reads
  `q_home` from the kicktipp 2-way price, runs `apso_optimal_prediction`, and emits
  `CardRow`s with knockout fields (`a_pso`, `advance_home`).
- **Decision/relative-EV layer runs in knockout mode.** `_attach_relative_ev(...,
  knockout=True)` -> `choose_pick(..., knockout=True)` reuses the rank optimiser with
  the a.PSO (draw-free) matrix as the decision match and opponents' decision picks +
  future legs **projected onto decisive scorelines** (`rank_sim._decisive_dist`;
  future legs resolve any level shared outcome to the favourite). The `## Decision
  card` (EV-max | rank-opt | P(win)) therefore renders for knockouts too. **Caveat:**
  opponents have made no knockout picks yet, so this reuses their group-stage
  tendencies minus the impossible draws - P(win) (MD11 ≈ 0.73-0.76, all HOLD) is
  **indicative**, to be recalibrated once real knockout picks land.
- New knockout card format in `_format_card_md`:
  `## EV-max card - a.PSO knockout`, with columns
  `a.PSO pick | EV | Decorr | ΔEV | Advance (kt) | Reg 1X2 (90m) | O/U`.

### 3.4 Tests
- `tests/test_knockout_scoring.py` - no draw mass, sums to 1, `q_home` honoured
  exactly, picks always decisive, tendency follows the advance favourite, margin
  kernel effect.
- `tests/test_kicktipp.py::TestParseKnockoutPredictionPage` - the 2-way `td.quoten`
  layout parses, `a_pso=True`, `prob_draw=0`, 2-way devig correct.
- Full suite green (296 tests).

---

## 4. What did **not** change
- The group-stage scorer (`scoring.kicktipp`), the 3-way devig/reconstruct score
  engine, the field model, `rank_sim`, and the `--target 1` Decision card.
- The score-engine invariant (matrix driven only by market odds).
- The data/collection flow (picks, results backfill, snapshots).

---

## 5. Scoring contrast at a glance

| | Group stage (MD1-10) | Knockout a.PSO (MD11+) |
|---|---|---|
| Result can be a draw | **Yes** | **No** (incl. ET + penalties) |
| Draw pick (1-1, ...) | often EV-max; scores 2 or 4 | strictly dominated; scores 0 |
| kicktipp odds | 3-way 1X2 (home/draw/away) | 2-way "advance" (no draw) |
| Tendency probability | from blended 1X2 | = kicktipp advance price `q_home` |
| Scoreline matrix | blended kicktipp ⊕ sharp ⊕ O/U | regulation (3-way market) -> a.PSO transform |
| EV-max pick | may be a draw on coin-flips | always decisive |
| Rank/Decision (P(win)) layer | active, real field model (`--target 1`) | active in **knockout mode** - opponents projected onto decisive scorelines; P(win) indicative (no real knockout picks yet) |

---

_Last updated: 2026-06-28 (Round of 32 / MD11)._
