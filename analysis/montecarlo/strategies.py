"""Selectable "me" (self) strategies for the Monte-Carlo fan chart.

A Strategy decides ONLY my own pick on each match (everyone else is the field
model). It never touches the model: it reuses the verified primitives
(``optimal_prediction``, ``expected_points``, ``_points_vec``) to turn a per-match
decision into the per-sim points I score against the sampled outcome.

Three strategies
----------------
(a) EV-MAX [default] - play the EV-max scoreline on matches that HAVE a real odds
    matrix (the genuine, realisable edge); be EDGE-FREE on odds-less generative
    legs (draw from field-model-self, like every opponent), reproducing
    ``rank_sim._sim_future_correlated``'s treatment. This is the honest default -
    a deterministic-optimal me vs a scattered field on 80 unseen future matchups
    would manufacture a fantasy edge (P(top3)->99%; SYSTEM_MAP §6 flag #4).

(b) RANK-RELATIVE HEURISTIC - *** AN APPROXIMATION of rank_sim.choose_pick, NOT
    the real function. *** A fast, qualitative per-match rule capturing its spirit
    (bold-when-behind / timid-when-ahead - Dubins-Savage; Browne; Bell-Cover), NOT
    its Monte Carlo. Running choose_pick's inner sim (~10^4 sims x 104 matches) per
    OUTER path is prohibitive, so we proxy the REGIME, not the probability:
    standardise the deficit to the top-`target` boundary by the remaining-match SD
    and switch to the decorrelating (contrarian) lever once safe play is unlikely
    to catch up. See ``RankRelativeStrategy`` for the rule + threshold derivation.

(c) CONTRARIAN [bonus] - the FAITHFUL "p10" archetype: ALWAYS fade the favourite,
    on every match, by playing the highest-EV scoreline whose tendency differs from
    the field/consensus tendency (EV decides draw-vs-underdog; not pinned to the
    draw). Verified against p10's real MD2 picks - it faded ESP-CPV (94% favourite;
    drew 2-2) and URU (73%; bet Saudi 1-0) and followed the favourite ONLY on the
    most contested match (NED-JPN, 57%): no selectivity by contestedness, a TOTAL
    fader. A deterministic negative-edge asymmetry. Over 104 matches it is
    structurally dominated (its best path loses to an average EV-max path;
    P(top3)≈0). The important, honest finding: p10's current lead (24 pts - a P99+
    path after a freak 5-draw matchday) is the realised right tail of a -EV strategy
    and is expected to regress hard toward its median over the remaining ~88 matches.
    The stats layer (Phase 3) makes that expected regression explicit.

This module imports from ``src`` only (one-way dependency); nothing in ``src``
imports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.rank_sim import _BASE_DIST, _VALS, _points_vec
from src.scoring.kicktipp import expected_points, optimal_prediction

# σ_match: SD of the baseline per-match pool points distribution, computed
# from rank_sim's own constants (NOT a magic number). _VALS=[0,2,3,4],
# _BASE_DIST=[0.46,0.28,0.10,0.16] -> mean 1.50, Var 2.33, SD ≈ 1.526.
_MEAN_MATCH = float((_VALS * _BASE_DIST).sum())
SIGMA_MATCH = float(math.sqrt(((_VALS - _MEAN_MATCH) ** 2 * _BASE_DIST).sum()))

# Default chase threshold (standardised-deficit units). At z, the chance of closing
# the deficit on neutral (correlated) play is ≈ Φ(-z/√2): z=1 -> ~24%. We deviate to
# bold/decorrelating play once safe play is more-likely-than-not to fall short
# (z>1). Tunable; exposed as a slider in the Phase-4 HTML.
Z_CHASE = 1.0

_SIGN = {"home": 1, "draw": 0, "away": -1}


@dataclass
class StrategyContext:
    """Everything a strategy needs to score MY pick on one match, vectorised over
    N sims. Built fresh by ``engine.simulate`` each match."""

    me_self: str                     # my player key ("self")
    kind: str                        # "odds" | "generative"
    matrix: np.ndarray               # decision matrix (odds matrix, or Poisson stand-in)
    cons_tend: str                   # consensus/favourite tendency
    cons_score: tuple[int, int]      # consensus scoreline (field anchor)
    ah: np.ndarray                   # (N,) sampled home goals
    aa: np.ndarray                   # (N,) sampled away goals
    me_run: np.ndarray               # (N,) my cumulative points BEFORE this match
    totals: np.ndarray               # (n_players, N) all totals BEFORE this match
    me_index: int
    n_remaining: int                 # matches left INCLUDING this one (>=1)
    target: int                      # rank target (3)
    field_points: object             # closure(player, tend, score, ah, aa) -> (N,)
    edge_free_odds: bool             # reconciliation knob (EV-max only)


# -- pick helpers (deterministic per match; reuse the verified scorer) --


def _evmax_pick(matrix: np.ndarray) -> tuple[int, int]:
    return tuple(optimal_prediction(matrix)[0])


def offtendency_evmax(matrix: np.ndarray, cons_tend: str, max_pred: int = 5) -> tuple[int, int]:
    """Highest-EV scoreline whose tendency DIFFERS from ``cons_tend`` - the
    contrarian lever (cheapest_decorrelation restricted to off-consensus
    tendencies). Falls back to the plain EV-max pick if there is no off-tendency
    scoreline (degenerate)."""
    cons_sign = _SIGN[cons_tend]
    g = matrix.shape[0]
    hi = min(max_pred, g - 1)
    best, best_ev = None, -1.0
    for i in range(hi + 1):
        for j in range(hi + 1):
            if ((i > j) - (i < j)) == cons_sign:
                continue
            ev = expected_points((i, j), matrix)
            if ev > best_ev:
                best_ev, best = ev, (i, j)
    return best if best is not None else _evmax_pick(matrix)


def _score_pick(pick: tuple[int, int], ah: np.ndarray, aa: np.ndarray) -> np.ndarray:
    return _points_vec(np.full(ah.shape, pick[0]), np.full(aa.shape, pick[1]), ah, aa)


# -- strategies ----------------------------------------------------------


class Strategy:
    name = "base"

    def me_increment(self, ctx: StrategyContext) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class EvMaxStrategy(Strategy):
    name = "evmax"

    def me_increment(self, ctx: StrategyContext) -> np.ndarray:
        if ctx.kind == "odds" and not ctx.edge_free_odds:
            return _score_pick(_evmax_pick(ctx.matrix), ctx.ah, ctx.aa)
        # edge-free: generative leg, OR upcoming odds under the reconciliation knob
        return ctx.field_points(ctx.me_self, ctx.cons_tend, ctx.cons_score, ctx.ah, ctx.aa)


class ContrarianStrategy(Strategy):
    name = "contrarian"

    def me_increment(self, ctx: StrategyContext) -> np.ndarray:
        # always fade the favourite, deterministically, on every match
        return _score_pick(offtendency_evmax(ctx.matrix, ctx.cons_tend), ctx.ah, ctx.aa)


class RankRelativeStrategy(Strategy):
    """Qualitative proxy for choose_pick's regime (NOT its Monte Carlo).

    Per match, per sim:
        boundary = the target-th highest total among the OTHER players (the cut line
                   I must exceed to sit in the top-`target`).
        D        = boundary - my_total          (>0 behind the cut, ≤0 in the top)
        z        = D / (√n_remaining · σ_match)  (standardised deficit)

    Rule:
        z ≤ z_chase  -> EV-MAX        (correlate: protects a lead, neutral at parity)
        z >  z_chase  -> DECORRELATING (off-consensus-tendency EV-max: bold catch-up)

    Lateness needs no separate term: n_remaining shrinks, so a fixed deficit's z
    grows over time - the rule flips to bold play only when behind AND late, which
    is exactly "deficit large relative to the shrinking remaining SD". This mirrors
    choose_pick's regime detector qualitatively (agree with EV-max at parity;
    deviate to take variance when clearly behind). It is a REGIME proxy, not a
    probability - do not read z as choose_pick's P(rank≤target).
    """

    name = "rank_relative"

    def __init__(self, z_chase: float = Z_CHASE):
        self.z_chase = z_chase

    def me_increment(self, ctx: StrategyContext) -> np.ndarray:
        # cut line = target-th highest of the OTHER players (exclude me)
        others = np.delete(ctx.totals, ctx.me_index, axis=0)        # (n-1, N)
        kth = others.shape[0] - ctx.target                          # target-th from top
        boundary = np.partition(others, kth, axis=0)[kth]           # (N,)
        deficit = boundary - ctx.me_run
        remaining_sd = math.sqrt(max(ctx.n_remaining, 1)) * SIGMA_MATCH
        z = deficit / remaining_sd
        chase = z > self.z_chase                                    # (N,) bold-play mask

        # comfortable branch = EV-max (odds: deterministic; generative: edge-free)
        if ctx.kind == "odds":
            safe_inc = _score_pick(_evmax_pick(ctx.matrix), ctx.ah, ctx.aa)
        else:
            safe_inc = ctx.field_points(ctx.me_self, ctx.cons_tend, ctx.cons_score, ctx.ah, ctx.aa)
        # chase branch = decorrelating (contrarian lever), deterministic
        bold_inc = _score_pick(offtendency_evmax(ctx.matrix, ctx.cons_tend), ctx.ah, ctx.aa)
        return np.where(chase, bold_inc, safe_inc)


_REGISTRY = {s.name: s for s in (
    EvMaxStrategy, ContrarianStrategy, RankRelativeStrategy)}


def make_strategy(name: str, *, z_chase: float = Z_CHASE) -> Strategy:
    """Factory used by ``engine.simulate``. ``rank_relative`` takes ``z_chase``."""
    if name == "rank_relative":
        return RankRelativeStrategy(z_chase=z_chase)
    if name not in _REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
