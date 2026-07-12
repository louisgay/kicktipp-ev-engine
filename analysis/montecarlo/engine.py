"""Monte-Carlo "fan chart" engine for the Kicktipp race (ANALYSIS / VIZ ONLY).

This module lives OUTSIDE ``src/`` on purpose: it is a teaching / visualisation
tool, NOT part of the production prediction pipeline, and must NEVER be imported
by anything under ``src/``. The dependency arrow points one way only:
``analysis.montecarlo`` -> ``src`` (we REUSE the production primitives), never the
reverse.

What it does
------------
Re-simulate the ENTIRE WC2026 tournament ``N`` times over the full 104-match
horizon (72 group + 32 knockout) and record, per simulation, *my* (self's)
cumulative-points path and everyone's final totals -> my final rank. The point of
the exercise (see the fan chart) is that even an EV-max strategy has a wide spread
of outcomes: over 104 matches variance dominates the edge.

REUSE (imported, never reimplemented) from ``src``
--------------------------------------------------
- ``src.rank_sim._favourite_strengths`` - the recentred (~0.62) synthetic
  favourite-strength stand-in for odds-less future matches.
- ``src.rank_sim._points_vec`` - the verified vectorised pool scorer
  (4 exact / 3 GD-non-draw / 2 tendency / 0), identical rule to
  ``src.scoring.kicktipp.points``.
- ``src.scoring.kicktipp.optimal_prediction`` - EV-max pick from a score matrix.
- ``src.odds.reconstruct.reconstruct_matrix`` / ``_score_matrix_from_lambdas`` -
  the Dixon-Coles matrix used by the production score engine.
- ``src.field_model.FieldModel`` - opponents' per-player pick distributions
  (shrinkage ``w(n)=n/(n+k)``), built from the post-MD2 banked picks.
- ``src.oracle.consensus_pick`` - the German-media consensus the field perceives
  (only MD1 is populated; later matches fall back to the market matrix modal,
  exactly as ``src.update._attach_relative_ev`` does).

The ONE thing not imported but replicated (flagged honestly): the SHARED-outcome
-per-match correlation structure. It is *implemented inside*
``rank_sim._sim_future_correlated``, which returns only summed final totals - it
does not expose the per-match increments this fan chart needs. So we reproduce the
exact same mechanism (one outcome sampled per match; ALL players scored against
that single outcome via ``_points_vec``; same ``lam_h=0.8+2.2*p_fav``,
``lam_a=1.2-0.8*p_fav`` favourite mapping; same field-model consensus anchoring)
rather than calling it. We reuse its atoms (``_favourite_strengths``,
``_points_vec``) and reproduce the loop; we do not invent a parallel model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# -- reused production primitives (one-way dependency: analysis -> src) --
from src import oracle
from src.field_model import FieldModel
from src.odds.reconstruct import reconstruct_matrix
from src.rank_sim import (
    GROUP_STAGE_MATCHES,
    TOURNAMENT_MATCHES,
    _favourite_strengths,
    _points_vec,
)
from analysis.montecarlo.strategies import (
    SIGMA_MATCH,
    Z_CHASE,
    StrategyContext,
    make_strategy,
)

_ROOT = Path(__file__).resolve().parents[2]
_PICKS = _ROOT / "data" / "opponents" / "picks.csv"
_SNAPSHOTS = _ROOT / "data" / "history" / "snapshots.csv"

SELF = oracle.SELF  # "self"
MATCHES_PER_MD = 8  # WC2026 group stage: 8 matches per matchday
_BLEND_W = 0.65     # production blend weight (config.yaml field_model is separate)
_REMAINING_SD_PER_MATCH = SIGMA_MATCH  # per-match points SD (from strategies, ≈1.53)


# -- tournament structure -----------------------------------------------


@dataclass
class MatchSlot:
    """One of the 104 tournament match slots, in play order."""

    gindex: int                      # global match index, 0-based
    spieltag: int | None             # matchday (group only), else None
    label: str                       # "MD1 m0 Mexico v South Africa" / "GRP m24" / "KO m80"
    kind: str                        # "odds" | "generative"
    is_knockout: bool
    matrix: np.ndarray | None = None             # odds matches only (P(i,j))
    consensus: tuple[str, tuple[int, int]] | None = None  # (tend, score), odds matches
    result: tuple[int, int] | None = None        # real outcome if played
    real_points: dict[str, float] = field(default_factory=dict)  # player -> real pts if played

    @property
    def played(self) -> bool:
        return self.result is not None


def _tend(a: int, b: int) -> str:
    return "home" if a > b else ("draw" if a == b else "away")


def _blended_probs(row) -> np.ndarray:
    """Blend kicktipp + sharp 1X2 exactly as the production score engine does
    (``blended = normalise(w·sharp + (1-w)·kicktipp)``, ``w``=blend_weight; pure
    kicktipp when sharp is absent). The snapshot already stores DEVIGGED probs."""
    kt = np.array([row.kt_home, row.kt_draw, row.kt_away], dtype=float)
    if pd.isna(row.sharp_home):
        return kt / kt.sum()
    sharp = np.array([row.sharp_home, row.sharp_draw, row.sharp_away], dtype=float)
    bl = _BLEND_W * sharp + (1 - _BLEND_W) * kt
    return bl / bl.sum()


def _load_real(picks_df: pd.DataFrame) -> tuple[dict[int, dict[str, float]], dict[int, tuple[int, int]]]:
    """Per-match real points-by-player and the realised scoreline, keyed by global
    match index, from the banked picks.csv (the deterministic past)."""
    real_pts: dict[int, dict[str, float]] = {}
    real_res: dict[int, tuple[int, int]] = {}
    for r in picks_df.itertuples(index=False):
        if pd.isna(r.points) or pd.isna(r.result):
            continue
        g = (int(r.spieltag) - 1) * MATCHES_PER_MD + int(r.match_index)
        real_pts.setdefault(g, {})[r.player] = float(r.points)
        try:
            ah, ab = (int(x) for x in str(r.result).split("-"))
            real_res[g] = (ah, ab)
        except ValueError:
            pass
    return real_pts, real_res


def build_tournament(horizon: int = TOURNAMENT_MATCHES) -> list[MatchSlot]:
    """Build the ordered 104- (or 72-) match list.

    - Matches with cached odds (snapshots.csv: MD1-MD3 so far) sample their outcome
      from the reconstructed Dixon-Coles matrix. Played matches also carry the real
      result + real per-player points (for hybrid mode + the realised overlay).
    - All other matches are GENERATIVE stand-ins (``_favourite_strengths``); the
      favourite strength is drawn per-match at simulation time.
    - Group stage = first 72 matches; matches 72..103 are knockouts (draw-at-90'
      dropped at sim time - see ``_sample_generative``).
    """
    picks = pd.read_csv(_PICKS) if _PICKS.exists() else pd.DataFrame()
    snap = pd.read_csv(_SNAPSHOTS) if _SNAPSHOTS.exists() else pd.DataFrame()
    real_pts, real_res = _load_real(picks) if len(picks) else ({}, {})

    # index odds rows by global match index
    odds_by_g: dict[int, object] = {}
    if len(snap):
        for r in snap.itertuples(index=False):
            g = (int(r.spieltag) - 1) * MATCHES_PER_MD + int(r.match_index)
            odds_by_g[g] = r

    slots: list[MatchSlot] = []
    for g in range(horizon):
        is_ko = g >= GROUP_STAGE_MATCHES
        if g in odds_by_g:
            r = odds_by_g[g]
            probs = _blended_probs(r)
            ou = None if pd.isna(r.ou_over_2_5) else float(r.ou_over_2_5)
            matrix = reconstruct_matrix(probs[0], probs[1], probs[2], ou)
            # consensus the field perceives: oracle (MD1 only) else market matrix modal
            cp = oracle.consensus_pick(int(r.spieltag), int(r.match_index))
            if cp is None:
                gsz = matrix.shape[0]
                fi = int(matrix.argmax())
                cp = (fi // gsz, fi % gsz)
            slot = MatchSlot(
                gindex=g, spieltag=int(r.spieltag),
                label=f"MD{int(r.spieltag)} m{int(r.match_index)} {r.home} v {r.away}",
                kind="odds", is_knockout=is_ko, matrix=matrix,
                consensus=(_tend(*cp), tuple(cp)),
                result=real_res.get(g), real_points=real_pts.get(g, {}),
            )
        else:
            slot = MatchSlot(
                gindex=g, spieltag=(g // MATCHES_PER_MD + 1) if not is_ko else None,
                label=f"{'KO' if is_ko else 'GRP'} m{g}",
                kind="generative", is_knockout=is_ko,
            )
        slots.append(slot)
    return slots


# -- outcome / pick sampling (shared-outcome-per-match correlation) ------


def _poisson_matrix(lam_h: float, lam_a: float, g: int = 8) -> np.ndarray:
    """Plain independent-Poisson score matrix (matches the GENERATIVE outcome
    sampler, which uses plain Poisson - no DC τ). Used only to pick the EV-max
    scoreline for an odds-less match."""
    ih = np.array([math.exp(-lam_h) * lam_h ** k / math.factorial(k) for k in range(g + 1)])
    ia = np.array([math.exp(-lam_a) * lam_a ** k / math.factorial(k) for k in range(g + 1)])
    return np.outer(ih, ia)


def _sample_generative(p_fav: float, is_knockout: bool, rng, n: int):
    """One shared outcome per generative match, for all N sims (reproducing the
    structure of ``rank_sim._sim_future_correlated``). Favourite is "home".

    KNOCKOUT APPROXIMATION: a 90' draw is dropped (knockouts resolve), so draws are
    nudged to a favourite (home) win by +1 goal. This is a deliberate modelling
    approximation, flagged here. CAVEAT: ``_favourite_strengths`` is centred for the
    GROUP stage; it may need recentering for knockouts (sharper favourites, no
    draw-incentive). Not addressed here (consistent with rank_sim's TODO(knockout)).
    """
    lam_h, lam_a = 0.8 + 2.2 * p_fav, 1.2 - 0.8 * p_fav
    ah = rng.poisson(lam_h, n)
    aa = rng.poisson(lam_a, n)
    if is_knockout:
        draw = ah == aa
        ah = np.where(draw, ah + 1, ah)  # resolve to a home (favourite) win
    cons_score = (int(round(lam_h)), int(round(lam_a)))
    return ah, aa, ("home", cons_score), (lam_h, lam_a)


# -- configuration + result containers -----------------------------------


@dataclass
class SimConfig:
    n_sims: int = 100_000
    seed: int = 0
    horizon: int = TOURNAMENT_MATCHES           # 104 (full) or GROUP_STAGE_MATCHES (72)
    regime: str = "counterfactual"              # "counterfactual" | "hybrid"
    strategy: str = "evmax"                     # "evmax" | "rank_relative" | "contrarian"
    z_chase: float = Z_CHASE                    # rank-relative chase threshold (Phase-4 slider)
    target: int = 3
    me_player: str = SELF                       # the strategy-driven tracked player (default self).
    #   Set to e.g. "p10" to project a rival forward under a strategy (leader-regression readout).
    track: str = "me"                           # "me" (light: only my path) | "all" (rival ref lines)
    fav_dist: tuple[float, float] | None = None  # override _favourite_strengths range
    # Reconciliation knob: when True, the EV-max strategy is edge-free on UPCOMING
    # odds matches too (me draws from field-model-self there as well), so the
    # engine carries NO forward edge at all. The engine then BRACKETS choose_pick:
    #   0 edge matches (this knob)      -> hybrid P(top3) ~= 5.6%
    #   1 edge match  (choose_pick)     ->               ~= 6.7%   (its lone decision match)
    #   8 edge matches (default below)  ->               ~= 9.3%   (all upcoming MD3)
    # i.e. choose_pick sits between the two engine settings, monotone in the number
    # of edge-bearing odds matches (~0.5pp/match), all three at median rank 7. The
    # default (False) keeps the honest EV-max edge on the 8 upcoming MD3 matches,
    # which reflects the real decisional edge on matches still to be played.
    evmax_edge_free_odds: bool = False


@dataclass
class SimResult:
    config: SimConfig
    players: list[str]
    me_index: int
    me_cum: np.ndarray            # (n_sims, horizon) my cumulative points after each match
    finals: np.ndarray            # (n_players, n_sims) final totals
    me_rank: np.ndarray           # (n_sims,) my final rank (1 = top)
    realised_cum: np.ndarray      # (n_played,) my deterministic realised cumulative path
    n_played: int
    cum: np.ndarray | None = None  # (n_players, n_sims, horizon) all paths, only if track="all"

    @property
    def p_top(self) -> float:
        return float((self.me_rank <= self.config.target).mean())

    @property
    def p_win(self) -> float:
        return float((self.me_rank == 1).mean())

    def mean_final(self) -> float:
        return float(self.finals[self.me_index].mean())

    def median_final(self) -> float:
        return float(np.median(self.finals[self.me_index]))


# -- the engine -----------------------------------------------------------


def simulate(config: SimConfig | None = None, *, field_model: FieldModel | None = None,
             slots: list[MatchSlot] | None = None) -> SimResult:
    """Run the vectorised Monte-Carlo. Reproducible: a single seeded Generator,
    drawn in deterministic match order."""
    cfg = config or SimConfig()
    fm = field_model or FieldModel.from_disk()
    slots = slots or build_tournament(cfg.horizon)
    slots = slots[: cfg.horizon]
    rng = np.random.default_rng(cfg.seed)
    N = cfg.n_sims
    strategy = make_strategy(cfg.strategy, z_chase=cfg.z_chase)

    # player universe = everyone who appears in banked picks (12), incl. SELF
    picks = pd.read_csv(_PICKS)
    players = sorted(picks["player"].unique().tolist())
    if SELF not in players:
        players.append(SELF)
    me = cfg.me_player                      # the strategy-driven tracked player
    me_i = players.index(me)
    n_players = len(players)

    hybrid = cfg.regime == "hybrid"
    # banked totals for hybrid start (after the played matches inside the horizon)
    totals = np.zeros((n_players, N), dtype=np.float64)
    me_cum = np.zeros((N, cfg.horizon), dtype=np.float32)
    cum = (np.zeros((n_players, N, cfg.horizon), dtype=np.float32)
           if cfg.track == "all" else None)

    # field-model dist cache keyed by (player, tend, score) -> (scores array, probs)
    dist_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def field_points(player: str, tend: str, score: tuple[int, int], ah, aa) -> np.ndarray:
        key = (player, tend, tuple(score))
        if key not in dist_cache:
            d = fm.pick_distribution_for_consensus(player, tend, score)
            sc = np.array(list(d.keys()))
            pr = np.array([d[tuple(s)] for s in sc], dtype=float)
            pr = pr / pr.sum()
            dist_cache[key] = (sc, pr)
        sc, pr = dist_cache[key]
        idx = rng.choice(len(sc), size=N, p=pr)
        return _points_vec(sc[idx, 0], sc[idx, 1], ah, aa)

    realised = []  # my deterministic realised cumulative path over played matches

    for k, slot in enumerate(slots):
        # -- hybrid: played matches are the deterministic past --
        if hybrid and slot.played:
            for pi, p in enumerate(players):
                totals[pi] += slot.real_points.get(p, 0.0)
            realised.append(float(slot.real_points.get(me, 0.0)))
        else:
            # -- simulate this match --
            # decision_matrix = the matrix a strategy reasons over: the real odds
            # matrix for odds matches, or the plain-Poisson stand-in for generative.
            if slot.kind == "odds":
                flat = slot.matrix.flatten()
                flat = flat / flat.sum()
                g = slot.matrix.shape[0]
                d = rng.choice(flat.size, size=N, p=flat)
                ah, aa = d // g, d % g
                tend, score = slot.consensus
                decision_matrix = slot.matrix
            else:
                p_fav = float(_favourite_strengths(1, rng, cfg.fav_dist)[0])
                ah, aa, (tend, score), (lam_h, lam_a) = _sample_generative(
                    p_fav, slot.is_knockout, rng, N)
                decision_matrix = _poisson_matrix(lam_h, lam_a)

            # me's increment, delegated to the selected strategy (see strategies.py).
            # me_run / totals are the standings BEFORE this match (pre-increment): the
            # strategy reads them for its deficit/boundary; a copy avoids aliasing.
            ctx = StrategyContext(
                me_self=me, kind=slot.kind, matrix=decision_matrix,
                cons_tend=tend, cons_score=tuple(score), ah=ah, aa=aa,
                me_run=totals[me_i].copy(), totals=totals, me_index=me_i,
                n_remaining=cfg.horizon - k, target=cfg.target,
                field_points=field_points, edge_free_odds=cfg.evmax_edge_free_odds)
            totals[me_i] += strategy.me_increment(ctx)

            # opponents (field model)
            for pi, p in enumerate(players):
                if pi != me_i:
                    totals[pi] += field_points(p, tend, score, ah, aa)

        me_cum[:, k] = totals[me_i].astype(np.float32)
        if cum is not None:
            cum[:, :, k] = totals.astype(np.float32)

    me_rank = ((-totals).argsort(0).argsort(0) + 1)[me_i]
    return SimResult(
        config=cfg, players=players, me_index=me_i, me_cum=me_cum,
        finals=totals, me_rank=me_rank, cum=cum,
        realised_cum=np.cumsum(realised) if realised else np.array([]),
        n_played=len(realised),
    )


# -- smoke test -----------------------------------------------------------


def _smoke(n: int = 1000) -> None:
    fm = FieldModel.from_disk()
    print(f"Monte-Carlo engine smoke test - N={n:,}\n")
    for regime in ("counterfactual", "hybrid"):
        print(f"  regime = {regime}")
        for strat in ("evmax", "rank_relative", "contrarian"):
            cfg = SimConfig(n_sims=n, seed=0, regime=regime, strategy=strat,
                            horizon=TOURNAMENT_MATCHES)
            res = simulate(cfg, field_model=fm)
            print(f"    [{strat:>13}]  mean={res.mean_final():6.2f}  "
                  f"median={res.median_final():6.1f}  "
                  f"P(top3)={res.p_top:6.2%}  P(win)={res.p_win:5.2%}  "
                  f"med_rank={np.median(res.me_rank):.0f}")
    print(f"\n  players={len(res.players)}  me={SELF}@idx{res.me_index}  "
          f"n_played(hybrid)={res.n_played}  realised={res.realised_cum[-1] if res.n_played else 0:.0f} pts")
    print("\n  reference: src.rank_sim.choose_pick EV-max P(top3) today ~= 6.7-7.1% "
          "(EV-max hybrid brackets it; counterfactual & other strategies differ by design).")


if __name__ == "__main__":
    import sys
    _smoke(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
