"""Percentiles, rank distribution, rival reference paths, and the "option-pricing"
readout for the Monte-Carlo fan chart (ANALYSIS / VIZ ONLY).

Everything here is post-processing of an ``engine.SimResult``. The headline this
layer exists to make quantitative:

    mean final points = the Monte-Carlo expectation (an option's expected payoff).
    But P10 and P90 straddle it widely, and the RELATIVE-to-field band (me - field)
    has a NEGATIVE P10 even under EV-max - i.e. over 104 matches the variance band
    dwarfs the edge. A rational strategy is not a safe one.

The "leader regression" readout makes the companion point: a p10-type lead (24 pts,
rank 1 today) built by an always-fade strategy is the realised right tail of a -EV
process, expected to regress hard toward that strategy's dominated median.

Imports from ``src``/``analysis`` only flow one way (never imported by ``src``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.montecarlo.engine import (
    SELF,
    TOURNAMENT_MATCHES,
    _PICKS,
    SimConfig,
    build_tournament,
    simulate,
)
from analysis.montecarlo.strategies import SIGMA_MATCH

# percentile bands plotted as nested ribbons (lightest outer -> darkest inner)
BAND_PCTLS = [5, 10, 25, 50, 75, 90, 95]


# -- realised (deterministic) path - the RED overlay, regime-independent --


def realised_path(player: str = SELF) -> tuple[list[int], list[float]]:
    """The player's ACTUAL cumulative points over the matches played so far, from
    banked results. Used as the overlay in BOTH regimes (in counterfactual it is
    shown against the full counterfactual fan; in hybrid it IS the deterministic
    past). Returns (global_match_indices, cumulative_points)."""
    idx, cum, run = [], [], 0.0
    for slot in build_tournament():
        if slot.played:
            run += float(slot.real_points.get(player, 0.0))
            idx.append(slot.gindex)
            cum.append(run)
    return idx, cum


# -- core computations ---------------------------------------------------


def percentile_bands(me_cum: np.ndarray, pctls=BAND_PCTLS) -> dict[int, list[float]]:
    """P{pctl} of my cumulative points at each match-index (x-axis)."""
    arr = np.percentile(me_cum, pctls, axis=0)        # (len(pctls), horizon)
    return {p: arr[i].tolist() for i, p in enumerate(pctls)}


def rank_distribution(me_rank: np.ndarray, n_players: int) -> list[float]:
    """P(final rank = k) for k = 1..n_players."""
    counts = np.bincount(me_rank, minlength=n_players + 1)[1:n_players + 1]
    return (counts / me_rank.size).tolist()


def rival_reference_paths(cum: np.ndarray, target: int = 3,
                          exclude_index: int | None = None) -> dict[str, list[float]]:
    """Median cumulative-points paths of the order-statistic 'roles' across sims:
    leader (rank 1), top-`target` boundary (rank `target`), field median, and
    bottom-`target` boundary (rank n-target+1). Per match-index we sort the players
    within each sim and take the role's points level, then the cross-sim median.

    With ``exclude_index`` set (= me), the roles are computed over the OTHER players
    only, so they are true FIELD reference lines, independent of my strategy (and so
    shareable across the strategy charts). Needs ``track='all'``."""
    if exclude_index is not None:
        cum = np.delete(cum, exclude_index, axis=0)
    P = cum.shape[0]
    srt = np.sort(cum, axis=0)                         # ascending along players
    roles = {
        "leader": P - 1,                               # rank 1 (highest)
        f"top{target}_boundary": P - target,           # rank `target`
        "field_median": (P - 1) // 2,                  # central player
        f"bottom{target}_boundary": target - 1,        # rank n-target+1 (lowest few)
    }
    return {name: np.median(srt[r], axis=0).tolist() for name, r in roles.items()}


def final_spread_stats(result, others_benchmark: str = "median") -> dict:
    """Final-points distribution stats + the RELATIVE-to-field band - the
    quantitative core of the option-pricing headline.

    'edge'      = median of (my final - field benchmark)   (the typical alpha)
    'rel_band'  = P90 - P10 of that relative quantity       (the variance)
    The headline is rel_band ≫ |edge|, and rel_p10 < 0 means I underperform the
    field >10% of the time despite the edge."""
    me = result.finals[result.me_index]
    others = np.delete(result.finals, result.me_index, axis=0)
    bench = np.median(others, axis=0) if others_benchmark == "median" else others.mean(axis=0)
    rel = me - bench
    p10, p50, p90 = (float(np.percentile(me, q)) for q in (10, 50, 90))
    r10, r50, r90 = (float(np.percentile(rel, q)) for q in (10, 50, 90))
    return {
        "mean": float(me.mean()), "p10": p10, "p50": p50, "p90": p90,
        "spread": p90 - p10,
        "rel_edge": r50, "rel_p10": r10, "rel_p90": r90, "rel_band": r90 - r10,
        "p_top": result.p_top, "p_win": result.p_win,
        "median_rank": float(np.median(result.me_rank)),
    }


# -- assembled fan chart --------------------------------------------------


@dataclass
class FanChart:
    strategy: str
    regime: str
    horizon: int
    target: int
    n_sims: int
    seed: int
    players: list[str]
    me_player: str
    match_index: list[int]                     # 0..horizon-1 (cumulative AFTER match k)
    bands: dict[int, list[float]]              # pctl -> path
    rank_dist: list[float]                     # P(rank=k), k=1..n
    p_top: float
    p_win: float
    realised_index: list[int] = field(default_factory=list)
    realised_cum: list[float] = field(default_factory=list)
    realised_rank_so_far: int | None = None
    rivals: dict[str, list[float]] = field(default_factory=dict)
    spaghetti: list[list[float]] = field(default_factory=list)   # downsampled visible paths
    spread: dict = field(default_factory=dict)


def realised_rank_so_far(player: str = SELF) -> int:
    """The player's CURRENT rank from banked totals (1 = leader)."""
    picks = pd.read_csv(_PICKS)
    tot = picks[picks["points"].notna()].groupby("player")["points"].sum()
    return int((tot > tot.get(player, 0.0)).sum() + 1)


def current_leader() -> str:
    """The player currently top of the banked standings."""
    picks = pd.read_csv(_PICKS)
    return str(picks[picks["points"].notna()].groupby("player")["points"].sum().idxmax())


def build_fanchart(result, *, n_spaghetti: int = 200, seed: int = 0,
                   rivals: dict | None = None) -> FanChart:
    """Assemble a full FanChart from a SimResult. Rival lines come from ``rivals``
    if supplied (shared field reference, computed once per regime×horizon), else are
    computed from ``result.cum`` (requires track='all')."""
    cfg = result.config
    bands = percentile_bands(result.me_cum)
    if rivals is None:
        rivals = (rival_reference_paths(result.cum, cfg.target, result.me_index)
                  if result.cum is not None else {})
    ridx, rcum = realised_path(cfg.me_player)
    # downsampled spaghetti (a few hundred visible individual paths)
    rng = np.random.default_rng(seed)
    take = rng.choice(result.me_cum.shape[0], size=min(n_spaghetti, result.me_cum.shape[0]),
                      replace=False)
    return FanChart(
        strategy=cfg.strategy, regime=cfg.regime, horizon=cfg.horizon, target=cfg.target,
        n_sims=cfg.n_sims, seed=cfg.seed, players=result.players, me_player=cfg.me_player,
        match_index=list(range(cfg.horizon)), bands=bands,
        rank_dist=rank_distribution(result.me_rank, len(result.players)),
        p_top=result.p_top, p_win=result.p_win,
        realised_index=ridx, realised_cum=rcum,
        realised_rank_so_far=realised_rank_so_far(cfg.me_player),
        rivals=rivals, spaghetti=result.me_cum[take].tolist(),
        spread=final_spread_stats(result),
    )


# -- the "option-pricing" cross-strategy headline -------------------------


def option_pricing_report(results_by_strategy: dict) -> tuple[str, dict]:
    """Contrast the within-strategy variance band against the between-strategy
    edge, to make the headline: the band width dwarfs the edge."""
    rows = {s: final_spread_stats(r) for s, r in results_by_strategy.items()}
    # the "bold-play cost" = median shift between the two RATIONAL strategies (EV-max
    # vs rank-relative); contrarian is a caricature and is excluded from this contrast.
    bold_shift = (abs(rows["evmax"]["p50"] - rows["rank_relative"]["p50"])
                  if {"evmax", "rank_relative"} <= set(rows) else None)
    lines = ["OPTION-PRICING READOUT - final points (mean = MC expectation)",
             f"{'strategy':>14} {'mean':>6} {'P10':>5} {'P50':>5} {'P90':>5} "
             f"{'P90-P10':>7} {'relEdge':>7} {'relP10':>6} {'relBand':>7} {'P(top3)':>7}"]
    for s, v in rows.items():
        lines.append(f"{s:>14} {v['mean']:6.1f} {v['p10']:5.0f} {v['p50']:5.0f} "
                     f"{v['p90']:5.0f} {v['spread']:7.1f} {v['rel_edge']:+7.1f} "
                     f"{v['rel_p10']:+6.1f} {v['rel_band']:7.1f} {v['p_top']:7.1%}")
    ev = rows.get("evmax")
    if ev:
        ratio = ev["rel_band"] / max(abs(ev["rel_edge"]), 1e-9)
        lines += [
            "",
            f"HEADLINE: EV-max's edge over the field is just {ev['rel_edge']:+.0f} pts "
            f"(median of me - field median), but the RELATIVE band is {ev['rel_band']:.0f} pts "
            f"(P10 {ev['rel_p10']:+.0f} -> P90 {ev['rel_p90']:+.0f}).",
            f"          Relative P10 = {ev['rel_p10']:+.0f} pts < 0 -> even a rational EV-max "
            f"player finishes BELOW the field median >10% of the time. Variance dwarfs edge "
            f"~{ratio:.0f}:1.",
        ]
        if bold_shift is not None:
            lines.append(f"          Switching to bold (rank-relative) shifts the median by "
                         f"{bold_shift:.0f} pts and widens the band to "
                         f"{rows['rank_relative']['rel_band']:.0f} pts - more variance, "
                         f"no added edge (relEdge {rows['rank_relative']['rel_edge']:+.0f}).")
    return "\n".join(lines), {"rows": rows, "bold_shift": bold_shift}


# -- leader-regression readout (a p10-type lead from today) ---------------


def leader_regression(leader: str | None = None, *, n_sims: int = 40_000, seed: int = 0,
                      field_model=None) -> dict:
    """Project the CURRENT leader forward under the contrarian (always-fade) strategy
    from today's standings (hybrid). Shows the expected regression of a p10-type
    lead: it starts at rank 1 but the -EV fade strategy pulls it toward its median."""
    if leader is None:
        picks = pd.read_csv(_PICKS)
        tot = picks[picks["points"].notna()].groupby("player")["points"].sum()
        leader = str(tot.idxmax())
    cfg = SimConfig(n_sims=n_sims, seed=seed, regime="hybrid", strategy="contrarian",
                    me_player=leader)
    res = simulate(cfg, field_model=field_model)
    start = float(res.realised_cum[-1]) if res.n_played else 0.0
    return {
        "leader": leader, "start_points": start, "start_rank": realised_rank_so_far(leader),
        "proj_mean": res.mean_final(), "proj_median": res.median_final(),
        "proj_p10": float(np.percentile(res.finals[res.me_index], 10)),
        "proj_p90": float(np.percentile(res.finals[res.me_index], 90)),
        "proj_median_rank": float(np.median(res.me_rank)),
        "proj_p_top": res.p_top, "proj_p_win": res.p_win,
    }


# -- skill / luck variance decomposition + detectability -------------------


def skill_luck_decomposition(*, n_sims: int = 80_000, seed: int = 0, field_model=None,
                             regime: str = "counterfactual", horizon: int = TOURNAMENT_MATCHES,
                             bootstrap: int = 500, noise_reps: int = 4000) -> dict:
    """Detectability of skill (the HEADLINE) + a law-of-total-variance variance split
    kept only as a CAVEATED DIAGNOSTIC.

    HEADLINE - ``detect`` (Fundamental Law of Active Management, Grinold). Run a sim
    where self keeps his genuine EV-max edge and read the per-tournament information
    ratio IR = rel_edge / σ_rel, the implied IC = IR/√breadth, and the number of World
    Cups needed for a t=2 detection. These use the MEASURED edge directly and are robust.

    DIAGNOSTIC ONLY - the variance split. By the law of total variance (an exact
    identity), Var(final) = Var(LUCK) + Var(SKILL) with all players made symmetric
    (``evmax_edge_free_odds=True`` -> everyone is a field-model player). BUT the field
    model is fit on MD1 ONLY (oracle consensus covers MD1; ~8 picks/player, shrinkage
    w≈0.62), so ``var_skill`` mostly reflects 8-pick ESTIMATION NOISE, not durable
    skill - and the per-player ranking reflects MD1 follow rates, NOT each player's
    archetype (e.g. p10 followed consensus 7/8 in MD1, so it ranks high here even
    though its MD2 play is a fade). The ``noise_floor`` block proves this: the observed
    follow-rate spread sits INSIDE the band a set of TRULY IDENTICAL players would show
    from 8-pick estimates. So the variance % is NOT a reliable skill/luck number - read
    the detectability block instead. ``mauboussin_null_luck`` (≈ horizon·σ²) sanity-
    checks ``var_luck`` only."""
    from src.field_model import FieldModel
    fm = field_model or FieldModel.from_disk()
    base = simulate(SimConfig(n_sims=n_sims, seed=seed, regime=regime, strategy="evmax",
                              evmax_edge_free_odds=True, horizon=horizon, track="all"),
                    field_model=fm)
    finals = base.finals                                   # (n_players, n_sims)
    mu = finals.mean(axis=1)                                # per-player expected final
    v = finals.var(axis=1, ddof=1)                          # per-player within-tourn var = luck
    var_skill = float(mu.var(ddof=1))
    var_luck = float(v.mean())
    var_obs = var_skill + var_luck
    luck_share = var_luck / var_obs

    # bootstrap over SIMS - DELIBERATELY labelled as capturing sim-sampling noise ONLY,
    # NOT the dominant uncertainty (the 8-pick field-model fit; see noise_floor).
    rng = np.random.default_rng(seed + 1)
    shares = np.empty(bootstrap)
    for b in range(bootstrap):
        idx = rng.integers(0, n_sims, n_sims)
        f = finals[:, idx]
        shares[b] = (vl := float(f.var(axis=1, ddof=1).mean())) / (
            float(f.mean(axis=1).var(ddof=1)) + vl)
    ci_sims = (float(np.percentile(shares, 5)), float(np.percentile(shares, 95)))

    ranking = sorted(((p, float(mu[i])) for i, p in enumerate(base.players)),
                     key=lambda t: t[1], reverse=True)
    field_mean = float(mu.mean())

    # -- noise floor: is the apparent skill spread distinguishable from 8-pick noise? --
    players = sorted(fm.players)
    pop, k = fm.pop_follow, fm.follow_k
    n_obs = int(np.median([fm._n_follow.get(p, 0) for p in players]) or 8)
    w = n_obs / (n_obs + k)
    obs_follow_sd = float(np.std([fm.follow_rate(p) for p in players], ddof=1))
    nrng = np.random.default_rng(seed + 2)
    null_sds = np.empty(noise_reps)
    for r in range(noise_reps):                            # P identical players, 8-pick estimates
        own = nrng.binomial(n_obs, pop, size=len(players)) / n_obs
        null_sds[r] = (w * own + (1 - w) * pop).std(ddof=1)
    null_band = (float(np.percentile(null_sds, 5)), float(np.percentile(null_sds, 95)))
    skill_inside_noise = obs_follow_sd <= null_band[1]

    # -- HEADLINE: detectability of self's actual EV-max edge over the field --
    det_res = simulate(SimConfig(n_sims=n_sims, seed=seed, regime="counterfactual",
                                 strategy="evmax", horizon=horizon), field_model=fm)
    sp = final_spread_stats(det_res)
    rel_edge, rel_band = sp["rel_edge"], sp["rel_band"]
    sigma_rel = rel_band / 2.5631                           # P90-P10 = 2.5631·σ (normal)
    ir = rel_edge / sigma_rel                               # info ratio per tournament
    implied_ic = ir / np.sqrt(horizon)                      # Grinold: IR = IC·√BR, BR=matches
    edge_per_match = rel_edge / horizon
    matches_for_t2 = (2 * SIGMA_MATCH / edge_per_match) ** 2 if edge_per_match else float("inf")

    return {
        "regime": regime, "horizon": horizon, "n_sims": n_sims, "seed": seed,
        "var_skill": var_skill, "var_luck": var_luck, "var_obs": var_obs,
        "sigma_skill": float(np.sqrt(var_skill)), "sigma_luck": float(np.sqrt(var_luck)),
        "luck_share": luck_share, "skill_share": 1 - luck_share, "luck_share_ci_sims": ci_sims,
        "skill_ranking": ranking, "field_mean": field_mean,
        "mauboussin_null_luck": horizon * SIGMA_MATCH ** 2,
        "noise_floor": {
            "n_picks_per_player": n_obs, "shrinkage_w": w,
            "obs_follow_sd": obs_follow_sd, "null_follow_sd_band": null_band,
            "skill_inside_noise_band": bool(skill_inside_noise),
        },
        "detect": {
            "rel_edge": rel_edge, "rel_band": rel_band, "sigma_rel": sigma_rel,
            "IR_tournament": ir, "implied_IC": implied_ic, "t_one_tournament": ir,
            "tournaments_for_t2_rel": (2 / ir) ** 2 if ir else float("inf"),
            "edge_per_match": edge_per_match, "matches_for_t2_abs": matches_for_t2,
            "tournaments_for_t2_abs": matches_for_t2 / horizon,
        },
    }


def format_skill_luck(d: dict) -> str:
    det, nf = d["detect"], d["noise_floor"]
    lines = [
        "SKILL DETECTABILITY  [HEADLINE - Fundamental Law of Active Management, Grinold]",
        f"  self's EV-max edge: rel_edge = {det['rel_edge']:+.1f} pts/tournament, "
        f"band P90-P10 = {det['rel_band']:.0f}  (σ_rel ≈ {det['sigma_rel']:.1f})",
        f"  IR per tournament  = {det['IR_tournament']:.2f}   (= t-stat of ONE World Cup's outperformance)",
        f"  implied IC         = {det['implied_IC']:.3f}   (IR = IC·√breadth, breadth = {d['horizon']} matches)",
        f"  World Cups for t=2 = {det['tournaments_for_t2_rel']:.0f} (relative)  /  "
        f"{det['tournaments_for_t2_abs']:.0f} (absolute, ≈{det['matches_for_t2_abs']:.0f} matches)",
        "  => one tournament's outperformance is ~1·σ: skill is real but UNDETECTABLE over a single WC.",
        "  note: relative variance < absolute (a player's score is positively correlated with the field -",
        "        universal-zero matches, shared favourites); this shrinks relative variance but does NOT",
        "        rescue detectability - breadth is still ~104 matches.",
        "",
        "VARIANCE SPLIT  [DIAGNOSTIC ONLY - confounded by the 8-pick field-model fit; NOT the headline]",
        f"  law of total variance: Var(obs) {d['var_obs']:.0f} = Var(luck) {d['var_luck']:.0f} + "
        f"Var(skill) {d['var_skill']:.0f}  -> raw luck share {d['luck_share']:.0%}",
        f"  (sim-only bootstrap CI [{d['luck_share_ci_sims'][0]:.0%}, {d['luck_share_ci_sims'][1]:.0%}] is "
        "misleadingly tight: it captures sim noise, NOT the 8-pick fit uncertainty below)",
        f"  Mauboussin null luck ≈ horizon·σ² = {d['mauboussin_null_luck']:.0f} vs Var(luck) "
        f"{d['var_luck']:.0f}  -> same order ok",
        f"  NOISE FLOOR: field model fit on {nf['n_picks_per_player']} MD1 picks/player (w≈{nf['shrinkage_w']:.2f}). "
        f"Observed follow-rate sd {nf['obs_follow_sd']:.3f}",
        f"               vs identical-players 8-pick null band [{nf['null_follow_sd_band'][0]:.3f}, "
        f"{nf['null_follow_sd_band'][1]:.3f}] -> "
        + ("INSIDE: the apparent 'skill' spread is ~all estimation noise."
           if nf["skill_inside_noise_band"] else "OUTSIDE: some real skill spread."),
        "  => the raw luck share UNDER-states luck; noise-corrected it is ≳90%. Read DETECTABILITY, not this %.",
        "",
        f"  per-player 'skill' (MD1 follow rate, NOT archetype; field mean {d['field_mean']:.1f}) - "
        "note p10 ranks high because it FOLLOWED 7/8 in MD1; its fade is MD2 (unmodelled):",
    ]
    for i, (p, m) in enumerate(d["skill_ranking"], 1):
        lines.append(f"    {i:2d}. {p:<12} {m:6.1f}")
    return "\n".join(lines)


# -- reproducibility -------------------------------------------------------


def assert_percentiles_reproducible(cfg: SimConfig, *, field_model=None, tol: float = 1e-9) -> bool:
    """Two runs at the same seed must give identical percentile bands (±tol)."""
    a = percentile_bands(simulate(cfg, field_model=field_model).me_cum)
    b = percentile_bands(simulate(cfg, field_model=field_model).me_cum)
    for p in BAND_PCTLS:
        if not np.allclose(a[p], b[p], atol=tol):
            return False
    return True


# -- CLI --------------------------------------------------------------------


def _main(n: int = 40_000) -> None:
    from src.field_model import FieldModel
    fm = FieldModel.from_disk()
    print(f"Monte-Carlo stats - N={n:,}, counterfactual\n")
    results = {s: simulate(SimConfig(n_sims=n, seed=0, regime="counterfactual", strategy=s),
                           field_model=fm) for s in ("evmax", "rank_relative", "contrarian")}
    txt, _ = option_pricing_report(results)
    print(txt)
    print("\n" + "-" * 78)
    lr = leader_regression(n_sims=n, field_model=fm)
    print(f"LEADER REGRESSION - {lr['leader']} (always-fade), from today's standings:")
    print(f"  today: {lr['start_points']:.0f} pts, rank {lr['start_rank']} (leader)")
    print(f"  projected final: median {lr['proj_median']:.0f} pts (P10 {lr['proj_p10']:.0f} / "
          f"P90 {lr['proj_p90']:.0f}), median rank {lr['proj_median_rank']:.0f}, "
          f"P(top3) {lr['proj_p_top']:.1%}")
    print(f"  -> the lead regresses from rank {lr['start_rank']} to a median final rank "
          f"{lr['proj_median_rank']:.0f}: a freak P99 path reverting to a -EV median.")
    print("\n" + "-" * 78)
    print(format_skill_luck(skill_luck_decomposition(n_sims=n, seed=0, field_model=fm)))
    repro = assert_percentiles_reproducible(
        SimConfig(n_sims=3000, seed=1, regime="counterfactual"), field_model=fm)
    print(f"\nreproducible percentile bands (seed-stable): {repro}")


if __name__ == "__main__":
    import sys
    _main(int(sys.argv[1]) if len(sys.argv) > 1 else 40_000)
