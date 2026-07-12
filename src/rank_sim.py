"""Monte-Carlo final-standings (rank) simulator.

The decision objective in a pool is P(finish at the target rank), not
E[points]. The two coincide early / at parity (variance is washed out by the
many matches still to come) and diverge as you move clearly ahead or behind -
the tournament / utility-curvature argument (Browne 1999/2000; Dubins-Savage
bold vs timid play; Bell-Cover; Lazear-Rosen).

This module simulates the rest of the tournament from the current standings and
estimates the distribution of *our* final rank for a candidate next-match pick,
so the pick can be chosen by ``argmax P(rank <= target)``. Comparing that pick
to the pure EV-max pick is the automatic "grey -> red" regime detector: when they
agree we're in EV-max territory; when the rank-optimal pick deviates (takes
variance), we've entered the chase/protect regime.

Opponent picks
--------------
For the match being decided, opponent picks are passed in: *actual* once they're
visible (post-kickoff), or the oracle's predicted consensus beforehand. Future
matches are modelled as per-match points draws (the standings-projection model);
wiring the oracle's per-opponent pick model into the future legs is the next
refinement.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from src.scoring.kicktipp import points

# Per-match pool points over {0,2,3,4}. Baseline mean = 1.50.
_VALS = np.array([0, 2, 3, 4])
_BASE_DIST = np.array([0.46, 0.28, 0.10, 0.16])  # mean = 1.50


def _points_dist(edge: float, base: np.ndarray | None = None) -> np.ndarray:
    """Points distribution shifted so ``edge`` is a TRUE per-match points edge.

    The raw shift [-e/2, 0, e/6, e/3] has points-mean 11/6, so rescale by 6/11 ->
    mean(dist) - mean(base) == edge exactly (before clip/renormalise).
    TODO: refit ``base`` from accumulated picks history (snapshot/picks) once available.
    """
    if base is None:
        base = _BASE_DIST
    shift = np.array([-edge / 2, 0.0, edge / 6, edge / 3]) * (6.0 / 11.0)
    p = np.clip(base + shift, 1e-6, None)
    return p / p.sum()


def _points_grid(pick: tuple[int, int], g: int) -> np.ndarray:
    """(g x g) array of pool points for `pick` vs every actual (i, j)."""
    return np.array([[points(pick, (i, j)) for j in range(g)] for i in range(g)])


@dataclass
class RankResult:
    pick: tuple[int, int]
    match_ev: float
    p_rank1: float
    p_top: float
    median_rank: float
    se_top: float = 0.0                  # Monte-Carlo standard error on p_top
    diff_vs_evmax: float | None = None   # paired Δ p_top vs the EV-max pick (CRN)
    diff_se: float | None = None         # paired-difference standard error
    top_sample: object = None            # per-sim (rank<=target) bool array (internal)


def simulate_rank(
    current_totals: dict[str, float],
    me: str,
    my_pick: tuple[int, int],
    score_matrix: np.ndarray,
    opponent_picks: dict[str, tuple[int, int]],
    *,
    n_future: int,
    edge: float = 0.0,
    target: int = 3,
    n_sims: int = 50_000,
    seed: int = 0,
) -> RankResult:
    """Estimate our final-rank distribution for one candidate next-match pick.

    Parameters
    ----------
    current_totals : player -> points so far
    me : our player name (key in current_totals)
    my_pick : the (home, away) score we're evaluating for the match being decided
    score_matrix : our model's P(i, j) for that match
    opponent_picks : player -> (home, away) for that match (visible or predicted)
    n_future : number of remaining matches *after* this one
    edge : our per-match points edge over the field on future matches
    target : rank threshold for P(rank <= target)
    """
    rng = np.random.default_rng(seed)
    players = list(current_totals)
    g = score_matrix.shape[0]

    # Sample the match result from our score matrix.
    flat = score_matrix.flatten()
    flat = flat / flat.sum()
    draw = rng.choice(flat.size, size=n_sims, p=flat)
    ri, rj = draw // g, draw % g

    totals = {p: np.full(n_sims, float(current_totals[p])) for p in players}

    # Points on the match being decided (vectorised via a points grid per pick).
    for p in players:
        pick = my_pick if p == me else opponent_picks.get(p)
        if pick is None:
            continue
        grid = _points_grid(pick, g)
        totals[p] = totals[p] + grid[ri, rj]

    # Future matches: per-match points draws (us with `edge`, field at baseline).
    for p in players:
        e = edge if p == me else 0.0
        if n_future > 0:
            totals[p] = totals[p] + rng.choice(
                _VALS, size=(n_sims, n_future), p=_points_dist(e)).sum(axis=1)

    mat = np.vstack([totals[p] for p in players])      # (n_players, n_sims)
    rank = (-mat).argsort(0).argsort(0) + 1            # 1 = top
    mi = players.index(me)
    me_rank = rank[mi]

    top = me_rank <= target
    p_top = float(top.mean())
    match_ev = float((score_matrix * _points_grid(my_pick, g)).sum())
    return RankResult(
        pick=my_pick,
        match_ev=round(match_ev, 3),
        p_rank1=float((me_rank == 1).mean()),
        p_top=p_top,
        median_rank=float(np.median(me_rank)),
        se_top=float(np.sqrt(p_top * (1.0 - p_top) / top.size)),
        top_sample=top,
    )


def simulate_rank_multi(
    current_totals: dict[str, float],
    me: str,
    modeled_matches: list[dict],
    *,
    n_generic_future: int,
    edge: float = 0.0,
    target: int = 3,
    n_sims: int = 50_000,
    seed: int = 0,
    field_model=None,
    fav_dist=None,
) -> RankResult:
    """Rank sim with several *explicitly modelled* matches + generic future legs.

    With ``field_model`` set, the future legs use the correlated field-model sim
    (shared outcome per match; we play EV-max, opponents are sampled). With
    ``field_model=None`` (default) the legacy i.i.d. path runs unchanged - and the
    ``edge`` knob applies only there (the field-model path's edge is endogenous).

    Each entry of ``modeled_matches`` is a dict with:
        score_matrix    : our P(i, j) for that match
        my_pick         : our (home, away) pick
        opponent_picks  : {player -> (home, away)} - visible picks (post-kickoff)
                          or oracle-predicted picks (``field_picks_consensus``)

    The first modelled match is treated as the *decision* match for the returned
    ``pick``/``match_ev``. Remaining matches model the near-term field; matches
    with no oracle data fall back to ``n_generic_future`` generic points draws.
    """
    rng = np.random.default_rng(seed)
    players = list(current_totals)
    totals = {p: np.full(n_sims, float(current_totals[p])) for p in players}

    for mm in modeled_matches:
        sm = mm["score_matrix"]
        g = sm.shape[0]
        flat = sm.flatten()
        flat = flat / flat.sum()
        d = rng.choice(flat.size, size=n_sims, p=flat)
        ri, rj = d // g, d % g
        grids: dict[tuple[int, int], np.ndarray] = {}
        for p in players:
            pick = mm["my_pick"] if p == me else mm["opponent_picks"].get(p)
            if pick is None:
                continue
            pick = tuple(pick)
            if pick not in grids:
                grids[pick] = _points_grid(pick, g)
            totals[p] = totals[p] + grids[pick][ri, rj]

    if field_model is not None and n_generic_future > 0:
        fut = _sim_future_correlated(players, me, n_generic_future, field_model, rng,
                                     n_sims, fav_dist)
        for p in players:
            totals[p] = totals[p] + fut[p]
    else:                                    # i.i.d. legacy path; the edge knob lives here
        for p in players:
            e = edge if p == me else 0.0
            if n_generic_future > 0:
                totals[p] = totals[p] + rng.choice(
                    _VALS, size=(n_sims, n_generic_future), p=_points_dist(e)).sum(axis=1)

    mat = np.vstack([totals[p] for p in players])
    rank = (-mat).argsort(0).argsort(0) + 1
    me_rank = rank[players.index(me)]

    d0 = modeled_matches[0]
    g0 = d0["score_matrix"].shape[0]
    top = me_rank <= target
    p_top = float(top.mean())
    match_ev = float((d0["score_matrix"] * _points_grid(tuple(d0["my_pick"]), g0)).sum())
    return RankResult(
        pick=tuple(d0["my_pick"]),
        match_ev=round(match_ev, 3),
        p_rank1=float((me_rank == 1).mean()),
        p_top=p_top,
        median_rank=float(np.median(me_rank)),
        se_top=float(np.sqrt(p_top * (1.0 - p_top) / top.size)),
        top_sample=top,
    )


def compare_picks(
    candidate_picks: list[tuple[int, int]],
    *,
    target: int = 3,
    **kwargs,
) -> list[RankResult]:
    """Run simulate_rank for several candidate own-picks; sorted by P(rank<=target).

    Each result is annotated with the PAIRED difference in p_top vs the EV-max
    pick (highest match_ev). All candidates share the same seed (common random
    numbers), so the paired SE is far tighter than se_top and tells us whether a
    pick gap is real or Monte-Carlo noise.
    """
    out = [simulate_rank(my_pick=p, target=target, **kwargs) for p in candidate_picks]
    ref = max(out, key=lambda r: r.match_ev)               # EV-max reference pick
    for r in out:
        if r is ref or r.top_sample is None or ref.top_sample is None:
            continue
        diff = r.top_sample.astype(float) - ref.top_sample.astype(float)
        r.diff_vs_evmax = float(diff.mean())
        r.diff_se = float(diff.std(ddof=1) / np.sqrt(diff.size))
    return sorted(out, key=lambda r: r.p_top, reverse=True)


# WC2026 (48 teams) - fixed tournament format. The pool scores the knockouts too
# (2-4 point tiers), so the rank-sim horizon must span the WHOLE tournament, not just
# the group stage.
GROUP_STAGE_MATCHES = 72      # 48 teams, 12 groups, 3 each -> 72
KNOCKOUT_MATCHES = 32         # R32(16)+R16(8)+QF(4)+SF(2)+3rd(1)+final(1)
TOURNAMENT_MATCHES = GROUP_STAGE_MATCHES + KNOCKOUT_MATCHES   # 104


def remaining_match_count(max_matchday: int = 15) -> dict:
    """Tournament matches remaining, for the rank-sim horizon.

    Counts scraped leaderboard fixtures dynamically, but FLOORS the total at
    TOURNAMENT_MATCHES (104): the knockout fixtures are not yet exposed by the
    scraper (spieltagIndex 11-15 return nothing; 16+ wrap to stale group data),
    so the group-only scrape would otherwise undercount the horizon at 72.
    max(scraped, 104) self-corrects with NO double-count once the bracket draw
    populates the knockout rounds (scraped rises 72 -> 104). 'played' counts only
    resolved fixtures (knockouts can't be played until the draw).

    TODO(knockout): when knockout rounds become scrapeable, (i) handle the
    spieltagIndex wrap past the last real matchday (md16+ returns stale group
    fixtures -> would double-count if the loop reached it; the early break
    currently protects us), and (ii) the FIELD MODEL needs review for knockout
    legs - a 90' draw still scores 2-4 but ET/penalties and the absence of
    group-stage draw incentives change the dynamics, and _favourite_strengths
    may need recentering. Not changing the field model now.
    (Verified 2026-06-15: scraped 72 group / 14 played / floor 104.)
    """
    from src.data.leaderboard import scrape_leaderboard
    scraped = played = mds = 0
    for md in range(1, max_matchday + 1):
        lb = scrape_leaderboard(spieltag=md)
        if not lb.fixtures:
            break
        mds += 1
        scraped += len(lb.fixtures)
        played += sum(1 for f in lb.fixtures if f.result is not None)
    total = max(scraped, TOURNAMENT_MATCHES)   # floor at 104 until knockouts are scrapeable
    return {"total": total, "played": played, "remaining": total - played,
            "matchdays": mds, "scraped": scraped}


# -- correlated field-model future legs + relative-EV optimiser -------


def _points_vec(pa, pb, ah, aa):
    """Vectorised pool scoring (numpy arrays): 4 exact / 3 GD / 2 tendency / 0.
    The ``ps != 0`` gate enforces 'no goal-difference tier for draws' (draws score
    4 or 2 only) - the same rule as scoring.kicktipp.points()."""
    exact = (pa == ah) & (pb == aa)
    ps, asg = np.sign(pa - pb), np.sign(ah - aa)
    tend = ps == asg
    gd = (pa - pb) == (ah - aa)
    return np.where(exact, 4, np.where(~tend, 0, np.where((ps != 0) & gd, 3, 2)))


@functools.lru_cache(maxsize=1)
def _empirical_match_structures():
    """Odds-derived ``(λ_home, λ_away)`` per snapshotted match - the empirical
    distribution of real match structures the future legs bootstrap from.

    Reconstructs λ from the blended (sharp ⊕ kicktipp) devigged 1X2 + O/U banked
    in snapshots.csv, exactly as the live score engine does. SIGNED and in fixture
    order, so resampling WHOLE ROWS jointly preserves the s-t dependence, the
    natural favourite-side split (~72% home / 28% away in WC group play) and the
    full supremacy spread - including the toss-ups the synthetic generator lacked.
    Returns an ``(N, 2)`` float array, or ``None`` when unavailable / too few atoms
    (-> graceful fallback to the synthetic ``_favourite_strengths`` generator).

    Cached: the snapshot set is stable within a run; ``cache_clear()`` after a
    new bank refreshes it.
    """
    try:
        from pathlib import Path

        import pandas as pd
        import yaml

        from src import snapshot
        from src.odds.reconstruct import reconstruct_lambdas
    except Exception:  # pragma: no cover - import guard
        return None
    try:
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "config.yaml").read_text())
        w = float(cfg.get("model", {}).get("blend_weight", 0.0))
    except Exception:
        w = 0.0
    out = []
    for r in snapshot.load_history().itertuples(index=False):
        try:
            kt = (float(r.kt_home), float(r.kt_draw), float(r.kt_away))
            if any(v != v for v in kt):                       # NaN kicktipp line
                continue
            sharp = (r.sharp_home, r.sharp_draw, r.sharp_away)
            if w > 0 and all(pd.notna(s) for s in sharp):
                bl = tuple(w * float(s) + (1 - w) * k for s, k in zip(sharp, kt))
                z = sum(bl)
                bl = tuple(b / z for b in bl)
            else:
                bl = kt
            ou = float(r.ou_over_2_5) if pd.notna(r.ou_over_2_5) else None
            lam, mu, _ = reconstruct_lambdas(*bl, ou)
            out.append((lam, mu))
        except Exception:
            continue
    return np.array(out, dtype=float) if len(out) >= 4 else None


def _future_leg_lambdas(n, rng, fav_dist=None):
    """Per-leg ``(λ_home, λ_away)`` for ``n`` synthetic future matches.

    PRIMARY: raw joint bootstrap of the empirical odds-derived structures
    (``_empirical_match_structures``) - resample whole observed match-rows with
    replacement, parameter-free (no smoothing/jitter; 32 atoms over ~72 legs is
    immaterial to the rank sim). This reflects the real spread of supremacy and
    the toss-up fraction, so the favourite-follow vs draw-fade style gap stops
    compounding undamped (the old always-home-favourite artefact).

    FALLBACK (empirical unavailable, or an explicit ``fav_dist`` override is
    given): the synthetic ``_favourite_strengths`` generator, mapped to a
    home-favoured ``(λ_h, λ_a)`` as before.
    """
    structures = None if fav_dist is not None else _empirical_match_structures()
    if structures is not None:
        return structures[rng.integers(0, len(structures), size=n)]   # joint row resample
    pf = _favourite_strengths(n, rng, fav_dist)
    return np.column_stack([0.8 + 2.2 * pf, 1.2 - 0.8 * pf])


def _favourite_strengths(n, rng, dist=None):
    """Draw n favourite-strength proxies in ~[0.5, 0.95], mean ≈ 0.62.

    GENERATIVE FALLBACK: used only when no empirical structures are available (or
    an explicit ``dist`` override is passed). The PRIMARY future-leg generator is
    now ``_future_leg_lambdas`` (empirical bootstrap). Kept for graceful
    degradation and as the ``dist=(lo, hi)`` override hook. NOTE: this stand-in
    is always a HOME favourite with little spread - the artefact that motivated
    the empirical bootstrap; do not route the main path through it.
    """
    if dist is None:
        return np.clip(rng.beta(1.8, 5.0, size=n) * 0.45 + 0.5, 0.5, 0.95)
    return rng.uniform(dist[0], dist[1], size=n)


def _decisive_dist(dist: dict) -> dict:
    """Project a scoreline pick-distribution onto DECISIVE results (drop draws,
    renormalise). a.PSO knockouts can never end level, so a drawn pick is
    impossible - this reuses each player's group-stage tendencies (follow rate,
    boldness, exact scatter) minus the now-impossible draw mass. Degenerate
    all-draw input is returned unchanged."""
    dec = {s: p for s, p in dist.items() if s[0] != s[1]}
    z = sum(dec.values())
    return {s: p / z for s, p in dec.items()} if z > 0 else dist


def _sim_future_correlated(players, me, n_future, field_model, rng, n_sims, fav_dist=None,
                           knockout=False):
    """Future legs with a SHARED outcome per match (true cross-player correlation).

    ``knockout``: a.PSO future legs - a level shared outcome is resolved to the
    favourite (+1 goal) so no future leg is a draw, and every player's pick
    distribution is projected onto decisive scorelines (:func:`_decisive_dist`).

    For each synthetic match a single outcome is sampled and *all* players -
    INCLUDING ``me`` - are scored against it from ``field_model`` (chalk => everyone
    moves together; upset => shared bust). The future is UNKNOWN and UNDECIDED, so
    NO player carries a systematic forward edge: "future me" behaves like the field
    model's self, not a perfect EV-max oracle. Players separate over the future
    ONLY through symmetric exact-score scatter, so P(rank) reflects current
    standings + honest variance - not a compounding fantasy edge. (The one
    legitimate asymmetry - me playing my candidate pick - lives in choose_pick's
    DECISION match, not here.) The matches are a generative stand-in for the
    unknown fixtures: their structures are a RAW JOINT BOOTSTRAP of the empirical
    odds-derived ``(λ_home, λ_away)`` (see ``_future_leg_lambdas`` /
    ``_empirical_match_structures``), so the favourite SIDE varies (home or away
    per the sign of supremacy) and the toss-up fraction is real - not the old
    always-strong-home-favourite stand-in that compounded a fake follower edge.
    ``me`` is accepted for call compatibility but treated like any other player.
    """
    totals = {p: np.zeros(n_sims) for p in players}
    for lam_h, lam_a in _future_leg_lambdas(n_future, rng, fav_dist):
        ah = rng.poisson(lam_h, n_sims)
        aa = rng.poisson(lam_a, n_sims)                       # SHARED outcome (favourite = sign(s))
        cons_score = (int(round(lam_h)), int(round(lam_a)))   # field-model anchor
        cons_tend = "home" if lam_h >= lam_a else "away"      # favourite side from sign(supremacy)
        if knockout:                                          # a.PSO: no level result survives
            tie = ah == aa
            if lam_h >= lam_a:
                ah = np.where(tie, ah + 1, ah)
            else:
                aa = np.where(tie, aa + 1, aa)
        for p in players:                                     # ALL players incl. me - edge-free
            dist = field_model.pick_distribution_for_consensus(p, cons_tend, cons_score)
            if knockout:
                dist = _decisive_dist(dist)
            scores = list(dist)
            idx = rng.choice(len(scores), size=n_sims, p=[dist[s] for s in scores])
            arr = np.array(scores)
            totals[p] = totals[p] + _points_vec(arr[idx, 0], arr[idx, 1], ah, aa)
    return totals


def choose_pick(decision_score_matrix, decision_consensus, candidates, field_model,
                current_totals, me, *, horizon, target=3, n_sims=40_000,
                fav_dist=None, seed=0, knockout=False):
    """Relative-EV optimiser: rank the candidate decision picks by P(rank<=target).

    Opponents on the decision match are SAMPLED from ``field_model``; future legs
    use the correlated field-model sim. Shared randomness (decision outcome,
    opponents' picks, future legs) is drawn ONCE - independent of our pick - so
    candidates are compared under common random numbers (tight paired SE). Each
    RankResult carries ``match_ev`` (so the EV cost of a rank-optimal deviation is
    visible) plus ``diff_vs_evmax`` / ``diff_se``. Sorted by p_top, descending.

    ``knockout``: a.PSO mode - the ``decision_score_matrix`` is already draw-free,
    and opponents' decision picks + future legs are projected onto decisive
    scorelines (no opponent knockout picks exist yet, so this reuses their
    group-stage tendencies minus the impossible draws - recalibrate once real
    knockout picks land).
    """
    rng = np.random.default_rng(seed)
    players = list(current_totals)
    g = decision_score_matrix.shape[0]
    flat = decision_score_matrix.flatten()
    flat = flat / flat.sum()
    d = rng.choice(flat.size, size=n_sims, p=flat)
    ah, aa = d // g, d % g
    cons_tend, cons_score = decision_consensus

    base = {p: np.full(n_sims, float(current_totals[p])) for p in players}
    for p in players:                        # opponents' decision picks (independent of OUR pick)
        if p == me:
            continue
        dist = field_model.pick_distribution_for_consensus(p, cons_tend, cons_score)
        if knockout:
            dist = _decisive_dist(dist)
        scores = list(dist)
        idx = rng.choice(len(scores), size=n_sims, p=[dist[s] for s in scores])
        arr = np.array(scores)
        base[p] = base[p] + _points_vec(arr[idx, 0], arr[idx, 1], ah, aa)
    fut = _sim_future_correlated(players, me, horizon, field_model, rng, n_sims, fav_dist,
                                 knockout=knockout)
    for p in players:
        base[p] = base[p] + fut[p]

    results = []
    for cand in candidates:
        me_total = base[me] + _points_vec(np.full(n_sims, cand[0]),
                                          np.full(n_sims, cand[1]), ah, aa)
        M = np.vstack([me_total if p == me else base[p] for p in players])
        me_rank = ((-M).argsort(0).argsort(0) + 1)[players.index(me)]
        top = me_rank <= target
        p_top = float(top.mean())
        results.append(RankResult(
            pick=tuple(cand),
            match_ev=round(float((decision_score_matrix * _points_grid(tuple(cand), g)).sum()), 3),
            p_rank1=float((me_rank == 1).mean()), p_top=p_top,
            median_rank=float(np.median(me_rank)),
            se_top=float(np.sqrt(p_top * (1 - p_top) / n_sims)), top_sample=top))
    ref = max(results, key=lambda r: r.match_ev)
    for r in results:
        if r is not ref:
            diff = r.top_sample.astype(float) - ref.top_sample.astype(float)
            r.diff_vs_evmax = float(diff.mean())
            r.diff_se = float(diff.std(ddof=1) / np.sqrt(diff.size))
    return sorted(results, key=lambda r: r.p_top, reverse=True)
