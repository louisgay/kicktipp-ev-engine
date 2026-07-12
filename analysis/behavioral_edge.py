"""Behavioral-EV-leak diagnostic - the generator-independent edge anchor.

The PREDICTIVE edge (forecasting scorelines better than the market) is undetectable
over one tournament: realized IC ≈ 0, everyone reads the same odds. The BEHAVIORAL
edge is different and measurable: under pool 4/3/2/0 scoring a draw forfeits the
goal-difference (3-pt) tier, so picking a draw on anything but a very-even low-scoring
match is *rules-based* -EV - a leak vs the EV-max pick, not a forecast. This module
measures that leak.

It is deliberately INDEPENDENT of the field model and the future-leg generator (the
broken MC-IR that swung 0.32->2.31 was the model grading its own homework). It uses
only three generator-independent inputs:

  * the SHARED market-odds Dixon-Coles score matrix per match (reconstructed from each
    match's banked pre-kickoff odds snapshot - the same blend the live score engine
    uses: sharp⊕kicktipp where a sharp line exists, else pure kicktipp, + O/U);
  * each player's OBSERVED pick;
  * the REALIZED result.

Two faces, reported separately (never conflated):
  (i)  EXPECTED leak  - picks + odds, NO outcome dependence. The "size of the prize",
       available now and every matchday. leak_p ≥ 0 by construction (EV-max is the max).
  (ii) REALIZED test  - paired self-vs-each-player point differentials + the
       leak<->realized-points regression. Uses results; carries the sampling noise.

Run:  python -m analysis.behavioral_edge            # report on the current bank
      python -m analysis.behavioral_edge --log      # also append t-stats to the CSV log

READ-ONLY: imports src primitives (reconstruct, scorer), never mutates picks/odds/model.
Dependency arrow is one-way (analysis -> src), per the repo contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.odds.reconstruct import reconstruct_lambdas, reconstruct_matrix
from src.scoring.kicktipp import expected_points, optimal_prediction, points

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data" / "history" / "snapshots.csv"
PICKS = ROOT / "data" / "opponents" / "picks.csv"
CONFIG = ROOT / "config" / "config.yaml"
LOG = ROOT / "data" / "history" / "behavioral_edge_log.csv"
SELF = "self"  # the pool participant running this engine (pseudonymised label)

# A draw is "harmless" (≈ EV-max) only on a very-even, low-scoring match; otherwise
# it forfeits the GD tier. These thresholds are for CLASSIFICATION/reporting only -
# the leak itself is computed exactly from the matrix, never bucketed.
_FAVOURITE_ABS_SUPREMACY = 1.0     # |λ_h - λ_a| above this => a clear net favourite
_HARMLESS_DRAW_LEAK = 0.05         # leak below this => the draw essentially was EV-max


def _blend_weight() -> float:
    try:
        import yaml
        return float(yaml.safe_load(CONFIG.read_text()).get("model", {}).get("blend_weight", 0.0))
    except Exception:
        return 0.0


def _parse_pick(s) -> tuple[int, int]:
    h, a = (int(x) for x in str(s).split("-"))
    return h, a


@dataclass
class MatchEval:
    """Everything generator-independent about one match: its market DC matrix, the
    EV-max pick under it, and the (supremacy, total) structure for classification."""
    spieltag: int
    match_index: int
    matrix: np.ndarray
    evmax_pick: tuple[int, int]
    evmax_ev: float
    supremacy: float
    total: float
    ou_source: str
    snapshot_at: str


def build_match_evals(snaps: pd.DataFrame | None = None, *, blend_weight: float | None = None
                      ) -> dict[tuple[int, int], MatchEval]:
    """Per-match market DC matrix + EV-max pick, keyed by (spieltag, match_index).

    Reconstructs each matrix from the banked snapshot's odds exactly as the live
    engine does. Skips a match only if its kicktipp line is missing (no evaluation
    distribution => that match cannot be scored)."""
    if snaps is None:
        snaps = pd.read_csv(SNAPSHOTS)
    w = _blend_weight() if blend_weight is None else blend_weight
    out: dict[tuple[int, int], MatchEval] = {}
    for r in snaps.itertuples(index=False):
        kt = (r.kt_home, r.kt_draw, r.kt_away)
        if any(pd.isna(v) for v in kt):
            continue
        kt = tuple(float(v) for v in kt)
        sharp = (r.sharp_home, r.sharp_draw, r.sharp_away)
        if w > 0 and all(pd.notna(s) for s in sharp):
            bl = tuple(w * float(s) + (1 - w) * k for s, k in zip(sharp, kt))
            z = sum(bl)
            bl = tuple(b / z for b in bl)
        else:
            bl = kt
        ou = float(r.ou_over_2_5) if pd.notna(r.ou_over_2_5) else None
        mat = reconstruct_matrix(*bl, ou)
        lam, mu, _ = reconstruct_lambdas(*bl, ou)
        pick, ev = optimal_prediction(mat)
        out[(int(r.spieltag), int(r.match_index))] = MatchEval(
            spieltag=int(r.spieltag), match_index=int(r.match_index), matrix=mat,
            evmax_pick=(int(pick[0]), int(pick[1])), evmax_ev=float(ev),
            supremacy=float(lam - mu), total=float(lam + mu),
            ou_source=("market" if ou is not None else "model"),
            snapshot_at=str(r.updated_at))
    return out


@dataclass
class PlayerLeak:
    player: str
    n: int
    leak_total: float
    leak_per_match: float
    leak_from_draws: float
    leak_from_other: float
    n_draws: int
    n_draws_on_fav: int
    n_draws_harmless: int
    leak_per_draw: float


def expected_leak(picks: pd.DataFrame, evals: dict[tuple[int, int], MatchEval]
                  ) -> dict[str, PlayerLeak]:
    """Face (i): per-player expected behavioral-EV leak - NO outcome dependence.

    leak_p = Σ_i ( E[pts | EV-max_i] - E[pts | pick_p,i] ) under the market matrix M_i.
    Decomposed into the part coming from draw picks vs other deviations, with a
    smart-vs-dumb-drawer split on the draw picks."""
    acc: dict[str, dict] = {}
    for r in picks.itertuples(index=False):
        key = (int(r.spieltag), int(r.match_index))
        ev = evals.get(key)
        if ev is None:
            continue
        pick = _parse_pick(r.pick)
        leak = ev.evmax_ev - expected_points(pick, ev.matrix)
        leak = max(0.0, leak)                       # guard fp noise; EV-max is the max
        is_draw = pick[0] == pick[1]
        a = acc.setdefault(r.player, dict(n=0, tot=0.0, draw=0.0, other=0.0,
                                          nd=0, ndfav=0, ndharm=0))
        a["n"] += 1
        a["tot"] += leak
        if is_draw:
            a["draw"] += leak
            a["nd"] += 1
            if abs(ev.supremacy) > _FAVOURITE_ABS_SUPREMACY:
                a["ndfav"] += 1
            if leak < _HARMLESS_DRAW_LEAK:
                a["ndharm"] += 1
        else:
            a["other"] += leak
    out = {}
    for p, a in acc.items():
        out[p] = PlayerLeak(
            player=p, n=a["n"], leak_total=a["tot"],
            leak_per_match=a["tot"] / a["n"] if a["n"] else 0.0,
            leak_from_draws=a["draw"], leak_from_other=a["other"],
            n_draws=a["nd"], n_draws_on_fav=a["ndfav"], n_draws_harmless=a["ndharm"],
            leak_per_draw=a["draw"] / a["nd"] if a["nd"] else 0.0)
    return out


@dataclass
class PairedResult:
    opponent: str
    n: int
    mu_d: float
    sigma_d: float
    ic: float
    t_now: float
    n_star_t2: float
    t_proj_104: float


def paired_vs_self(picks: pd.DataFrame, evals: dict[tuple[int, int], MatchEval],
                   *, horizon: int = 104) -> list[PairedResult]:
    """Face (ii-a): paired self-vs-each-player realized point differentials.

    d_i = pts_self,i - pts_p,i over matches BOTH played with a known result (same
    outcome => paired => low variance). Realized points are recomputed from pick+result
    with the canonical scorer (independent of the stored 'points' column)."""
    played = {k for k, ev in evals.items()}
    # realized points per (player, match)
    pts: dict[tuple[int, int], dict[str, int]] = {}
    for r in picks.itertuples(index=False):
        key = (int(r.spieltag), int(r.match_index))
        if key not in played or pd.isna(r.result):
            continue
        pts.setdefault(key, {})[r.player] = points(_parse_pick(r.pick), _parse_pick(r.result))
    out = []
    for opp in sorted({r.player for r in picks.itertuples(index=False)} - {SELF}):
        d = [m[SELF] - m[opp] for m in pts.values() if SELF in m and opp in m]
        d = np.array(d, dtype=float)
        n = len(d)
        if n < 2:
            continue
        mu, sd = float(d.mean()), float(d.std(ddof=1))
        ic = mu / sd if sd > 0 else float("nan")
        out.append(PairedResult(
            opponent=opp, n=n, mu_d=mu, sigma_d=sd, ic=ic,
            t_now=ic * np.sqrt(n) if sd > 0 else float("nan"),
            n_star_t2=(2 / ic) ** 2 if ic and not np.isnan(ic) and ic != 0 else float("inf"),
            t_proj_104=ic * np.sqrt(horizon) if sd > 0 else float("nan")))
    return out


@dataclass
class Regression:
    slope: float
    intercept: float
    ci_lo: float
    ci_hi: float
    r2: float
    p_value: float
    n: int


def leak_points_regression(picks: pd.DataFrame, evals: dict[tuple[int, int], MatchEval],
                           leaks: dict[str, PlayerLeak], *, relative_to_field: bool = False
                           ) -> Regression:
    """Face (ii-b): realized points-per-match (y) on expected leak-per-match (x), across
    players. slope ≈ -1 <=> a point of expected leak ≈ a point of realized underperformance."""
    # realized points per match per player
    rp: dict[str, list[int]] = {}
    field_by_match: dict[tuple[int, int], list[int]] = {}
    for r in picks.itertuples(index=False):
        key = (int(r.spieltag), int(r.match_index))
        if key not in evals or pd.isna(r.result):
            continue
        pt = points(_parse_pick(r.pick), _parse_pick(r.result))
        rp.setdefault(r.player, []).append(pt)
        field_by_match.setdefault(key, []).append(pt)
    field_mean = {k: float(np.mean(v)) for k, v in field_by_match.items()}
    xs, ys = [], []
    for p, lk in leaks.items():
        if p not in rp or not rp[p]:
            continue
        if relative_to_field:
            # realized points above the field mean, per match
            diffs = []
            for r in picks.itertuples(index=False):
                key = (int(r.spieltag), int(r.match_index))
                if r.player == p and key in field_mean and not pd.isna(r.result):
                    diffs.append(points(_parse_pick(r.pick), _parse_pick(r.result)) - field_mean[key])
            y = float(np.mean(diffs))
        else:
            y = float(np.mean(rp[p]))
        xs.append(lk.leak_per_match)
        ys.append(y)
    x, y = np.array(xs), np.array(ys)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # slope SE + 95% CI + p-value (t on n-2 df)
    if n > 2:
        sxx = float(((x - x.mean()) ** 2).sum())
        se = np.sqrt(ss_res / (n - 2) / sxx) if sxx > 0 else float("nan")
        from scipy.stats import t as tdist
        tcrit = tdist.ppf(0.975, n - 2)
        tstat = slope / se if se and not np.isnan(se) and se > 0 else float("nan")
        p_value = float(2 * tdist.sf(abs(tstat), n - 2)) if not np.isnan(tstat) else float("nan")
        ci_lo, ci_hi = slope - tcrit * se, slope + tcrit * se
    else:
        ci_lo = ci_hi = p_value = float("nan")
    return Regression(slope=float(slope), intercept=float(intercept), ci_lo=float(ci_lo),
                      ci_hi=float(ci_hi), r2=float(r2), p_value=float(p_value), n=n)


# -- reporting --------------------------------------------------------------


def format_report(picks, evals, leaks, paired, reg, reg_rel) -> str:
    L = []
    md = max(e.spieltag for e in evals.values())
    L.append(f"=== BEHAVIORAL-EDGE ANCHOR - {len(evals)} matches (MD1-{md}), "
             f"{len(picks)} picks ===")
    L.append("Evaluation distribution: market-odds Dixon-Coles per match's banked "
             "pre-kickoff snapshot")
    L.append("(MD1 = closing kicktipp+O/U, no sharp; MD2+ = blended sharp⊕kicktipp w=0.65).\n")

    L.append("-- FACE (i): EXPECTED leak per player (no outcomes - the prize) --")
    L.append(f"{'player':<11}{'n':>3}{'leak':>8}{'/match':>8}{'fromDraw':>9}"
             f"{'fromOth':>8}{'#draw':>6}{'drawFav':>8}{'lk/draw':>8}")
    for p in sorted(leaks.values(), key=lambda z: -z.leak_per_match):
        L.append(f"{p.player:<11}{p.n:>3}{p.leak_total:>8.2f}{p.leak_per_match:>8.3f}"
                 f"{p.leak_from_draws:>9.2f}{p.leak_from_other:>8.2f}{p.n_draws:>6}"
                 f"{p.n_draws_on_fav:>8}{p.leak_per_draw:>8.3f}")
    nonneg = all(p.leak_total >= -1e-9 for p in leaks.values())
    self_leak = leaks.get(SELF)
    L.append(f"  sanity: all leak_p ≥ 0 -> {nonneg};  "
             f"{SELF} leak = {self_leak.leak_total:.3f} over {self_leak.n} "
             f"({'pure EV-max' if self_leak.leak_total < 1e-6 else 'deviated - see below'})")

    L.append("\n-- FACE (ii-a): realized paired t - self vs each player --")
    L.append(f"{'opponent':<11}{'N':>4}{'mu_d':>8}{'sigma':>8}{'IC':>7}"
             f"{'t@N':>7}{'N*(t=2)':>9}{'t@104':>8}")
    for r in sorted(paired, key=lambda z: -z.t_now):
        nstar = "inf" if not np.isfinite(r.n_star_t2) else f"{r.n_star_t2:.0f}"
        L.append(f"{r.opponent:<11}{r.n:>4}{r.mu_d:>8.3f}{r.sigma_d:>8.3f}{r.ic:>7.3f}"
                 f"{r.t_now:>7.2f}{nstar:>9}{r.t_proj_104:>8.2f}")

    L.append("\n-- FACE (ii-b): anchor regression - realized pts/match on expected leak/match --")
    for name, R in (("points/match", reg), ("pts-above-field/match", reg_rel)):
        L.append(f"  [{name}] slope={R.slope:+.3f}  95%CI[{R.ci_lo:+.2f},{R.ci_hi:+.2f}]  "
                 f"R²={R.r2:.3f}  p={R.p_value:.3f}  n={R.n}")
    return "\n".join(L)


def append_log(paired, reg, *, spieltag: int, n_matches: int, log_path: Path = LOG) -> None:
    """Append this run's t-stats to the trajectory CSV (per opponent + regression slope)."""
    row = {"max_spieltag": spieltag, "n_matches": n_matches,
           "reg_slope": round(reg.slope, 4), "reg_ci_lo": round(reg.ci_lo, 4),
           "reg_ci_hi": round(reg.ci_hi, 4), "reg_r2": round(reg.r2, 4)}
    for r in paired:
        row[f"t_{r.opponent}"] = round(r.t_now, 4)
    df = pd.DataFrame([row])
    if log_path.exists():
        prev = pd.read_csv(log_path)
        df = pd.concat([prev[prev["max_spieltag"] != spieltag], df], ignore_index=True)
    df.to_csv(log_path, index=False)


def run(*, do_log: bool = False) -> dict:
    picks = pd.read_csv(PICKS)
    snaps = pd.read_csv(SNAPSHOTS)
    evals = build_match_evals(snaps)
    leaks = expected_leak(picks, evals)
    paired = paired_vs_self(picks, evals)
    reg = leak_points_regression(picks, evals, leaks, relative_to_field=False)
    reg_rel = leak_points_regression(picks, evals, leaks, relative_to_field=True)
    print(format_report(picks, evals, leaks, paired, reg, reg_rel))
    if do_log:
        md = max(e.spieltag for e in evals.values())
        append_log(paired, reg, spieltag=md, n_matches=len(evals))
        print(f"\n[logged trajectory -> {LOG.relative_to(ROOT)}]")
    return dict(evals=evals, leaks=leaks, paired=paired, reg=reg, reg_rel=reg_rel)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Behavioral-EV-leak edge anchor (read-only).")
    ap.add_argument("--log", action="store_true", help="append t-stats to the trajectory CSV")
    args = ap.parse_args(argv)
    run(do_log=args.log)


if __name__ == "__main__":
    main()
