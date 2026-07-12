"""Ad-hoc player-behaviour analysis across all banked picks (MD1-MD4 partial).

NOT production. Reads tracked CSVs only. Anchors follow/deviation on the MARKET
favourite (kt devigged 1X2 in snapshots.csv), since the production FieldModel
only ingests matchdays with oracle data (MD1). Reports per-player archetype +
an honest noise-floor read (Mauboussin null: a rate over n picks has
SD = sqrt(p(1-p)/n) under truly-identical players).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
picks = pd.read_csv(ROOT / "data/opponents/picks.csv")
snaps = pd.read_csv(ROOT / "data/history/snapshots.csv")


def tend(a, b):
    return "home" if a > b else ("draw" if a == b else "away")


def parse(p):
    h, a = (int(x) for x in str(p).split("-"))
    return h, a


# -- market reference per (spieltag, match_index) --------------------------
mkt = {}
for r in snaps.itertuples(index=False):
    if pd.isna(r.kt_home):
        continue
    probs = {"home": r.kt_home, "draw": r.kt_draw, "away": r.kt_away}
    modal_t = max(probs, key=probs.get)
    fav_side = "home" if r.kt_home >= r.kt_away else "away"
    fav_strength = max(r.kt_home, r.kt_away)
    mkt[(r.spieltag, r.match_index)] = dict(
        probs=probs, modal_t=modal_t, fav_side=fav_side, fav_strength=fav_strength)

# -- field-modal scoreline per match (plurality of player picks) -----------
field_mode = {}
for (st, mi), grp in picks.groupby(["spieltag", "match_index"]):
    c = Counter(grp["pick"])
    field_mode[(st, mi)] = parse(c.most_common(1)[0][0])

# -- per-pick classification -----------------------------------------------
rows = []
for r in picks.itertuples(index=False):
    key = (r.spieltag, r.match_index)
    if key not in mkt:
        continue
    m = mkt[key]
    ph, pa = parse(r.pick)
    pt = tend(ph, pa)
    follows = pt == m["modal_t"]
    # deviation direction (relative to market favourite)
    direction = None
    if not follows:
        if pt == "draw":
            direction = "draw"
        elif pt != m["fav_side"]:
            direction = "underdog"
        else:
            direction = "fav-other"  # modal_t was draw but player picked fav
    # pile-on: follows fav tendency but more extreme GD than field-modal score
    fm = field_mode[key]
    fav_gd = abs(ph - pa)
    modal_gd = abs(fm[0] - fm[1])
    pile_on = follows and m["modal_t"] == m["fav_side"] and fav_gd > modal_gd
    rows.append(dict(
        player=r.player, st=r.spieltag, mi=r.match_index, pick=r.pick, pt=pt,
        follows=follows, direction=direction, pile_on=pile_on,
        fav_strength=m["fav_strength"], modal_t=m["modal_t"],
        points=(int(r.points) if not pd.isna(r.points) else None),
        contested=(0.55 <= m["fav_strength"] <= 0.75),
        blowout=(m["fav_strength"] > 0.85)))

df = pd.DataFrame(rows)
players = sorted(df["player"].unique())

# -- pooled population follow rate -----------------------------------------
pop_follow = df["follows"].mean()
print(f"POOLED follow rate (vs market-modal tendency): {pop_follow:.3f}  "
      f"(n={len(df)} graded picks, {len(players)} players)")
print(f"POOLED deviation rate: {1-pop_follow:.3f}\n")

# -- per-player table ------------------------------------------------------
print(f"{'player':<11}{'n':>4}{'foll%':>7}{'dev':>5}{'->draw':>7}{'->dog':>6}"
      f"{'pile':>6}{'EVdev':>7}{'EVfol':>7}{'cont.dev%':>10}{'blow.dev%':>10}")
prof = {}
for p in players:
    sub = df[df["player"] == p]
    n = len(sub)
    foll = sub["follows"].mean()
    dev = sub[~sub["follows"]]
    ndev = len(dev)
    ndraw = (dev["direction"] == "draw").sum()
    ndog = (dev["direction"] == "underdog").sum()
    npile = sub["pile_on"].sum()
    # points: deviations vs consensus (follows), per-pick average (graded only)
    g = sub[sub["points"].notna()]
    ev_dev = g[~g["follows"]]["points"].mean() if len(g[~g["follows"]]) else float("nan")
    ev_fol = g[g["follows"]]["points"].mean() if len(g[g["follows"]]) else float("nan")
    cont = sub[sub["contested"]]
    blow = sub[sub["blowout"]]
    cdev = (1 - cont["follows"].mean()) if len(cont) else float("nan")
    bdev = (1 - blow["follows"].mean()) if len(blow) else float("nan")
    prof[p] = dict(n=n, foll=foll, ndev=ndev, ndraw=ndraw, ndog=ndog, npile=npile,
                   ev_dev=ev_dev, ev_fol=ev_fol, cdev=cdev, bdev=bdev)
    print(f"{p:<11}{n:>4}{foll*100:>6.0f}%{ndev:>5}{ndraw:>7}{ndog:>6}{npile:>6}"
          f"{ev_dev:>7.2f}{ev_fol:>7.2f}"
          f"{(cdev*100 if not math.isnan(cdev) else float('nan')):>9.0f}%"
          f"{(bdev*100 if not math.isnan(bdev) else float('nan')):>9.0f}%")

# -- NOISE FLOOR on follow-rate spread (Mauboussin null) -------------------
print("\n-- STATISTICAL HONESTY: follow-rate spread vs noise floor --")
foll_rates = [prof[p]["foll"] for p in players]
ns = [prof[p]["n"] for p in players]
obs_sd = pd.Series(foll_rates).std(ddof=1)
# expected between-player SD if all players shared pop_follow (Mauboussin):
# Var of an observed mean = p(1-p)/n; average across players, then sqrt.
exp_var = sum(pop_follow * (1 - pop_follow) / n for n in ns) / len(ns)
exp_sd = math.sqrt(exp_var)
print(f"observed between-player SD of follow rate : {obs_sd:.4f}")
print(f"expected SD if all 12 identical (p={pop_follow:.3f}): {exp_sd:.4f}")
print(f"ratio observed/expected                    : {obs_sd/exp_sd:.2f}x")
# chi-square homogeneity on follow counts
chi = sum((prof[p]["foll"] * prof[p]["n"] - pop_follow * prof[p]["n"]) ** 2
          / (pop_follow * (1 - pop_follow) * prof[p]["n"]) for p in players)
ddof = len(players) - 1
print(f"chi-square homogeneity stat: {chi:.2f} on {ddof} df  "
      f"(critical 95% ≈ {ddof + 1.645*math.sqrt(2*ddof):.1f}, "
      f"99% ≈ {ddof + 2.326*math.sqrt(2*ddof):.1f})")
try:
    from scipy.stats import chi2
    print(f"  p-value = {chi2.sf(chi, ddof):.4f}")
except Exception:
    pass

# -- DRAW-HEAVY ARCHETYPE focus --------------------------------------------
# One pool member is a systematic favourite-fader (elevated draw/underdog rate).
# Referenced by a stable pseudonym so the archetype stays legible in the code.
DRAW_HEAVY_ARCHETYPE = "p10"
print(f"\n-- {DRAW_HEAVY_ARCHETYPE} (draw-heavy archetype): fade-pattern significance --")
arch = df[df["player"] == DRAW_HEAVY_ARCHETYPE]
rn = len(arch)
rdev = arch[~arch["follows"]]
rfade = ((rdev["direction"] == "draw") | (rdev["direction"] == "underdog")).sum()
print(f"{DRAW_HEAVY_ARCHETYPE}: {rn} picks, {len(rdev)} deviations from market-modal tendency "
      f"({len(rdev)/rn*100:.0f}%); of those {rfade} toward draw/underdog (fade)")
print(f"  draws={ (rdev['direction']=='draw').sum() }, "
      f"underdogs={ (rdev['direction']=='underdog').sum() }, "
      f"pile-ons (overconfident-fav)={arch['pile_on'].sum()}")
# Is this player's deviation rate distinguishable from the pooled rate?
p0 = 1 - pop_follow
k, n = len(rdev), rn
se = math.sqrt(p0 * (1 - p0) / n)
z = (k / n - p0) / se
print(f"  {DRAW_HEAVY_ARCHETYPE} dev rate {k/n:.3f} vs pooled {p0:.3f}: z = {z:.2f}")
try:
    from scipy.stats import binomtest
    bt = binomtest(k, n, p0, alternative="greater")
    print(f"  one-sided binomial p(dev rate > pooled) = {bt.pvalue:.4f}")
except Exception:
    pass
# this player's net points
rg = arch[arch["points"].notna()]
print(f"  {DRAW_HEAVY_ARCHETYPE} realised pts/pick: deviations {rg[~rg['follows']]['points'].mean():.2f} "
      f"vs follows {rg[rg['follows']]['points'].mean():.2f}  "
      f"(total {int(rg['points'].sum())} over {len(rg)} graded)")
