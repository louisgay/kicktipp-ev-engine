"""Hierarchical (partial-pooling) per-player pick model.

Predicts each opponent's scoreline pick as a distribution, anchored on the German
consensus (the population prior from the oracle) and bent toward the player's own
tendencies *only as their data accrues*. Three shrunk per-player signals:

  1. tendency-follow rate - how often player p's tendency matches the consensus
     (Beta-Binomial), shrunk toward the population mean with own-data weight
     w(n) = n / (n + follow_k);
  2. draw-lean - of the picks where p deviates from the consensus tendency, how
     much of that mass goes to a draw vs the opposite outcome (shrunk);
  3. exact-score scatter - within a tendency, p's distribution over scorelines
     (a reduced Dirichlet-multinomial: own counts shrunk toward a prior point mass
     on the consensus / modal score with weight w(n) = n / (n + exact_k)).

GRACEFUL DEGRADATION (guaranteed): with no data the population follow rate
defaults to 1.0, so ``pick_distribution`` collapses to a point mass on the
consensus pick - i.e. it equals ``oracle.field_picks_consensus``. Individuality
emerges only as discriminating matches accrue; a large history recovers the type.

``correlate_per_player`` may be passed as a SOFT input (``per_player_source``): it
nudges the per-player prior the follow rate is shrunk toward - never a hard
assignment, and the player's own data always dominates as n grows.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_PICKS = _ROOT / "data" / "opponents" / "picks.csv"
_CONFIG = _ROOT / "config" / "config.yaml"

# Modal fallback score per tendency (the prior when a tendency has no data and is
# not the consensus tendency).
_DEFAULT_SCORE = {"home": (1, 0), "away": (0, 1), "draw": (1, 1)}


def _tend(a: int, b: int) -> str:
    return "home" if a > b else ("draw" if a == b else "away")


def _market_modal_consensus() -> dict[tuple[int, int], tuple[int, int]]:
    """Per-match field anchor from snapshots.csv = modal scoreline of the blended
    Dixon-Coles matrix (its argmax), for every snapshotted match.

    The blend reproduces the live score engine: ``normalise(w·sharp_shin +
    (1-w)·kt)`` when a Shin sharp line is present, else pure kicktipp (both are
    already devigged in snapshots.csv). This deliberately matches the
    prediction-time consensus fallback in ``update._attach_relative_ev`` (matrix
    argmax), so the FIT reference and the DECISION-match reference agree.

    Returns ``{(spieltag, match_index): (home, away)}``; empty on any failure
    (graceful degradation - callers then fall back to whatever overrides exist).
    """
    try:
        import pandas as pd
        import yaml

        from src import snapshot
        from src.odds.reconstruct import reconstruct_matrix
    except Exception:  # pragma: no cover - import guard
        return {}
    try:
        w = float(yaml.safe_load(_CONFIG.read_text())
                  .get("model", {}).get("blend_weight", 0.0))
    except Exception:
        w = 0.0
    snaps = snapshot.load_history()
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for r in snaps.itertuples(index=False):
        try:
            kt = (float(r.kt_home), float(r.kt_draw), float(r.kt_away))
            if any(v != v for v in kt):           # NaN kicktipp line -> unusable
                continue
            sharp = (r.sharp_home, r.sharp_draw, r.sharp_away)
            if w > 0 and all(pd.notna(s) for s in sharp):
                blended = tuple(w * float(s) + (1 - w) * k for s, k in zip(sharp, kt))
                z = sum(blended)
                blended = tuple(b / z for b in blended)
            else:
                blended = kt
            ou = float(r.ou_over_2_5) if pd.notna(r.ou_over_2_5) else None
            mat = reconstruct_matrix(*blended, ou)
            g = mat.shape[0]
            fi = int(mat.argmax())
            out[(int(r.spieltag), int(r.match_index))] = (fi // g, fi % g)
        except Exception:
            continue
    return out


class FieldModel:
    def __init__(
        self,
        picks_df,
        consensus_lookup: dict[tuple[int, int], tuple[int, int]],
        *,
        follow_k: float = 5.0,
        exact_k: float = 3.0,
        draw_k: float | None = None,
        default_follow: float = 1.0,
        default_draw_share: float = 0.5,
        per_player_source: dict[str, float] | None = None,
        source_preds: dict | None = None,
        source_match: dict | None = None,
        min_discriminating: int = 8,
    ):
        self.follow_k = follow_k
        self.exact_k = exact_k
        self._draw_k_override = draw_k        # None => empirical-Bayes from the fit (below)
        self.default_follow = default_follow
        self.default_draw_share = default_draw_share
        self.per_player_source = per_player_source or {}
        self._source_preds = source_preds or {}        # (spieltag, match_index) -> {source: (a, b)}
        self._source_match = source_match or {}         # player -> {source: tendency-match rate}
        self.min_discriminating = min_discriminating
        self._disc: dict[str, int] = defaultdict(int)   # per-player discriminating-match count
        self._consensus = dict(consensus_lookup)
        self.players: set[str] = set()

        self._n_follow: dict[str, int] = defaultdict(int)
        self._m_follow: dict[str, int] = defaultdict(int)
        self._n_dev: dict[str, int] = defaultdict(int)
        self._m_draw_dev: dict[str, int] = defaultdict(int)
        self._exact: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

        self._pop_follow_num = self._pop_follow_den = 0
        self._pop_draw_num = self._pop_dev_den = 0

        if picks_df is not None and len(picks_df):
            for r in picks_df.itertuples(index=False):
                cons = consensus_lookup.get((int(r.spieltag), int(r.match_index)))
                if cons is None:
                    continue
                try:
                    pa, pb = (int(x) for x in str(r.pick).split("-"))
                except (ValueError, AttributeError):
                    continue
                ct, pt = _tend(*cons), _tend(pa, pb)
                p = r.player
                self.players.add(p)
                sp = self._source_preds.get((int(r.spieltag), int(r.match_index)))
                if sp and len({_tend(*v) for v in sp.values()}) > 1:
                    self._disc[p] += 1     # sources disagree AND this player's pick is observed
                self._n_follow[p] += 1
                self._pop_follow_den += 1
                if pt == ct:
                    self._m_follow[p] += 1
                    self._pop_follow_num += 1
                else:
                    self._n_dev[p] += 1
                    self._pop_dev_den += 1
                    if pt == "draw":
                        self._m_draw_dev[p] += 1
                        self._pop_draw_num += 1
                self._exact[p][pt][(pa, pb)] += 1

        # draw_share shrinkage constant: empirical-Bayes by default (computed once
        # from the fit), an explicit override otherwise.
        self.draw_k = (self._draw_k_override if self._draw_k_override is not None
                       else self._empirical_bayes_draw_k())

    def _empirical_bayes_draw_k(self) -> float:
        """Method-of-moments empirical-Bayes shrinkage constant for ``draw_share``.

        ``k = pop·(1-pop) / τ²_between``, where ``τ²_between`` is the between-player
        variance of the raw draw-among-deviations rate NET of within-player binomial
        noise (``mean_p pop·(1-pop)/n_dev_p``). This is soft-significance shrinkage:
        when the per-player fade signal is weak (noisy, small ``n_dev``), τ² is small,
        ``k`` is large, and every ``draw_share`` collapses toward ``pop_draw_share`` -
        the same principle as ``follow_k``, but calibrated to the data instead of the
        reused ``follow_k=5``. (At the MD1-4 bank: 0/12 players' draw_share survives a
        Bonferroni test, τ²≈0.037 => k≈6.7, modestly stronger than 5.) Falls back to
        ``follow_k`` when too few players carry deviations to estimate τ²."""
        pop = self.pop_draw_share
        raws, ns = [], []
        for p in self.players:
            n = self._n_dev.get(p, 0)
            if n > 0:
                raws.append(self._m_draw_dev.get(p, 0) / n)
                ns.append(n)
        if len(raws) < 2 or not (0.0 < pop < 1.0):
            return self.follow_k
        mean = sum(raws) / len(raws)
        v_obs = sum((r - mean) ** 2 for r in raws) / (len(raws) - 1)
        v_within = sum(pop * (1 - pop) / n for n in ns) / len(ns)
        tau2 = max(v_obs - v_within, 1e-4)
        return pop * (1 - pop) / tau2

    # -- construction from disk ----------------------------------------

    @classmethod
    def from_disk(cls, **kwargs) -> "FieldModel":
        import pandas as pd
        import yaml

        from src import oracle
        picks = pd.read_csv(_PICKS) if _PICKS.exists() else pd.DataFrame(
            columns=["player", "spieltag", "match_index", "pick"])
        # -- Follow/deviation reference anchor -----------------------------
        # BASE: the market-modal scoreline (blended-DC matrix argmax) for EVERY
        # snapshotted match, so the fit ingests ALL banked matchdays - not just
        # the MD1 matches the (MD1-only) oracle happens to cover. Without this
        # the model is frozen on MD1 (~8 picks/player) and never learns later
        # behaviour (e.g. a player's favourite-fade). Mirrors the prediction-time
        # fallback in update._attach_relative_ev (matrix argmax) so the FIT and
        # the DECISION-match anchors agree.
        cons: dict[tuple[int, int], tuple[int, int]] = _market_modal_consensus()
        # OVERRIDE: where German-media data exists (currently MD1), prefer the
        # oracle consensus - it carries a small media-specific signal the market
        # favourite lacks (empirically the field tracked an oracle contrarian
        # draw the market called a home win).
        if not picks.empty:
            for st in picks["spieltag"].dropna().unique():
                c = oracle.consensus(int(st))
                if c.empty:
                    continue
                for mi in c["match_index"]:
                    cp = oracle.consensus_pick(int(st), int(mi))
                    if cp:
                        cons[(int(st), int(mi))] = cp
        # prior strengths from config (fall back to constructor defaults)
        try:
            fm = yaml.safe_load(_CONFIG.read_text()).get("field_model", {}) or {}
            kwargs.setdefault("follow_k", float(fm.get("follow_prior_strength", 5.0)))
            kwargs.setdefault("exact_k", float(fm.get("exact_prior_strength", 3.0)))
        except Exception:
            pass
        # Per-source predictions + per-player source-tracking (for the gated mixture).
        from collections import defaultdict as _dd
        odf = oracle.load_oracle()
        source_preds = _dd(dict)
        for r in odf.itertuples(index=False):
            source_preds[(int(r.spieltag), int(r.match_index))][r.source] = (int(r.pred_home), int(r.pred_away))
        cpp = oracle.correlate_per_player()
        source_match: dict = {}
        if not cpp.empty:
            for player in cpp.index:
                source_match[player] = {s: float(cpp.loc[player, s]) / 100.0
                                        for s in cpp.columns if pd.notna(cpp.loc[player, s])}
        kwargs.setdefault("source_preds", dict(source_preds))
        kwargs.setdefault("source_match", source_match)
        return cls(picks, cons, **kwargs)

    # -- population priors ---------------------------------------------

    @property
    def pop_follow(self) -> float:
        return (self._pop_follow_num / self._pop_follow_den
                if self._pop_follow_den else self.default_follow)

    @property
    def pop_draw_share(self) -> float:
        return (self._pop_draw_num / self._pop_dev_den
                if self._pop_dev_den else self.default_draw_share)

    # -- shrunk per-player parameters ----------------------------------

    def follow_rate(self, player: str) -> float:
        n, m = self._n_follow.get(player, 0), self._m_follow.get(player, 0)
        prior = self.per_player_source.get(player, self.pop_follow)  # soft input
        w = n / (n + self.follow_k) if n else 0.0
        own = m / n if n else prior
        return w * own + (1 - w) * prior

    def draw_share(self, player: str) -> float:
        n, m = self._n_dev.get(player, 0), self._m_draw_dev.get(player, 0)
        pop = self.pop_draw_share
        w = n / (n + self.draw_k) if n else 0.0
        own = m / n if n else pop
        return w * own + (1 - w) * pop

    def _exact_dist(self, player: str, tendency: str,
                    anchor_score: tuple[int, int]) -> dict[tuple[int, int], float]:
        own = self._exact.get(player, {}).get(tendency, Counter())
        n = sum(own.values())
        w = n / (n + self.exact_k) if n else 0.0
        keys = set(own) | {anchor_score}
        return {k: w * (own.get(k, 0) / n if n else 0.0)
                + (1 - w) * (1.0 if k == anchor_score else 0.0)
                for k in keys}

    # -- prediction ----------------------------------------------------

    def pick_distribution_for_consensus(
        self, player: str, cons_tendency: str, cons_score: tuple[int, int],
    ) -> dict[tuple[int, int], float]:
        follow = self.follow_rate(player)
        rest = max(0.0, 1.0 - follow)
        tprob: dict[str, float] = {cons_tendency: follow}
        if cons_tendency in ("home", "away"):
            opp = "away" if cons_tendency == "home" else "home"
            ds = self.draw_share(player)
            tprob["draw"] = rest * ds
            tprob[opp] = rest * (1 - ds)
        else:  # consensus is a draw - split deviation across the two wins
            tprob["home"] = rest * 0.5
            tprob["away"] = rest * 0.5

        dist: dict[tuple[int, int], float] = defaultdict(float)
        for t, pt in tprob.items():
            if pt <= 0:
                continue
            anchor = cons_score if t == cons_tendency else _DEFAULT_SCORE[t]
            for s, ps in self._exact_dist(player, t, anchor).items():
                dist[s] += pt * ps
        tot = sum(dist.values())
        return {s: v / tot for s, v in dist.items()} if tot > 0 else {cons_score: 1.0}

    # -- per-player source mixture (gated) ----------------------------

    def discriminating_count(self, player: str) -> int:
        """Matches where the oracle sources DISAGREE and this player's pick is observed."""
        return self._disc.get(player, 0)

    def discriminating_counts(self) -> dict[str, int]:
        return {p: self._disc.get(p, 0) for p in sorted(self.players)}

    def source_weights(self, player: str) -> dict[str, float]:
        """Shrunk mixture weights over sources (from correlate_per_player), shrunk
        toward uniform with own-data weight w = n_disc / (n_disc + follow_k)."""
        sm = self._source_match.get(player, {})
        if not sm:
            return {}
        n = self._disc.get(player, 0)
        w = n / (n + self.follow_k)
        mean = sum(sm.values()) / len(sm)
        shrunk = {s: w * r + (1 - w) * mean for s, r in sm.items()}
        z = sum(shrunk.values())
        return {s: v / z for s, v in shrunk.items()} if z > 0 else {}

    def _mixture_dist(self, player, spieltag, match_index):
        sp = self._source_preds.get((int(spieltag), int(match_index)))
        wts = self.source_weights(player)
        if not sp or not wts:
            return None
        dist: dict[tuple[int, int], float] = defaultdict(float)
        for source, pick in sp.items():
            wt = wts.get(source, 0.0)
            if wt > 0:
                dist[tuple(pick)] += wt
        z = sum(dist.values())
        return {s: v / z for s, v in dist.items()} if z > 0 else None

    def pick_distribution(self, player: str, spieltag: int,
                          match_index: int) -> dict[tuple[int, int], float] | None:
        # TODO(flag #6, deferred): the gated mixture lives here, but
        # rank_sim.choose_pick still samples the DECISION match's opponents via
        # pick_distribution_for_consensus (Tier-2). Once any player crosses the
        # min_discriminating gate, thread (spieltag, match_index) into choose_pick
        # so the decision match also uses the mixture. Dormant until then (all
        # players currently below the gate), so not built.
        cons = self._consensus.get((int(spieltag), int(match_index)))
        if cons is None:
            return None
        # GATE: the per-player source mixture activates only with >= min_discriminating
        # matches where sources disagreed (and source data exists for this match);
        # otherwise fall back to the Tier-2 tendency model.
        if self.discriminating_count(player) >= self.min_discriminating:
            mix = self._mixture_dist(player, spieltag, match_index)
            if mix is not None:
                return mix
        return self.pick_distribution_for_consensus(player, _tend(*cons), cons)

    def sample_pick_for_consensus(self, player, cons_tendency, cons_score, rng) -> tuple[int, int]:
        dist = self.pick_distribution_for_consensus(player, cons_tendency, cons_score)
        scores = list(dist)
        idx = rng.choice(len(scores), p=[dist[s] for s in scores])
        return scores[idx]

    def sample_pick(self, player, spieltag, match_index, rng) -> tuple[int, int] | None:
        cons = self._consensus.get((int(spieltag), int(match_index)))
        if cons is None:
            return None
        return self.sample_pick_for_consensus(player, _tend(*cons), cons, rng)
