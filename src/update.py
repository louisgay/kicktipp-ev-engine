"""Quota-aware pre-kickoff refresh / export chain.

One ``refresh(spieltag)`` does, in order:
  1. live-scrape kicktipp 1X2 (with an expired-session check),
  2. deadline-aware select: only matches kicking off within ``--lead-minutes``,
  3. quota guard: skip the credited fetch if a snapshot younger than
     ``--min-interval`` exists,
  4. ONE ``get_odds(markets="h2h,totals")`` (≈2 credits) - sharp h2h + totals +
     each match's commence_time,
  5. recompute the EV-max + decorrelation card (reuses recommend/decorrelate via
     KicktippOddsModel + scoring + decorrelate.cheapest_decorrelation), with the
     config blend toward sharp 1X2,
  6. bank everything via snapshot.record_match,
  7. write a dated CSV + Markdown card to data/exports/ and print the card.

OPERATIONAL CAVEATS (read me)
-----------------------------
* Requires KICKTIPP_SESSION (login cookie) and ODDS_API_KEY in .env.
* ``--watch`` keeps a process alive - the machine must stay ON and awake. The
  emitted crontab line is the robust alternative (survives reboots/sleep);
  prefer it for unattended operation.
* The Odds API free tier is ~500 credits/month and each fresh refresh costs ~2.
  The quota guard + deadline window mean the credited fetch only fires when a
  match is within the lead window AND no snapshot newer than --min-interval
  exists. NEVER run this per-minute; the suggested cron cadence is every 5 min,
  which is safe because most runs are no-ops.
* The kicktipp login cookie expires periodically; on expiry the scraper receives
  a login page instead of the prediction table. We detect that and log
  ``RE-AUTH NEEDED`` rather than crashing - refresh KICKTIPP_SESSION in .env.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from src import snapshot
from src.data.clean import normalise_team
from src.decorrelate import cheapest_decorrelation
from src.models.odds_model import KicktippOddsModel
from src.odds.client import get_odds
from src.odds.devig import devig_1x2, devig_over_under
from src.odds.reconstruct import _1x2_from_matrix, extract_odds_for_event
from src.scoring.kicktipp import optimal_prediction

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_EXPORT_DIR = _ROOT / "data" / "exports"
_CONFIG = _ROOT / "config" / "config.yaml"
_SPORT = "soccer_fifa_world_cup"

# A DEVIATE fires only when a candidate's P(rank<=target) beats the EV-max pick's by
# more than z times the CRN-paired SE of that difference. z=2 ≈ 95% one-sided: on a
# flat surface (mid-pack) nothing clears it -> EV-max; with a real gradient (far
# ahead/behind) a genuine deviation clears it. Replaces the old global-spread test,
# which a far-apart bold candidate could trip while the argmax stayed tied to EV-max.
_REL_GATE_Z = 2.0           # paired-SE multiples a DEVIATE must clear vs EV-max
_REL_SATURATION = 0.95      # P(top) at/above this is flagged as non-discriminating


class ReAuthNeeded(RuntimeError):
    """Raised when kicktipp returns a login page (session cookie expired)."""


# Belt-and-suspenders guard. src.update's windowed/quota refresh and --watch
# loop are the AUTONOMOUS, time-triggered path (cron / long-running process);
# the manual, run-on-demand prediction is src.pipeline. The CLI refuses to run
# autonomously unless the caller opts in explicitly.
_AUTONOMOUS_REFUSAL = (
    "REFUSING to run src.update without --i-know-this-is-autonomous.\n"
    "  src.update is the AUTONOMOUS / time-triggered path (its --watch loop and\n"
    "  the generated cron line do windowed, quota-gated pre-kickoff refreshes).\n"
    "  For a MANUAL, run-on-demand prediction use:\n"
    "        python -m src.pipeline --spieltag N\n"
    "  If you really intend an unattended/cron run, re-invoke with\n"
    "  --i-know-this-is-autonomous."
)


# -- time / config (wrapped so tests can inject) ----------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _config_blend_weight() -> float:
    try:
        cfg = yaml.safe_load(_CONFIG.read_text())
        return float(cfg.get("model", {}).get("blend_weight", 0.0))
    except Exception:
        return 0.0


def _key(home: str, away: str) -> tuple[str, str]:
    return (normalise_team(home), normalise_team(away))


# -- PURE helpers (unit-tested; no I/O) -------------------------------


def looks_like_login(html: str) -> bool:
    """Heuristic: kicktipp served a login page (expired cookie) not predictions."""
    h = html.lower()
    has_login = ('type="password"' in h) or ("kennwort" in h) or ('id="login-form"' in h)
    has_table = "tippabgabe" in h or "kicktipp-wettquote" in h or "tippabgabe-quoten" in h
    return has_login and not has_table


def parse_commence(s: str) -> datetime | None:
    """Parse an Odds-API ISO8601 commence_time to tz-aware UTC."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def matches_in_window(
    commence: dict[tuple[str, str], datetime],
    now: datetime,
    lead_minutes: int,
) -> set[tuple[str, str]]:
    """Keys whose kickoff is in [now, now + lead_minutes] (upcoming within window)."""
    lead = timedelta(minutes=lead_minutes)
    return {k for k, dt in commence.items() if timedelta(0) <= dt - now <= lead}


def next_refresh_time(
    commence: dict[tuple[str, str], datetime],
    now: datetime,
    lead_minutes: int,
) -> datetime | None:
    """When to next run refresh: now if a match is already in window, else the
    soonest (kickoff - lead) over not-yet-started matches; None if none remain."""
    lead = timedelta(minutes=lead_minutes)
    future = [dt for dt in commence.values() if dt - now > timedelta(0)]
    if not future:
        return None
    if any(dt - now <= lead for dt in future):
        return now
    return min(dt - lead for dt in future)


def should_refetch(snapshot_age_minutes: float | None, min_interval_minutes: float) -> bool:
    """Quota guard: fetch only if no recent snapshot exists for this matchday."""
    return snapshot_age_minutes is None or snapshot_age_minutes >= min_interval_minutes


def crontab_line(
    spieltag: int,
    *,
    lead_minutes: int = 45,
    min_interval: int = 30,
    every_minutes: int = 5,
    project_dir: str | None = None,
    python: str | None = None,
) -> str:
    """Ready-to-paste crontab line (alternative to --watch). Runs every
    ``every_minutes``; the quota guard + lead window make most runs no-ops."""
    project_dir = project_dir or str(_ROOT)
    python = python or f"{project_dir}/.venv/bin/python"
    # The line carries --i-know-this-is-autonomous: cron is the intended
    # unattended path, so the generated command opts in explicitly (the CLI
    # refuses to run autonomously without it).
    return (f"*/{every_minutes} * * * * cd {project_dir} && {python} -m src.update "
            f"--spieltag {spieltag} --lead-minutes {lead_minutes} "
            f"--min-interval {min_interval} --i-know-this-is-autonomous "
            f">> {project_dir}/data/exports/update.log 2>&1")


# -- live data access -------------------------------------------------


def _scrape_with_auth_check(spieltag: int):
    """Scrape a matchday's kicktipp 1X2; raise ReAuthNeeded on a login page."""
    from src.data.kicktipp_scrape import (
        BASE_URL, COMMUNITY, _fetch_page, _make_session, parse_prediction_page,
    )
    session = _make_session()
    soup = _fetch_page(session, f"{BASE_URL}/{COMMUNITY}/tippabgabe?spieltagIndex={spieltag}")
    html = str(soup)
    matches = parse_prediction_page(html)
    if not matches and looks_like_login(html):
        raise ReAuthNeeded(
            "kicktipp returned a login page - KICKTIPP_SESSION cookie has expired. "
            "Refresh the 'login' cookie value in .env."
        )
    return matches


def _extract_events(force_refresh: bool):
    """One get_odds(h2h,totals) call -> (sharp_records, totals_records, commence)."""
    events = get_odds(sport=_SPORT, markets="h2h,totals", regions="eu",
                      force_refresh=force_refresh)
    sharp: list[dict] = []
    totals: list[dict] = []
    commence: dict[tuple[str, str], datetime] = {}
    for ev in events:
        rec = extract_odds_for_event(ev)
        if not rec:
            continue
        k = _key(rec["home_team"], rec["away_team"])
        dt = parse_commence(rec.get("commence_time", ""))
        if dt:
            commence[k] = dt
        if rec.get("h2h_odds"):
            sharp.append({"home_team": rec["home_team"], "away_team": rec["away_team"],
                          "h2h_odds": rec["h2h_odds"]})
        if rec.get("totals_odds"):
            totals.append({"home_team": rec["home_team"], "away_team": rec["away_team"],
                           "totals_odds": rec["totals_odds"],
                           "totals_line": rec.get("totals_line", 2.5)})
    return sharp, totals, commence


def _snapshot_age_minutes(spieltag: int, now: datetime) -> float | None:
    df = snapshot.load_history()
    if df.empty or "updated_at" not in df.columns:
        return None
    sub = df[df["spieltag"] == spieltag]
    times = [parse_commence(str(t)) for t in sub.get("updated_at", []) if str(t) != "nan"]
    times = [t for t in times if t]
    if not times:
        return None
    return (now - max(times)).total_seconds() / 60.0


# -- card construction + export ---------------------------------------


@dataclass
class CardRow:
    index: int
    home: str
    away: str
    kickoff: str
    kt: tuple[float, float, float]
    sharp: tuple[float, float, float] | None
    blended: tuple[float, float, float]
    ou_over: float | None
    ev_pick: tuple[int, int]
    ev: float
    decorr_pick: tuple[int, int]
    delta_ev: float
    ou_source: str
    # transient matrix + relative-EV fields (populated by _attach_relative_ev)
    matrix: object = None
    rel_pick: tuple[int, int] | None = None
    rel_p_top: float | None = None
    rel_delta_ev: float | None = None
    rel_discriminating: bool | None = None   # did the candidates separate beyond noise?
    rel_evmax_p_top: float | None = None      # the EV-max pick's P(rank<=target)
    rel_target: int | None = None             # rank target the gate judged (1=win, 3=top-3)
    rel_p_win: float | None = None            # the chosen pick's P(finish 1st) (target-independent)
    rel_challenger: tuple[int, int] | None = None      # best non-EV-max candidate (gated or not)
    rel_challenger_gain: float | None = None  # its paired ΔP(rank<=target) vs EV-max
    rel_challenger_se: float | None = None    # paired-difference SE of that gain
    rel_gate_z: float | None = None           # SE-multiples a DEVIATE must clear (_REL_GATE_Z)
    # knockout (a.PSO) fields - populated by _build_card_knockout
    a_pso: bool = False                        # this match is scored a.PSO (knockout)
    advance_home: float | None = None          # kicktipp 2-way P(home advances)
    advance_realised: float | None = None      # P(home advances) realised by the a.PSO matrix


def _build_card(indexed_matches, sharp_records, totals_records, commence, blend_weight):
    model = KicktippOddsModel(blend_weight=blend_weight, require_ou=False)
    model.load_kicktipp_odds([m for _, m in indexed_matches])
    model.load_sharp_1x2(sharp_records)
    model.load_totals(totals_records)
    sharp_lk = {_key(r["home_team"], r["away_team"]): r["h2h_odds"] for r in sharp_records}
    ou_lk = {_key(r["home_team"], r["away_team"]): r["totals_odds"] for r in totals_records}

    rows = []
    for i, m in indexed_matches:
        k = _key(m.home_team, m.away_team)
        matrix = model.predict_score_matrix(*k)
        ev_pick, ev = optimal_prediction(matrix)
        dec_pick, dec_ev, _ = cheapest_decorrelation(matrix, ev_pick)
        kickoff = commence.get(k)
        rows.append(CardRow(
            index=i, home=m.home_team, away=m.away_team,
            kickoff=kickoff.isoformat() if kickoff else "",
            kt=(m.prob_home, m.prob_draw, m.prob_away),
            sharp=(devig_1x2(*sharp_lk[k], method="shin") if k in sharp_lk else None),
            blended=_1x2_from_matrix(matrix),
            ou_over=(devig_over_under(*ou_lk[k], method="normalise")[0] if k in ou_lk else None),
            ev_pick=ev_pick, ev=round(ev, 3),
            decorr_pick=dec_pick, delta_ev=round(dec_ev - ev, 3),
            ou_source=("market" if k in ou_lk else "model"),
            matrix=matrix,
        ))
    return rows


def _build_card_knockout(indexed_matches, sharp_records, totals_records, commence):
    """Build EV-max card rows for a.PSO knockout matches.

    Score engine (invariant preserved): a 90-minute regulation matrix is built
    the normal way from the 3-way *market* odds (sharp h2h ⊕ O/U -> reconstruct).
    The kicktipp 2-way advance price only sets the a.PSO resolution split (it
    never reaches the regulation matrix). ``apso_optimal_prediction`` then turns
    the regulation matrix into the a.PSO final-result distribution (draws removed,
    calibrated to the advance price) and picks argmax E[points] on it.

    The rank/decision (relative-EV) layer is intentionally NOT attached: the field
    model only ever saw group-stage picks and the horizon logic is knockout-blind
    (SYSTEM_MAP TODO(knockout)). Only the EV-max card is produced.
    """
    from src.odds.reconstruct import reconstruct_matrix
    from src.scoring.knockout import apso_optimal_prediction

    sharp_lk = {_key(r["home_team"], r["away_team"]): r["h2h_odds"] for r in sharp_records}
    ou_lk = {_key(r["home_team"], r["away_team"]): r["totals_odds"] for r in totals_records}

    rows = []
    for i, m in indexed_matches:
        k = _key(m.home_team, m.away_team)
        q_home = m.prob_home                       # kicktipp 2-way P(home advances)
        if k in sharp_lk:
            ph, pd, pa = devig_1x2(*sharp_lk[k], method="shin")
            over = devig_over_under(*ou_lk[k], method="normalise")[0] if k in ou_lk else None
            ou_source = "market" if k in ou_lk else "model"
            sharp = (ph, pd, pa)
        else:
            # No 3-way regulation odds for this fixture: infer a rough regulation
            # 1X2 from the advance price with a neutral draw share. Last-resort
            # safety net (normally every fixture matches the Odds API) - flagged.
            draw_share = 0.26
            ph = q_home * (1 - draw_share)
            pa = (1 - q_home) * (1 - draw_share)
            pd = draw_share
            over = None
            ou_source = "model"
            sharp = None
        reg_matrix = reconstruct_matrix(ph, pd, pa, over)
        pick, ev, final_matrix, info = apso_optimal_prediction(reg_matrix, q_home)
        dec_pick, dec_ev, _ = cheapest_decorrelation(final_matrix, pick)
        kickoff = commence.get(k)
        rows.append(CardRow(
            index=i, home=m.home_team, away=m.away_team,
            kickoff=kickoff.isoformat() if kickoff else "",
            kt=(m.prob_home, 0.0, m.prob_away),    # kicktipp 2-way advance (draw=0)
            sharp=sharp,
            blended=_1x2_from_matrix(reg_matrix),  # regulation 1X2 (scoreline shape)
            ou_over=over,
            ev_pick=pick, ev=round(ev, 3),
            decorr_pick=dec_pick, delta_ev=round(dec_ev - ev, 3),
            ou_source=ou_source,
            matrix=final_matrix,
            a_pso=True, advance_home=round(q_home, 4),
            advance_realised=round(info.realised_q_home, 4),
        ))
    return rows


def _candidate_picks(row, n_top: int = 4):
    """Decision-pick candidates for the relative-EV optimiser: EV-max + decorr +
    the most probable scorelines, de-duplicated."""
    mat = row.matrix
    g = mat.shape[0]
    ranked = sorted(((mat[i, j], (i, j)) for i in range(g) for j in range(g)), reverse=True)
    out, seen = [], set()
    for c in [tuple(row.ev_pick), tuple(row.decorr_pick)] + [s for _, s in ranked[:n_top]]:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _attach_relative_ev(rows, spieltag, current_totals, me, horizon, field_model,
                        *, target: int = 1, knockout: bool = False):
    """Populate rel_pick / rel_p_top / rel_delta_ev via the relative-EV optimiser.

    ``target`` is the rank the DEVIATE gate optimises for: ``1`` = P(finish 1st)
    (the live objective once top-3 is secured - the pipeline default), ``3`` =
    P(rank<=3) (the legacy band). Only the gate's boundary changes; the EV-max
    pick is always the action default, and ``rel_p_win`` (P-of-1st) is recorded
    regardless of ``target`` so the win odds are always visible on the card.

    ``knockout``: a.PSO mode - the decision matrix is draw-free and opponents'
    picks/future legs are projected onto decisive scorelines (see
    ``rank_sim.choose_pick``). Until opponents make real knockout picks this reuses
    their group-stage tendencies, so treat the P(win) as indicative.
    """
    from src import oracle
    from src.rank_sim import choose_pick
    for r in rows:
        if r.matrix is None:
            continue
        cp = oracle.consensus_pick(spieltag, r.index)
        if cp is None:                          # fallback: modal scoreline of the matrix
            g = r.matrix.shape[0]
            fi = int(r.matrix.argmax())
            cp = (fi // g, fi % g)
        cons_tend = "home" if cp[0] > cp[1] else ("draw" if cp[0] == cp[1] else "away")
        res = choose_pick(r.matrix, (cons_tend, cp), _candidate_picks(r), field_model,
                          current_totals, me, horizon=horizon, target=target,
                          n_sims=15_000, seed=r.index, knockout=knockout)
        # Honest discrimination, vs the EV-max pick directly. A candidate may DEVIATE
        # only if its P(rank<=target) beats EV-max's by more than the CRN-PAIRED noise
        # floor (z·diff_se; the shared-randomness SE of the *difference*, ~4x tighter
        # than the marginal SE). This kills the old failure mode where a large global
        # spread - driven by an unrelated bold candidate - let the argmax through even
        # when it was statistically tied with EV-max at negative EV. diff_vs_evmax /
        # diff_se are computed in choose_pick relative to the max-match_ev pick.
        evmax = max(res, key=lambda x: x.match_ev)        # the EV-max pick's result
        r.rel_target = target
        r.rel_gate_z = _REL_GATE_Z
        r.rel_evmax_p_top = round(evmax.p_top, 4)
        eligible = [x for x in res
                    if x is not evmax
                    and x.diff_vs_evmax is not None and x.diff_se is not None
                    and x.diff_vs_evmax > _REL_GATE_Z * x.diff_se]   # clears paired floor
        #   (diff_vs_evmax > 0 by the inequality => a strictly dominated candidate
        #    - ΔP(target)<=0 - never fires; exact ties break toward EV-max.)
        # Top challenger = best non-EV-max candidate by P(rank<=target), reported even
        # when it FAILS the gate so the "how close did a deviation come" math is visible.
        # Prefer candidates that carry paired diff math (choose_pick leaves its own
        # EV-reference's diff None; under an EV tie that reference may differ from our
        # ``evmax``, so fall back to the full pool only if none carry math).
        non_evmax = [x for x in res if x is not evmax]
        with_math = [x for x in non_evmax if x.diff_vs_evmax is not None]
        pool = with_math or non_evmax
        challenger = max(pool, key=lambda x: x.p_top) if pool else None
        if challenger is not None:
            r.rel_challenger = challenger.pick
            r.rel_challenger_gain = (round(challenger.diff_vs_evmax, 4)
                                     if challenger.diff_vs_evmax is not None else None)
            r.rel_challenger_se = (round(challenger.diff_se, 4)
                                   if challenger.diff_se is not None else None)
        r.rel_discriminating = bool(eligible)
        if eligible:
            best = max(eligible, key=lambda x: x.p_top)    # max eligible P(rank<=target)
            r.rel_pick = best.pick
            r.rel_p_top = round(best.p_top, 4)
            r.rel_p_win = round(best.p_rank1, 4)
            r.rel_delta_ev = round(best.match_ev - r.ev, 3)   # EV cost of the rank-optimal pick
        else:
            r.rel_pick = tuple(r.ev_pick)                  # EV-max is rank-optimal
            r.rel_p_top = round(evmax.p_top, 4)
            r.rel_p_win = round(evmax.p_rank1, 4)
            r.rel_delta_ev = 0.0


def _write_exports(spieltag: int, rows: list[CardRow], now: datetime) -> tuple[Path, Path]:
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    csv_path = _EXPORT_DIR / f"md{spieltag}_{stamp}.csv"
    md_path = _EXPORT_DIR / f"md{spieltag}_{stamp}.md"

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spieltag", "match_index", "home", "away", "kickoff",
                    "kt_h", "kt_d", "kt_a", "sharp_h", "sharp_d", "sharp_a",
                    "blend_h", "blend_d", "blend_a", "ou_over_2_5",
                    "ev_pick", "ev", "decorr_pick", "delta_ev", "ou_source",
                    "rel_target", "rel_pick", "rel_p_top", "rel_p_win", "rel_delta_ev",
                    "rel_challenger", "rel_challenger_gain", "rel_challenger_se", "rel_gate_z"])
        for r in rows:
            sh = r.sharp or ("", "", "")
            w.writerow([spieltag, r.index, r.home, r.away, r.kickoff,
                        *[round(x, 4) for x in r.kt],
                        *([round(x, 4) for x in sh] if r.sharp else ["", "", ""]),
                        *[round(x, 4) for x in r.blended],
                        round(r.ou_over, 4) if r.ou_over is not None else "",
                        r.rel_target if r.rel_target is not None else "",
                        f"{r.rel_pick[0]}-{r.rel_pick[1]}" if r.rel_pick else "",
                        r.rel_p_top if r.rel_p_top is not None else "",
                        r.rel_p_win if r.rel_p_win is not None else "",
                        r.rel_delta_ev if r.rel_delta_ev is not None else "",
                        f"{r.rel_challenger[0]}-{r.rel_challenger[1]}" if r.rel_challenger else "",
                        r.rel_challenger_gain if r.rel_challenger_gain is not None else "",
                        r.rel_challenger_se if r.rel_challenger_se is not None else "",
                        r.rel_gate_z if r.rel_gate_z is not None else ""])

    md_path.write_text(_format_card_md(spieltag, rows, now))
    return csv_path, md_path


def _format_card_md(spieltag: int, rows: list[CardRow], now: datetime) -> str:
    lines = [f"# Matchday {spieltag} - refresh card ({now.isoformat()})", ""]

    if any(getattr(r, "a_pso", False) for r in rows):
        # Knockout (a.PSO) EV-max card. The pick is scored on the FINAL result
        # incl. extra time + penalties (never a draw); "Advance" is kicktipp's
        # 2-way price, "Reg 1X2" the 90-minute market shape behind the scoreline.
        lines += ["## EV-max card - a.PSO knockout "
                  "(result incl. extra time + penalties; no draws)",
                  "| Match | Kickoff (UTC) | a.PSO pick | EV | Decorr | ΔEV | "
                  "Advance (kt) | Reg 1X2 (90m) | O/U |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ko = r.kickoff[11:16] if len(r.kickoff) >= 16 else "?"
            lines.append(
                f"| {r.home} v {r.away} | {ko} | "
                f"{r.ev_pick[0]}-{r.ev_pick[1]} | {r.ev:.2f} | "
                f"{r.decorr_pick[0]}-{r.decorr_pick[1]} | {r.delta_ev:+.2f} | "
                f"{(r.advance_home or 0.0):.0%} | "
                f"{r.blended[0]:.0%}/{r.blended[1]:.0%}/{r.blended[2]:.0%} | {r.ou_source} |")
        lines += ["", "_a.PSO: the result counts extra time + penalties, so it is never "
                  "a draw - the pick's tendency = who advances (Advance = kicktipp 2-way "
                  "price). Reg 1X2 is the 90-minute market shape behind the scoreline. "
                  "Decision card below uses a knockout rank sim with opponents' picks "
                  "projected onto decisive scorelines (group-stage tendencies; no real "
                  "knockout picks yet - P(win) is indicative)._"]
    else:
        lines += ["## EV-max card",
                  "| Match | Kickoff (UTC) | EV-max | EV | Decorr | ΔEV | Blend H/D/A | O/U |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ko = r.kickoff[11:16] if len(r.kickoff) >= 16 else "?"
            lines.append(
                f"| {r.home} v {r.away} | {ko} | "
                f"{r.ev_pick[0]}-{r.ev_pick[1]} | {r.ev:.2f} | "
                f"{r.decorr_pick[0]}-{r.decorr_pick[1]} | {r.delta_ev:+.2f} | "
                f"{r.blended[0]:.0%}/{r.blended[1]:.0%}/{r.blended[2]:.0%} | {r.ou_source} |")

    if any(r.rel_pick is not None for r in rows):
        target = next((r.rel_target for r in rows if r.rel_target is not None), 1)
        tgt_label = "top-1 (win)" if target == 1 else f"top-{target}"
        lines += ["", f"## Decision card - target: {tgt_label}",
                  "| Match | EV-max | Rank-opt | P(win) | ΔEV | What's happening |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            if r.rel_pick is None:
                continue
            lines.append(
                f"| {r.home} v {r.away} | {r.ev_pick[0]}-{r.ev_pick[1]} | "
                f"{r.rel_pick[0]}-{r.rel_pick[1]} | "
                f"{(r.rel_p_win or 0.0):.1%} | {(r.rel_delta_ev or 0.0):+.2f} | "
                f"{_decision_verdict(r)} |")
    return "\n".join(lines)


def _decision_verdict(r: CardRow) -> str:
    """The 'what's happening' diagnostic: HOLD vs DEVIATE plus the gate math.

    Always shows the top challenger's paired ΔP(rank<=target) ± SE against the
    gate (``_REL_GATE_Z``·SE), so the reason a deviation did or did not fire is
    legible even on HOLD rows.
    """
    band = "P(top-1)" if r.rel_target == 1 else f"P(top-{r.rel_target})"
    gate_z = r.rel_gate_z if r.rel_gate_z is not None else _REL_GATE_Z
    # challenger ΔP ± SE string (may be absent if no non-EV-max candidate existed)
    if r.rel_challenger is not None and r.rel_challenger_gain is not None \
            and r.rel_challenger_se is not None:
        cp = f"{r.rel_challenger[0]}-{r.rel_challenger[1]}"
        math = f"{r.rel_challenger_gain:+.1%}±{r.rel_challenger_se:.1%}"
    else:
        cp, math = None, None

    if r.rel_discriminating and tuple(r.rel_pick) != tuple(r.ev_pick):
        # A DEVIATE fired: the chosen rank pick IS the challenger that cleared the gate.
        return (f"DEVIATE -> {r.rel_pick[0]}-{r.rel_pick[1]} · {band} {math} "
                f"clears {gate_z:.1f}·SE gate · EV cost {r.rel_delta_ev:+.2f}")
    # HOLD - EV-max is rank-optimal. Report how close the best challenger came.
    if cp is None:
        return f"HOLD · EV-max is rank-optimal · {band} {(r.rel_p_top or 0.0):.1%}"
    return (f"HOLD · EV-max is rank-optimal · challenger {cp} {band} {math} "
            f"fails {gate_z:.1f}·SE gate")


# -- orchestration ----------------------------------------------------


def refresh(
    spieltag: int,
    *,
    lead_minutes: int = 45,
    min_interval: int = 30,
    force: bool = False,
    now: datetime | None = None,
) -> list[CardRow] | None:
    """Run the full refresh chain for one matchday. Returns the card, or None
    if skipped (no match in window / quota guard / re-auth needed)."""
    now = now or _now()

    try:
        matches = _scrape_with_auth_check(spieltag)
    except ReAuthNeeded as e:
        logger.error("RE-AUTH NEEDED: %s", e)
        return None
    if not matches:
        logger.warning("No matches for spieltag %d (none posted or all played).", spieltag)
        return None

    # Cheap (cached) odds read for commence_time -> window selection.
    _, _, commence = _extract_events(force_refresh=False)
    if force:
        sel = list(enumerate(matches))
    else:
        win = matches_in_window(commence, now, lead_minutes)
        sel = [(i, m) for i, m in enumerate(matches) if _key(m.home_team, m.away_team) in win]
        if not sel:
            logger.info("No matches kicking off within %d min of %s - skipping refresh.",
                        lead_minutes, now.isoformat())
            return None

    # Quota guard: skip the credited fetch if a snapshot is younger than min_interval.
    age = _snapshot_age_minutes(spieltag, now)
    if not force and not should_refetch(age, min_interval):
        logger.info("Quota guard: snapshot is %.0f min old (< %d) - skipping refetch.",
                    age, min_interval)
        return None

    # Credited fresh fetch (~2 credits).
    sharp_records, totals_records, commence = _extract_events(force_refresh=True)

    # Cross-check kicktipp deadline vs Odds-API commence_time.
    for i, m in sel:
        k = _key(m.home_team, m.away_team)
        if k in commence:
            logger.debug("commence %s vs %s: kicktipp='%s' oddsapi=%s",
                         m.home_team, m.away_team, m.datetime_str, commence[k].isoformat())

    blend_weight = _config_blend_weight()
    rows = _build_card(sel, sharp_records, totals_records, commence, blend_weight)

    # Relative-EV card (rank-optimal). Best-effort; needs standings + field model.
    try:
        from src import oracle as _oracle
        from src.data.leaderboard import scrape_leaderboard
        from src.field_model import FieldModel
        from src.rank_sim import remaining_match_count
        current_totals = {p.name: p.total for p in scrape_leaderboard(spieltag=spieltag).players}
        horizon = max(0, remaining_match_count().get("remaining", 0) - len(rows))
        if _oracle.SELF in current_totals:
            _attach_relative_ev(rows, spieltag, current_totals, _oracle.SELF, horizon,
                                FieldModel.from_disk())
            logger.info("Relative-EV card attached (horizon=%d).", horizon)
    except Exception as e:
        logger.warning("Relative-EV card skipped (non-fatal): %s", e)

    for r in rows:
        snapshot.record_match(
            spieltag, r.index, r.home, r.away,
            kicktipp_1x2=r.kt, sharp_1x2=r.sharp, ou_over_2_5=r.ou_over)

    csv_path, md_path = _write_exports(spieltag, rows, now)
    print(_format_card_md(spieltag, rows, now))
    logger.info("Refreshed %d match(es); exported %s + %s (blend_weight=%.2f).",
                len(rows), csv_path.name, md_path.name, blend_weight)
    return rows


def watch(spieltag: int, *, lead_minutes: int = 45, min_interval: int = 30,
          max_sleep_minutes: int = 30, i_know_autonomous: bool = False) -> None:
    """Loop: sleep until the next (kickoff - lead), run refresh, repeat.

    Uses cached commence_time for scheduling (0 credits; kickoff times are
    static); only refresh() spends credits, gated by the quota guard.

    Autonomous by nature - refuses to start unless ``i_know_autonomous`` is set
    (the CLI maps this to --i-know-this-is-autonomous). Use src.pipeline for
    manual runs.
    """
    if not i_know_autonomous:
        raise RuntimeError(_AUTONOMOUS_REFUSAL)
    logger.info("watch: spieltag=%d lead=%dmin min_interval=%dmin "
                "(machine must stay awake; Ctrl-C to stop)",
                spieltag, lead_minutes, min_interval)
    while True:
        _, _, commence = _extract_events(force_refresh=False)
        now = _now()
        nxt = next_refresh_time(commence, now, lead_minutes)
        if nxt is None:
            logger.info("watch: no upcoming matches with odds - done.")
            return
        if nxt <= now:
            refresh(spieltag, lead_minutes=lead_minutes, min_interval=min_interval)
            time.sleep(min_interval * 60)              # respect quota after a refresh
        else:
            sleep_s = min((nxt - now).total_seconds(), max_sleep_minutes * 60)
            logger.info("watch: sleeping %.0f min (next window at %s).",
                        sleep_s / 60, nxt.isoformat())
            time.sleep(max(sleep_s, 1))


# -- CLI ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Quota-aware pre-kickoff refresh/export chain.")
    p.add_argument("--spieltag", type=int, required=True)
    p.add_argument("--lead-minutes", type=int, default=45,
                   help="only refresh matches kicking off within this window (default 45)")
    p.add_argument("--min-interval", type=int, default=30,
                   help="quota guard: skip refetch if a snapshot is younger than this (min)")
    p.add_argument("--force", action="store_true", help="ignore window + quota guard")
    p.add_argument("--watch", action="store_true", help="loop until matchday is done")
    p.add_argument("--crontab", action="store_true",
                   help="print a ready-to-paste crontab line and exit")
    p.add_argument("--i-know-this-is-autonomous", action="store_true",
                   help="opt in to the autonomous refresh/--watch path "
                        "(manual runs should use python -m src.pipeline instead)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    if args.crontab:   # just prints a line (which itself carries the opt-in flag)
        print(crontab_line(args.spieltag, lead_minutes=args.lead_minutes,
                           min_interval=args.min_interval))
        return

    # Guard: src.update only runs autonomously with the explicit opt-in.
    if not args.i_know_this_is_autonomous:
        print(_AUTONOMOUS_REFUSAL)
        return

    if args.watch:
        watch(args.spieltag, lead_minutes=args.lead_minutes,
              min_interval=args.min_interval, i_know_autonomous=True)
        return
    refresh(args.spieltag, lead_minutes=args.lead_minutes,
            min_interval=args.min_interval, force=args.force)


if __name__ == "__main__":
    main()
