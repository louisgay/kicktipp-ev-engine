"""Single on-demand "refresh-everything + health-check" pipeline.

ONE manual command - ``python -m src.pipeline --spieltag N`` - runs NOW, with
no lead window, no quota gating, no loop and no cron. It refreshes everything
for the matches that have NOT yet kicked off and ends with a health/staleness
report. It REUSES the existing engine helpers (it does not re-implement the
model):

  1. COLLECT (a posteriori)  - ``opponents.update_log`` appends newly-visible
     player picks to picks.csv; ``snapshot.backfill_results`` writes realised
     scores for matches that kicked off since the last run. (Picks unlock at
     kickoff, so every run catches the field-model data up.)
  2. REFRESH ODDS            - scrape kicktipp 1X2 + ONE
     ``get_odds(markets="h2h,totals", force_refresh=True)`` (bypasses the
     no-TTL cache so the line is fresh).
  3. RECOMPUTE               - EV-max + decorrelation + relative-EV cards from
     the freshly-updated field model (``update._build_card`` /
     ``update._attach_relative_ev``).
  4. SNAPSHOT ODDS           - upsert the full tuple (kicktipp 1X2, Shin sharp
     1X2, consensus O/U) per match, with capture-time ``updated_at`` and an
     additive ``lead_minutes_to_kickoff`` column (from the Odds-API
     commence_time) so the snapshot doubles as an odds time-series.
  5. EXPORT                  - dated CSV + Markdown card to data/exports/
     (``update._write_exports``). These also serve as the odds time-series.
  6. PRINT                   - both cards to stdout. NEVER submits to kicktipp.
  7. HEALTH / STALENESS      - a compact PASS/WARN table vs the most recent
     PRIOR dated export (see ``health_report``). WARN, never crash.

This command deliberately bypasses src.update's window/quota/--watch logic.
That machinery is kept (still tested) for unattended operation; this is the
manual, run-on-demand path the user triggers a few hours before kickoff.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import snapshot
from src.update import (
    ReAuthNeeded,
    _attach_relative_ev,
    _build_card,
    _build_card_knockout,
    _config_blend_weight,
    _extract_events,
    _format_card_md,
    _key,
    _scrape_with_auth_check,
    _write_exports,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_EXPORT_DIR = _ROOT / "data" / "exports"

# Acceptable raw kicktipp overround band (kicktipp posts ~vig-free, ~1.00).
_OVERROUND_LO, _OVERROUND_HI = 0.98, 1.06
_FIT_ERR_WARN = 0.01


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError, TypeError):
        return None


def _lead_minutes(kickoff_iso: str, now: datetime) -> float | None:
    dt = _parse_iso(kickoff_iso) if kickoff_iso else None
    return (dt - now).total_seconds() / 60.0 if dt else None


def _is_upcoming(commence_dt, fixture_result, now: datetime) -> bool:
    """Predict a match only if it has NOT started.

    The kicktipp leaderboard shows a (provisional) result the instant a match
    KICKS OFF, so a non-None ``fixture_result`` means STARTED (live OR finished)
    -> off the card, and never re-predicted (the result only persists/grows, so
    a moving live score can't flip it back to upcoming). This is a "started"
    test, NOT a "finished" test - both live and finished must stay off the card.

    Uses the leaderboard (which never drops a match) instead of the Odds-API
    commence alone (which vanishes once a match finishes -> the old spurious
    off-board rows). When no leaderboard row exists (fixture_result is None
    because the scrape was empty/failed), it falls back to the commence rule, so
    behaviour degrades to the previous commence-only selector.
    """
    if fixture_result is not None:
        return False                       # STARTED (live or finished)
    return commence_dt is None or commence_dt > now   # fallback: future/unknown kickoff


# -- prior export discovery (for the staleness comparison) -------------


def _find_prior_export(spieltag: int) -> Path | None:
    """Most recent existing md{spieltag}_*.csv (the previous run). Call BEFORE
    writing this run's export so we compare against the last run, not ourselves."""
    if not _EXPORT_DIR.exists():
        return None
    cands = sorted(_EXPORT_DIR.glob(f"md{spieltag}_*.csv"))
    return cands[-1] if cands else None


def _load_prior_rows(path: Path | None) -> dict[int, dict]:
    """Prior export rows keyed by match_index (empty if no prior export)."""
    if path is None or not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["match_index"])] = r
            except (KeyError, ValueError):
                continue
    return out


# -- health / staleness report -----------------------------------------


@dataclass
class HealthRow:
    status: str      # "PASS" | "WARN"
    category: str
    scope: str
    detail: str


def _f(x) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return ""


def _market_inputs_identical(row, prior: dict) -> tuple[bool, bool, bool, bool]:
    """(kt_same, sharp_same, ou_same, evpick_same) vs a prior export row."""
    kt_same = all(_f(c) == (prior.get(k, "") or "")
                  for c, k in zip(row.kt, ("kt_h", "kt_d", "kt_a")))
    if row.sharp is None:
        sharp_same = all((prior.get(k, "") or "") == "" for k in ("sharp_h", "sharp_d", "sharp_a"))
    else:
        sharp_same = all(_f(c) == (prior.get(k, "") or "")
                         for c, k in zip(row.sharp, ("sharp_h", "sharp_d", "sharp_a")))
    if row.ou_over is None:
        ou_same = (prior.get("ou_over_2_5", "") or "") == ""
    else:
        ou_same = _f(row.ou_over) == (prior.get("ou_over_2_5", "") or "")
    evpick_same = f"{row.ev_pick[0]}-{row.ev_pick[1]}" == (prior.get("ev_pick", "") or "")
    return kt_same, sharp_same, ou_same, evpick_same


def health_report(
    spieltag: int,
    rows: list,
    indexed_matches: list,
    *,
    n_scraped: int,
    sharp_matched: int,
    fixtures: list,
    picks_df,
    snap_df,
    prior_rows: dict[int, dict],
    prior_existed: bool,
    picks_added: int,
    results_backfilled: int,
) -> list[HealthRow]:
    """Compare this run to the most recent prior export; WARN on problems.

    Categories: missing/empty, unchanged-from-last (the key one), not-updated,
    degraded-input. Never raises - every issue is a row, not an exception.
    """
    from src.odds.reconstruct import reconstruct_lambdas

    out: list[HealthRow] = []
    match_by_idx = {i: m for i, m in indexed_matches}

    # -- MISSING / EMPTY (run level) ----------------------------------
    if n_scraped == 0:
        out.append(HealthRow("WARN", "missing/empty", "kicktipp",
                             "scrape returned 0 matches (re-auth? wrong spieltag?)"))
    else:
        out.append(HealthRow("PASS", "scrape", "kicktipp",
                             f"{n_scraped} matches scraped, {len(rows)} not yet kicked off"))
    if rows and sharp_matched == 0:
        out.append(HealthRow("WARN", "missing/empty", "sharp",
                             f"sharp h2h matched 0/{len(rows)} - blend inert (pure kicktipp)"))
    elif rows:
        out.append(HealthRow("PASS", "sharp", "all",
                             f"sharp h2h matched {sharp_matched}/{len(rows)}"))

    # -- per-match checks ---------------------------------------------
    for r in rows:
        scope = f"{r.home} v {r.away}"

        if r.ou_source == "model" or r.ou_over is None:
            out.append(HealthRow("WARN", "degraded-input", scope,
                                 "O/U missing -> ou_source=model (1X2-only reconstruction)"))
        if r.sharp is None:
            out.append(HealthRow("WARN", "degraded-input", scope, "sharp 1X2 unmatched"))

        m = match_by_idx.get(r.index)
        if m is not None and not (_OVERROUND_LO <= m.overround <= _OVERROUND_HI):
            out.append(HealthRow("WARN", "degraded-input", scope,
                                 f"raw kicktipp overround {m.overround:.4f} outside "
                                 f"[{_OVERROUND_LO}, {_OVERROUND_HI}]"))

        _, _, fit_err = reconstruct_lambdas(*r.blended, r.ou_over)
        if fit_err > _FIT_ERR_WARN:
            out.append(HealthRow("WARN", "degraded-input", scope,
                                 f"high reconstruct fit_err {fit_err:.4f}"))

        # Relative-EV regression sentinel: a chosen pick at/above 0.95 P(rank<=target)
        # is non-discriminating. At target=3 that is the saturation symptom we fixed
        # (flag so it can't creep back); at target=1 a ≥95% P(win) is instead a genuine
        # near-lock - still worth a heads-up, but not a regression.
        rel_p = getattr(r, "rel_p_top", None)
        if rel_p is not None and rel_p >= 0.95:
            tgt = getattr(r, "rel_target", None)
            if tgt == 1:
                detail = (f"P(win)={rel_p:.0%} ≥95% - near-locked first place; "
                          "decision layer non-discriminating (trust EV-max)")
            else:
                detail = (f"P(top{tgt or 3})={rel_p:.0%} ≥95% - treat as non-discriminating "
                          "(possible saturation regression)")
            out.append(HealthRow("WARN", "relative-EV", scope, detail))

        # UNCHANGED-FROM-LAST
        if prior_existed and r.index in prior_rows:
            kt_s, sharp_s, ou_s, ev_s = _market_inputs_identical(r, prior_rows[r.index])
            if kt_s and sharp_s and ou_s:
                tail = " (EV-max pick also unchanged)" if ev_s else ""
                out.append(HealthRow("WARN", "unchanged-from-last", scope,
                                     "kicktipp+sharp+O/U byte-identical to last export"
                                     f"{tail} - possibly stale (cache / force_refresh) "
                                     "OR genuinely unmoved market"))

    # -- NOT-UPDATED (collect step, run level) ------------------------
    resolved = [f for f in fixtures if getattr(f, "result", None) is not None]
    picked_idx, snap_result_idx = set(), set()
    if picks_df is not None and len(picks_df):
        sub = picks_df[picks_df["spieltag"] == spieltag] if "spieltag" in picks_df else picks_df
        picked_idx = set(int(x) for x in sub.get("match_index", []))
    if snap_df is not None and len(snap_df) and "result" in snap_df:
        s = snap_df[(snap_df["spieltag"] == spieltag) & (snap_df["result"].notna())]
        snap_result_idx = set(int(x) for x in s.get("match_index", []))

    missing_picks = [f.index for f in resolved if f.index not in picked_idx]
    missing_snap = [f.index for f in resolved if f.index not in snap_result_idx]
    if missing_picks:
        out.append(HealthRow("WARN", "not-updated", "picks.csv",
                             f"{len(missing_picks)} resolved match(es) with NO logged picks "
                             f"(idx {missing_picks}); picks added this run: {picks_added}"))
    elif resolved:
        out.append(HealthRow("PASS", "picks.csv", "all",
                             f"{len(resolved)} resolved match(es) covered "
                             f"(+{picks_added} new picks this run)"))
    if missing_snap:
        out.append(HealthRow("WARN", "not-updated", "snapshots.csv",
                             f"{len(missing_snap)} resolved match(es) with NO backfilled result "
                             f"(idx {missing_snap}); backfilled this run: {results_backfilled}"))
    elif resolved:
        out.append(HealthRow("PASS", "snapshots.csv", "all",
                             f"results present for {len(resolved)} resolved match(es)"))

    return out


def format_health(report: list[HealthRow], spieltag: int, prior: Path | None) -> str:
    ref = prior.name if prior else "no prior export"
    n_warn = sum(1 for r in report if r.status == "WARN")
    n_pass = sum(1 for r in report if r.status == "PASS")
    lines = [f"# Health / staleness - md{spieltag} (vs {ref})", "",
             f"{'STATUS':<7}{'CATEGORY':<22}{'SCOPE':<26}DETAIL",
             "-" * 100]
    for r in report:
        lines.append(f"{r.status:<7}{r.category:<22}{r.scope[:25]:<26}{r.detail}")
    lines.append("-" * 100)
    lines.append(f"SUMMARY: {n_pass} pass / {n_warn} warn"
                 + ("  <- review WARNs by hand before entering picks" if n_warn else ""))
    return "\n".join(lines)


# -- orchestration -----------------------------------------------------


def run(spieltag: int, *, now: datetime | None = None,
        target: int = 1) -> tuple[list, list[HealthRow]]:
    """Run the full on-demand pipeline for one matchday. Returns (rows, report).

    ``target`` is the rank the relative-EV DEVIATE gate optimises for: ``1`` =
    P(finish 1st) - the default, the live objective once top-3 is secured - or
    ``3`` = the legacy P(rank<=3) band. Only the decision card's gate boundary
    changes; the EV-max pick is always the action default.
    """
    now = now or _now()

    # -- 1. COLLECT (a posteriori) ------------------------------------
    from src import oracle
    from src.data.leaderboard import scrape_leaderboard
    from src.opponents import load_picks, update_log

    picks_added = 0
    try:
        picks_added = update_log([spieltag])
    except Exception as e:
        logger.warning("COLLECT: update_log failed (non-fatal): %s", e)

    fixtures, current_totals, results_backfilled = [], {}, 0
    try:
        lb = scrape_leaderboard(spieltag=spieltag)
        fixtures = lb.fixtures
        current_totals = {p.name: p.total for p in lb.players}
        before = len(snapshot.load_history())
        snapshot.backfill_results(spieltag, fixtures)
        results_backfilled = len(snapshot.load_history()) - before
    except Exception as e:
        logger.warning("COLLECT: leaderboard/backfill failed (non-fatal): %s", e)

    # -- 2. REFRESH ODDS ----------------------------------------------
    try:
        matches = _scrape_with_auth_check(spieltag)
    except ReAuthNeeded as e:
        logger.error("RE-AUTH NEEDED: %s", e)
        report = [HealthRow("WARN", "missing/empty", "kicktipp",
                            "login page returned - KICKTIPP_SESSION expired")]
        print(format_health(report, spieltag, _find_prior_export(spieltag)))
        return [], report

    n_scraped = len(matches)
    sharp_records, totals_records, commence = _extract_events(force_refresh=True)

    # Predict only matches the leaderboard shows as NOT STARTED. A non-None
    # Fixture.result = kicked off (live OR finished) -> excluded, never
    # re-predicted; falls back to the commence rule when a leaderboard row is
    # missing (empty/failed scrape -> prior behaviour).
    fixture_result = {_key(f.home, f.away): getattr(f, "result", None) for f in fixtures}
    sel = [(i, m) for i, m in enumerate(matches)
           if _is_upcoming(commence.get(_key(m.home_team, m.away_team)),
                           fixture_result.get(_key(m.home_team, m.away_team)), now)]

    prior_path = _find_prior_export(spieltag)          # BEFORE writing this run's export
    prior_rows = _load_prior_rows(prior_path)

    if not sel:
        logger.info("No upcoming (not-yet-kicked-off) matches for md%d.", spieltag)
        report = health_report(
            spieltag, [], [], n_scraped=n_scraped, sharp_matched=0, fixtures=fixtures,
            picks_df=load_picks(), snap_df=snapshot.load_history(), prior_rows=prior_rows,
            prior_existed=bool(prior_rows), picks_added=picks_added,
            results_backfilled=results_backfilled)
        print(format_health(report, spieltag, prior_path))
        return [], report

    # -- 3. RECOMPUTE -------------------------------------------------
    # Knockout matchdays are scored a.PSO (result incl. ET + penalties, no draws):
    # a different score path, and the rank/decision layer is knockout-blind.
    is_knockout = any(getattr(m, "a_pso", False) for _, m in sel)
    blend_weight = _config_blend_weight()
    if is_knockout:
        rows = _build_card_knockout(sel, sharp_records, totals_records, commence)
        logger.info("Knockout (a.PSO) EV-max card built for %d match(es).", len(rows))
    else:
        rows = _build_card(sel, sharp_records, totals_records, commence, blend_weight)
    sharp_matched = sum(1 for r in rows if r.sharp is not None)

    try:
        from src.field_model import FieldModel
        from src.rank_sim import remaining_match_count
        horizon = max(0, remaining_match_count().get("remaining", 0) - len(rows))
        if oracle.SELF in current_totals:
            # Knockout mode reuses the same optimiser with a draw-free (a.PSO) matrix
            # and opponents' picks projected onto decisive scorelines (no real
            # knockout picks exist yet - P(win) is indicative until they land).
            _attach_relative_ev(rows, spieltag, current_totals, oracle.SELF, horizon,
                                FieldModel.from_disk(), target=target, knockout=is_knockout)
            logger.info("Relative-EV card attached (horizon=%d, target=%d, knockout=%s).",
                        horizon, target, is_knockout)
    except Exception as e:
        logger.warning("RECOMPUTE: relative-EV card skipped (non-fatal): %s", e)

    # -- 4. SNAPSHOT ODDS (capture-on-run, labelled) ------------------
    for r in rows:
        snapshot.record_match(
            spieltag, r.index, r.home, r.away,
            kicktipp_1x2=r.kt, sharp_1x2=r.sharp, ou_over_2_5=r.ou_over,
            lead_minutes_to_kickoff=_lead_minutes(r.kickoff, now), updated_at=now.isoformat())

    # -- 5. EXPORT + 6. PRINT -----------------------------------------
    csv_path, md_path = _write_exports(spieltag, rows, now)
    print(_format_card_md(spieltag, rows, now))
    logger.info("Exported %s + %s (blend_weight=%.2f).", csv_path.name, md_path.name, blend_weight)

    # -- 7. HEALTH / STALENESS ----------------------------------------
    report = health_report(
        spieltag, rows, sel, n_scraped=n_scraped, sharp_matched=sharp_matched,
        fixtures=fixtures, picks_df=load_picks(), snap_df=snapshot.load_history(),
        prior_rows=prior_rows, prior_existed=bool(prior_rows),
        picks_added=picks_added, results_backfilled=results_backfilled)
    print()
    print(format_health(report, spieltag, prior_path))
    return rows, report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="On-demand refresh-everything + health-check (no window, no loop, no cron).")
    p.add_argument("--spieltag", type=int, required=True)
    p.add_argument("--target", type=int, choices=(1, 3), default=1,
                   help="rank the DEVIATE gate optimises for: 1=P(win) [default], 3=P(rank<=3)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    run(args.spieltag, target=args.target)


if __name__ == "__main__":
    main()
