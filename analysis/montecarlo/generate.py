"""Pre-compute the fan-chart data and emit a self-contained interactive HTML.

The Monte-Carlo is run HERE (in Python), never in the browser: we sweep the control
matrix (strategy × regime × horizon, plus a few z_chase stops for the rank-relative
slider), assemble a FanChart per combo, and write:

  - fanchart_data.json      - the raw payload (for inspection / re-use)
  - fanchart.html           - the standalone visualiser with the JSON injected inline
                              (opens with file://, no server, no in-browser compute)

Rival reference lines are the FIELD's order statistics (me excluded), so they are
strategy-independent and computed once per (regime, horizon) from a track='all' run;
every other combo uses a light track='me' run and reuses those rivals.

Run:  python -m analysis.montecarlo.generate            # default 40k sims
      python -m analysis.montecarlo.generate 80000      # more sims (tighter bands)
HTML lands at analysis/montecarlo/fanchart.html
"""

from __future__ import annotations

import json
from pathlib import Path

from src.field_model import FieldModel
from src.rank_sim import GROUP_STAGE_MATCHES, TOURNAMENT_MATCHES

from analysis.montecarlo.engine import SELF, SimConfig, simulate
from analysis.montecarlo.strategies import SIGMA_MATCH
from analysis.montecarlo.stats import (
    build_fanchart,
    current_leader,
    leader_regression,
    option_pricing_report,
    rival_reference_paths,
)

_HERE = Path(__file__).resolve().parent
_OUT_JSON = _HERE / "fanchart_data.json"
_OUT_HTML = _HERE / "fanchart.html"
_TEMPLATE = _HERE / "_fanchart_template.html"

REGIMES = ["counterfactual", "hybrid"]
HORIZONS = {"group": GROUP_STAGE_MATCHES, "full": TOURNAMENT_MATCHES}
Z_STOPS = [0.5, 1.0, 1.5, 2.0]          # rank-relative slider snap points
N_SPAGHETTI = 120


def _round_bands(bands: dict) -> dict:
    return {str(p): [round(v, 1) for v in arr] for p, arr in bands.items()}


def _fc_to_dict(fc) -> dict:
    return {
        "bands": _round_bands(fc.bands),
        "rank_dist": [round(x, 5) for x in fc.rank_dist],
        "p_top": round(fc.p_top, 5), "p_win": round(fc.p_win, 5),
        "rivals": {k: [round(v, 1) for v in arr] for k, arr in fc.rivals.items()},
        # cumulative points are integer-valued -> store spaghetti as ints (compact)
        "spaghetti": [[int(round(v)) for v in path] for path in fc.spaghetti],
        "spread": {k: round(v, 2) for k, v in fc.spread.items()},
        "horizon": fc.horizon,
    }


def generate_payload(n_sims: int = 40_000, seed: int = 0) -> dict:
    fm = FieldModel.from_disk()
    leader = current_leader()
    players_tracked = [SELF] + ([leader] if leader != SELF else [])
    # the player's actual real-world play style (for the red-path label clarifier)
    styles = {SELF: "EV-max", leader: "fade (contrarian)"}

    charts: dict[str, dict] = {}
    realised: dict[str, dict] = {}
    op_results: dict = {}
    # FIELD rivals: computed ONCE per (regime,horizon) from a track='all' EV-max run
    # (me=self excluded). Reused across tracked players - they are field-reference
    # lines and the one-of-12 exclusion difference is within Monte-Carlo noise.
    rivals_by_rh: dict[tuple, dict] = {}

    for regime in REGIMES:
        for hname, hz in HORIZONS.items():
            base = simulate(SimConfig(n_sims=n_sims, seed=seed, regime=regime,
                                      strategy="evmax", horizon=hz, track="all"),
                            field_model=fm)
            rivals = rival_reference_paths(base.cum, base.config.target, base.me_index)
            rivals_by_rh[(regime, hname)] = rivals

    for player in players_tracked:
        for regime in REGIMES:
            for hname, hz in HORIZONS.items():
                rivals = rivals_by_rh[(regime, hname)]

                def _run(strategy, zc=None):
                    cfg = SimConfig(n_sims=n_sims, seed=seed, regime=regime, strategy=strategy,
                                    horizon=hz, me_player=player,
                                    **({"z_chase": zc} if zc is not None else {}))
                    res = simulate(cfg, field_model=fm)
                    fc = build_fanchart(res, n_spaghetti=N_SPAGHETTI, rivals=rivals)
                    if player not in realised:
                        realised[player] = {"index": fc.realised_index, "cum": fc.realised_cum,
                                            "rank_so_far": fc.realised_rank_so_far,
                                            "style": styles.get(player, "real")}
                    return res, fc

                res_ev, fc_ev = _run("evmax")
                charts[f"{player}|evmax|{regime}|{hname}|na"] = _fc_to_dict(fc_ev)
                res_c, fc_c = _run("contrarian")
                charts[f"{player}|contrarian|{regime}|{hname}|na"] = _fc_to_dict(fc_c)
                for zc in Z_STOPS:
                    _, fc_r = _run("rank_relative", zc)
                    charts[f"{player}|rank_relative|{regime}|{hname}|{zc}"] = _fc_to_dict(fc_r)

                # option-pricing readout uses self, counterfactual, full horizon
                if player == SELF and regime == "counterfactual" and hname == "full":
                    op_results = {"evmax": res_ev, "contrarian": res_c,
                                  "rank_relative": _run("rank_relative", 1.0)[0]}

    op_text, _ = option_pricing_report(op_results)
    lr = leader_regression(leader=leader, n_sims=n_sims, seed=seed, field_model=fm)

    return {
        "meta": {
            "n_sims": n_sims, "seed": seed, "me_player": SELF, "leader": leader,
            "players_tracked": players_tracked,
            "players": base.players, "n_players": len(base.players), "target": 3,
            "sigma_match": round(SIGMA_MATCH, 4),
            "horizons": HORIZONS, "z_stops": Z_STOPS,
            "group_stage_matches": GROUP_STAGE_MATCHES,
            "option_pricing_text": op_text, "leader_regression": lr,
        },
        "realised": realised,
        "charts": charts,
    }


def write(n_sims: int = 40_000, seed: int = 0) -> None:
    payload = generate_payload(n_sims, seed)
    _OUT_JSON.write_text(json.dumps(payload))
    print(f"wrote {_OUT_JSON.relative_to(_HERE.parents[1])} "
          f"({_OUT_JSON.stat().st_size / 1024:.0f} KB, {len(payload['charts'])} charts)")
    if _TEMPLATE.exists():
        html = _TEMPLATE.read_text().replace(
            "/*__FANCHART_DATA__*/", json.dumps(payload))
        _OUT_HTML.write_text(html)
        print(f"wrote {_OUT_HTML.relative_to(_HERE.parents[1])} "
              f"({_OUT_HTML.stat().st_size / 1024:.0f} KB) - open it in a browser")
    else:
        print(f"NOTE: template {_TEMPLATE.name} missing; wrote JSON only")


if __name__ == "__main__":
    import sys
    write(int(sys.argv[1]) if len(sys.argv) > 1 else 40_000)
