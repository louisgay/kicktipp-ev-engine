"""Standings-evolution chart for the pool.

Builds a self-contained interactive HTML showing, per matchday (MD1->current),
each player's standing (rank) and cumulative points since the start.

Two point bases (toggle in the page):
  - ``official``  - real Kicktipp leaderboard total (incl. group-stage bonus
                    questions), reconstructed per matchday from the banked
                    leaderboard HTML snapshots. Ranks come straight from the
                    site (authoritative tie-breaking).
  - ``tipping``   - pure per-match pick points (no bonus), summed from
                    ``data/opponents/picks.csv``. A "tipping-skill" view.

The latest matchday may be live (not all matches resolved); it is drawn
dashed and annotated as provisional.

This is isolated analysis/visualisation code. Dependency arrow is one-way:
``analysis -> src`` only. Run with::

    python -m analysis.standings.generate

Outputs ``analysis/standings/standings.html`` and ``standings_data.json``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from src.opponents import load_picks, parse_leaderboard

_HERE = Path(__file__).resolve().parent
_SNAP_DIR = Path("data/opponents/snapshots")
SELF = "self"


def _latest_html(md: int) -> Path | None:
    files = sorted(glob.glob(str(_SNAP_DIR / f"leaderboard_md{md}_*.html")))
    return Path(files[-1]) if files else None


def _official_by_matchday() -> tuple[dict[int, dict[str, float]], dict[int, dict[str, int]], list[int]]:
    """Per-matchday official cumulative totals and ranks from banked HTML.

    Returns (totals[md][player], ranks[md][player], matchdays_present).
    """
    totals: dict[int, dict[str, float]] = {}
    ranks: dict[int, dict[str, int]] = {}
    mds: list[int] = []
    for md in range(1, 30):
        path = _latest_html(md)
        if path is None:
            continue
        lb = parse_leaderboard(path.read_text(encoding="utf-8"), spieltag=md)
        totals[md] = {p.name: float(p.total) for p in lb.players if p.total is not None}
        ranks[md] = {p.name: int(p.rank) for p in lb.players if p.total is not None}
        mds.append(md)
    return totals, ranks, mds


def _tipping_by_matchday(mds: list[int]) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, int]], list[str]]:
    """Per-matchday cumulative tipping-only points + derived ranks from picks.csv."""
    df = load_picks()
    players = sorted(df["player"].unique())
    totals: dict[int, dict[str, float]] = {md: {} for md in mds}
    ranks: dict[int, dict[str, int]] = {md: {} for md in mds}
    # cumulative sum, but only count a player from the first matchday they appear
    first_md = {p: int(df[df.player == p]["spieltag"].min()) for p in players}
    for p in players:
        cum = 0.0
        for md in mds:
            sub = df[(df.player == p) & (df.spieltag == md)]
            cum += float(sub["points"].sum())
            if md >= first_md[p]:
                totals[md][p] = cum
    # competition ranking (min-rank for ties) per matchday
    for md in mds:
        ordered = sorted(totals[md].items(), key=lambda kv: -kv[1])
        prev_val = None
        prev_rank = 0
        for i, (name, val) in enumerate(ordered, start=1):
            if val != prev_val:
                prev_rank = i
                prev_val = val
            ranks[md][name] = prev_rank
    return totals, ranks, players


def build_data() -> dict:
    off_tot, off_rank, off_mds = _official_by_matchday()
    df = load_picks()
    # The official leaderboard view is reconstructed from banked leaderboard HTML
    # snapshots, which are NOT shipped (they embed internal member IDs). When they
    # are absent, fall back to the picks-derived matchdays so the tipping-skill
    # view still renders.
    pick_mds = sorted(int(x) for x in df["spieltag"].dropna().unique())
    mds = off_mds or pick_mds
    if not off_mds:
        print("note: no leaderboard snapshots found - rendering the tipping-only view "
              "(the official standings need the unshipped data/opponents/snapshots/).")
    tip_tot, tip_rank, players = _tipping_by_matchday(mds)

    # which matchday is "live" (in progress): the last one with any unresolved
    # match in picks.csv (a pick whose match has no result yet). We detect it as
    # the last matchday where official total differs hint isn't available, so we
    # flag the max matchday as live unless every player's matchday is complete.
    # The final matchday is provisional whenever its pick count is below the
    # modal full-matchday count (entries/results still trickling in).
    last_md = max(mds)
    counts = df.groupby("spieltag").size()
    full = int(counts[counts.index < last_md].max()) if (counts.index < last_md).any() else int(counts.max())
    live_md = last_md if int(counts.get(last_md, 0)) < full else None

    def series(tot, rnk):
        out = {}
        for p in players:
            pts, rks = [], []
            for md in mds:
                pts.append(tot.get(md, {}).get(p))
                rks.append(rnk_get(rnk, md, p))
            out[p] = {"points": pts, "ranks": rks}
        return out

    def rnk_get(rnk, md, p):
        v = rnk.get(md, {}).get(p)
        return int(v) if v is not None else None

    return {
        "matchdays": mds,
        "players": players,
        "self": SELF,
        "live_matchday": live_md,
        "n_players": len(players),
        "official": series(off_tot, off_rank),
        "tipping": series(tip_tot, tip_rank),
    }


# -- distinct, colour-blind-friendlier categorical palette (11 + self) ----------
_PALETTE = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#9A6324", "#dcbeff",
]


def render_html(data: dict) -> str:
    # assign colours; self gets a fixed bright gold and is drawn on top
    players = data["players"]
    colours = {}
    pi = 0
    for p in players:
        if p == data["self"]:
            colours[p] = "#f5c451"
        else:
            colours[p] = _PALETTE[pi % len(_PALETTE)]
            pi += 1
    data = {**data, "colours": colours}
    payload = json.dumps(data, ensure_ascii=False)
    return _TEMPLATE.replace("/*__DATA__*/", payload)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pool - Standings Evolution</title>
<style>
  :root{
    --bg:#0d1117; --panel:#131a24; --panel2:#0f151d; --ink:#c9d1d9; --muted:#7d8794;
    --grid:#1f2a37; --line:#223; --chip:#1b2430; --chipon:#243447; --self:#f5c451;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:13px/1.5 ui-monospace,"SF Mono",Menlo,Consolas,monospace;-webkit-font-smoothing:antialiased}
  header{padding:18px 22px 6px}
  h1{font-size:18px;margin:0;font-weight:600;letter-spacing:.3px}
  .sub{color:var(--muted);font-size:12px;margin-top:3px;max-width:1000px}
  .controls{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:center;
    padding:10px 22px;background:var(--panel2);border-top:1px solid var(--line);
    border-bottom:1px solid var(--line)}
  .seg{display:inline-flex;border:1px solid var(--grid);border-radius:7px;overflow:hidden}
  .seg button{background:transparent;color:var(--muted);border:0;padding:6px 13px;
    font:inherit;cursor:pointer}
  .seg button.on{background:var(--chipon);color:var(--ink)}
  label.ck{color:var(--muted);cursor:pointer;user-select:none}
  .wrap{padding:8px 14px 30px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    margin:14px 8px;padding:8px 10px 4px}
  .ptitle{font-size:12px;color:var(--muted);padding:4px 8px 0;letter-spacing:.4px;
    text-transform:uppercase}
  svg{display:block;width:100%;height:auto}
  .legend{display:flex;flex-wrap:wrap;gap:6px 8px;padding:8px 12px 14px}
  .lg{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:20px;
    background:var(--chip);border:1px solid var(--grid);cursor:pointer;font-size:12px}
  .lg .sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto}
  .lg.off{opacity:.34}
  .lg.self{border-color:var(--self)}
  .tip{position:fixed;pointer-events:none;background:#0b1018;border:1px solid var(--grid);
    border-radius:7px;padding:7px 10px;font-size:12px;color:var(--ink);z-index:9;
    box-shadow:0 6px 20px #0008;opacity:0;transition:opacity .08s}
  .tip b{color:#fff}
  line.grid{stroke:var(--grid);stroke-width:1}
  text{fill:var(--muted);font:11px ui-monospace,monospace}
  path.series{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  .foot{color:var(--muted);font-size:11px;padding:2px 22px 24px;max-width:1000px}
</style>
</head>
<body>
<header>
  <h1>pool - Standings Evolution</h1>
  <div class="sub" id="sub"></div>
</header>
<div class="controls">
  <span style="color:var(--muted)">basis</span>
  <span class="seg" id="basis">
    <button data-b="official" class="on">Official (incl. bonus)</button>
    <button data-b="tipping">Tipping only</button>
  </span>
  <label class="ck"><input type="checkbox" id="ghost"> show other basis as ghost</label>
  <label class="ck"><input type="checkbox" id="soloself"> isolate self</label>
</div>
<div class="wrap">
  <div class="panel">
    <div class="ptitle" id="t-rank">Standing (rank · #1 on top)</div>
    <svg id="rankSvg"></svg>
  </div>
  <div class="panel">
    <div class="ptitle" id="t-pts">Cumulative points</div>
    <svg id="ptsSvg"></svg>
  </div>
  <div class="legend" id="legend"></div>
  <div class="foot" id="foot"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const DATA = /*__DATA__*/;
const NS="http://www.w3.org/2000/svg";
const MDS=DATA.matchdays, PL=DATA.players, COL=DATA.colours, SELF=DATA.self;
const LIVE=DATA.live_matchday;
let basis="official", ghost=false, solo=false;
const hidden=new Set();

document.getElementById("sub").textContent =
  `${PL.length} players · MD${MDS[0]}-MD${MDS[MDS.length-1]}`
  + (LIVE!=null ? ` · MD${LIVE} is live (provisional, dashed)` : "");

function el(tag,attrs){const e=document.createElementNS(NS,tag);
  for(const k in attrs)e.setAttribute(k,attrs[k]);return e;}
function activePlayers(){
  if(solo) return PL.filter(p=>p===SELF);
  return PL.filter(p=>!hidden.has(p));
}

function draw(svg, kind){
  // kind: "ranks" (inverted, 1..n) or "points"
  svg.innerHTML="";
  const W=1000, H=kind==="ranks"?330:360, mL=46, mR=120, mT=18, mB=28;
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const iw=W-mL-mR, ih=H-mT-mB;
  const xs=md=>mL+ (MDS.length===1?iw/2: iw*(MDS.indexOf(md)/(MDS.length-1)));
  const act=activePlayers();

  let yMin,yMax,yOf;
  if(kind==="ranks"){
    yMin=1; yMax=DATA.n_players;
    yOf=v=> mT + ih*((v-yMin)/(yMax-yMin)); // 1 at top
  }else{
    let mx=0;
    for(const p of act) for(const b of (ghost?["official","tipping"]:[basis]))
      for(const v of DATA[b][p].points) if(v!=null&&v>mx)mx=v;
    yMin=0; yMax=Math.max(10, Math.ceil(mx/10)*10);
    yOf=v=> mT + ih*(1-(v-yMin)/(yMax-yMin));
  }

  // gridlines + y labels
  if(kind==="ranks"){
    for(let r=1;r<=DATA.n_players;r++){const y=yOf(r);
      svg.appendChild(el("line",{class:"grid",x1:mL,x2:W-mR,y1:y,y2:y,opacity:.5}));
      const t=el("text",{x:mL-8,y:y+3,"text-anchor":"end"});t.textContent=r;svg.appendChild(t);}
  }else{
    const step=yMax<=40?10:20;
    for(let v=0;v<=yMax;v+=step){const y=yOf(v);
      svg.appendChild(el("line",{class:"grid",x1:mL,x2:W-mR,y1:y,y2:y}));
      const t=el("text",{x:mL-8,y:y+3,"text-anchor":"end"});t.textContent=v;svg.appendChild(t);}
  }
  // x labels
  for(const md of MDS){const x=xs(md);
    const t=el("text",{x:x,y:H-8,"text-anchor":"middle"});
    t.textContent="MD"+md+(md===LIVE?" -":"");svg.appendChild(t);}

  function path(p, b, dashGhost){
    const arr=DATA[b][p][kind];
    const isSelf=p===SELF, col=COL[p];
    const w = dashGhost?1.4 : (isSelf?3.4:2);
    const o = dashGhost?0.32 : (isSelf?1:0.92);
    // collect present nodes
    const pts=[];
    MDS.forEach((md,i)=>{const v=arr[i];
      if(v!=null) pts.push([xs(md),yOf(v),md,v]);});
    // build solid path for every leg whose right endpoint is NOT the live md,
    // and a separate dashed path for the live leg (so it reads as provisional)
    let solid="", live="";
    for(let k=0;k<pts.length;k++){
      const [x,y,md]=pts[k];
      if(k===0){solid+=`M${x} ${y}`;continue;}
      if(!dashGhost && md===LIVE){
        const [px,py]=pts[k-1];
        live+=`M${px} ${py}L${x} ${y}`;
      }else{
        solid+=`L${x} ${y}`;
      }
    }
    const pe=el("path",{class:"series",d:solid,stroke:col,"stroke-width":w,opacity:o});
    if(dashGhost) pe.setAttribute("stroke-dasharray","2 3");
    svg.appendChild(pe);
    if(live){const lv=el("path",{class:"series",d:live,stroke:col,
      "stroke-width":w,opacity:o});lv.setAttribute("stroke-dasharray","5 4");
      svg.appendChild(lv);}
    return pts;
  }

  // ghost (other basis) first, behind
  if(ghost){const other=basis==="official"?"tipping":"official";
    for(const p of act) path(p,other,true);}
  // main series + dots + end labels
  for(const p of act){
    const pts=path(p,basis,false);
    const isSelf=p===SELF;
    for(const [x,y,md,v] of pts){
      const dot=el("circle",{cx:x,cy:y,r:isSelf?4:3,fill:COL[p],
        stroke:"#0d1117","stroke-width":1});
      dot.addEventListener("mousemove",ev=>showTip(ev,p,md));
      dot.addEventListener("mouseleave",hideTip);
      svg.appendChild(dot);}
    if(pts.length){const [x,y]=pts[pts.length-1];
      const t=el("text",{x:x+8,y:y+3,fill:COL[p]});
      t.setAttribute("font-weight",isSelf?"700":"400");
      t.textContent=p;svg.appendChild(t);}
  }
}

const tip=document.getElementById("tip");
function showTip(ev,p,md){
  const i=MDS.indexOf(md);
  const off=DATA.official[p], tp=DATA.tipping[p];
  const r=off.ranks[i], pts=off.points[i], tpts=tp.points[i];
  tip.innerHTML=`<b>${p}</b>${p===SELF?" *":""} · MD${md}${md===LIVE?" (live)":""}<br>`
    +`rank <b>#${r??"-"}</b> · official <b>${pts??"-"}</b> pts`
    +(tpts!=null?` · tipping ${tpts}`:"");
  tip.style.opacity=1;
  tip.style.left=(ev.clientX+14)+"px";
  tip.style.top=(ev.clientY+14)+"px";
}
function hideTip(){tip.style.opacity=0;}

function legend(){
  const box=document.getElementById("legend");box.innerHTML="";
  for(const p of PL){
    const d=document.createElement("div");
    d.className="lg"+(hidden.has(p)?" off":"")+(p===SELF?" self":"");
    d.innerHTML=`<span class="sw" style="background:${COL[p]}"></span>${p}`;
    d.onclick=()=>{if(hidden.has(p))hidden.delete(p);else hidden.add(p);renderAll();};
    box.appendChild(d);
  }
}
function renderAll(){
  draw(document.getElementById("rankSvg"),"ranks");
  draw(document.getElementById("ptsSvg"),"points");
  legend();
}

document.querySelectorAll("#basis button").forEach(b=>b.onclick=()=>{
  basis=b.dataset.b;
  document.querySelectorAll("#basis button").forEach(x=>x.classList.toggle("on",x===b));
  renderAll();
});
document.getElementById("ghost").onchange=e=>{ghost=e.target.checked;renderAll();};
document.getElementById("soloself").onchange=e=>{solo=e.target.checked;renderAll();};
document.getElementById("foot").textContent =
  "Official = real Kicktipp leaderboard total incl. group-stage bonus, reconstructed per "
  +"matchday from banked snapshots; ranks straight from the site. Tipping = per-match pick "
  +"points only (no bonus). Hover a node for detail; click a legend chip to toggle a player.";
renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    (_HERE / "standings_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html = render_html(data)
    out = _HERE / "standings.html"
    out.write_text(html, encoding="utf-8")
    live = data["live_matchday"]
    print(f"Wrote {out}")
    print(f"  matchdays: MD{data['matchdays'][0]}-MD{data['matchdays'][-1]}"
          + (f"  (MD{live} live/provisional)" if live else ""))
    print(f"  players:   {len(data['players'])}")


if __name__ == "__main__":
    main()
