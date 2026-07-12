# analysis/standings - pool standings evolution

Isolated visualisation (not the decision pipeline). Dependency arrow is one-way:
`analysis -> src` only - it reuses `src.opponents.parse_leaderboard` / `load_picks`.

```bash
python -m analysis.standings.generate    # -> standings.html + standings_data.json
open analysis/standings/standings.html
```

Two stacked panels, per matchday MD1->current:

- **Standing** - official rank over time (#1 on top, a bump chart).
- **Cumulative points** - running total over time.

Two point bases (toggle in-page):

- **Official (incl. bonus)** - the real Kicktipp leaderboard total, reconstructed
  per matchday from the banked `data/opponents/snapshots/leaderboard_md*.html`.
  Ranks come straight from the site (authoritative tie-breaking). Includes the
  group-stage bonus questions, so it matches what you see on Kicktipp.
- **Tipping only** - per-match pick points summed from `data/opponents/picks.csv`
  (no bonus). A pure tipping-skill view; diverges from official once bonus points
  are awarded (here: from MD4 on).

The latest matchday is drawn **dashed** when it is still live/provisional (not all
matches resolved / not all players entered). `self` (= SELF) is highlighted.
Hover any node for rank + points; click a legend chip to toggle a player; "isolate
self" and "show other basis as ghost" are optional overlays.

Generated `standings.html` / `standings_data.json` are rebuildable artifacts.
