# Data - provenance, licensing & how to regenerate

This repo ships only small, transformed datasets needed to run the tests and
notebooks offline. Everything else is regenerable or deliberately excluded.

## Shipped (committed)

| File | What it is | Notes |
|------|------------|-------|
| `opponents/picks.csv` | One row per (matchday, player, match): the player's pick, the result, and the points scored under the 4/3/2/0 rules. | **Pseudonymised.** Real display names are replaced by stable labels `p01`-`p11`; the engine's own player is `self`. No real identities ship. |
| `opponents/oracle.csv` | Pre-kickoff predicted scorelines from public German/French tip media, per match. | Site names kept (public); individual tipsters' first names replaced by `tipster_a...d`. |
| `history/snapshots.csv` | Per-match pre-kickoff market snapshot: Kicktipp 1X2, sharp 1X2, over/under 2.5, and the realised result. Probabilities are already de-vigged/aggregated. | Contains only teams and numbers - no personal data. The durable market record the models read. |
| `history/behavioral_edge_log.csv` | Appended t-stat log from `analysis/behavioral_edge.py`. | Per-player columns use the pseudonyms (`t_p01`, ...). |

## Not shipped (excluded on purpose)

- **Bookmaker odds cache** (`data/raw/odds_cache/*.json`). These are raw bookmaker
  prices retrieved from **The Odds API**, whose terms restrict redistribution, so
  they are **not** republished. The test suite does not depend on them; live odds
  can be re-fetched with your own `ODDS_API_KEY` (see `.env.example`). Nothing in
  the shipped tests needs a synthetic replacement - the offline path reads the
  aggregated `snapshots.csv` instead.
- **Leaderboard HTML snapshots** (`data/opponents/snapshots/`). Raw scraped pages
  embed internal member IDs, so they are excluded. The `analysis/standings` viz
  degrades gracefully without them (it renders the picks-based "tipping" view).
- **Raw/processed historical match data** (`data/raw/`, `data/processed/`).
  Regenerable - see below.

## Regenerating the historical match data

Some diagnostics (e.g. the calibration curated-join test, the exploration
notebook) use a cleaned table of historical international results. It is
gitignored and rebuilt from public sources:

```bash
python -m src.data.download   # fetches results + Elo ratings
python -m src.data.clean      # normalises into data/processed/results_clean.csv
```

Sources, with thanks:
- **martj42/international_results** - international match results
  (https://github.com/martj42/international_results).
- **World Football Elo Ratings** - national-team Elo (https://www.eloratings.net).

Tests that need this table `skip` cleanly when it is absent, so `pytest` is green
on a fresh clone.
