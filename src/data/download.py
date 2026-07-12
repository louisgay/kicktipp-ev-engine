"""Idempotent download of raw data sources.

Sources
-------
1. martj42/international_results (GitHub)
   - results.csv (~49k international matches since 1872)
   - shootouts.csv (penalty shootout results)
   - goalscorers.csv (individual goals)
   License: public-domain-equivalent (CC0)

2. World Football Elo Ratings (eloratings.net)
   - National-team Elo ratings
   - Scraped from the website (no official API)
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG) as f:
        return yaml.safe_load(f)


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* with streaming."""
    logger.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    logger.info("Saved %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)


# -- martj42 international results --------------------------------------


def download_martj42(raw_dir: Path | None = None, force: bool = False) -> list[Path]:
    """Download international results CSVs from martj42/international_results.

    Returns list of downloaded file paths.
    """
    cfg = _load_config()
    base_url = cfg["sources"]["martj42"]["base_url"]
    files = cfg["sources"]["martj42"]["files"]
    if raw_dir is None:
        raw_dir = _ROOT / cfg["paths"]["raw_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for fname in files:
        dest = raw_dir / fname
        if dest.exists() and not force:
            logger.info("Already present: %s (skip)", dest)
        else:
            _download_file(f"{base_url}/{fname}", dest)
        paths.append(dest)
    return paths


# -- Elo ratings --------------------------------------------------------


def download_elo_ratings(raw_dir: Path | None = None, force: bool = False) -> Path:
    """Download / scrape World Football Elo ratings for national teams.

    Strategy: eloratings.net exposes a page per team and a global ranking.
    We scrape the global ranking page to get current Elo for all teams.
    For historical Elo we compute it ourselves from results.csv in the
    features module (more reliable than scraping historical pages).

    Returns path to the saved CSV.
    """
    cfg = _load_config()
    if raw_dir is None:
        raw_dir = _ROOT / cfg["paths"]["raw_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "elo_ratings.csv"
    if dest.exists() and not force:
        logger.info("Already present: %s (skip)", dest)
        return dest

    # Scrape the main ranking page
    from bs4 import BeautifulSoup

    url = "https://www.eloratings.net/"
    logger.info("Scraping Elo ratings from %s", url)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "kicktipp-research/0.1"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []
    # The site uses a table with class 'maintable' or similar
    table = soup.find("table")
    if table is None:
        logger.warning("Could not find rating table on eloratings.net; "
                       "will compute Elo from results instead.")
        # Write empty placeholder
        import pandas as pd
        pd.DataFrame(columns=["rank", "team", "elo"]).to_csv(dest, index=False)
        return dest

    for tr in table.find_all("tr")[1:]:  # skip header
        cells = tr.find_all("td")
        if len(cells) >= 3:
            rank = cells[0].get_text(strip=True)
            team = cells[1].get_text(strip=True)
            elo = cells[2].get_text(strip=True)
            if rank.isdigit():
                rows.append({"rank": int(rank), "team": team, "elo": int(elo)})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False)
    logger.info("Saved %d Elo ratings to %s", len(df), dest)
    return dest


# -- Compute historical Elo from results -------------------------------


def compute_elo_history(results_csv: Path, dest: Path | None = None) -> Path:
    """Compute historical Elo ratings from scratch using results.csv.

    This is more reliable than scraping historical pages and gives us
    Elo for any date we need. Uses the World Football Elo formula:
      - K-factor depends on match type and goal difference
      - Home advantage = +100 Elo points (adjustable)

    Returns path to the output CSV with columns:
      date, team, elo_before, elo_after
    """
    import pandas as pd
    import numpy as np

    if dest is None:
        dest = results_csv.parent.parent / "processed" / "elo_history.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Elo history already computed: %s", dest)
        return dest

    df = pd.read_csv(results_csv, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    # Initial Elo for all teams
    elo: dict[str, float] = {}
    INITIAL_ELO = 1500.0
    HOME_ADV = 100.0

    # K-factor by tournament type
    K_FACTORS = {
        "FIFA World Cup": 60,
        "FIFA World Cup qualification": 40,
        "Continental championship": 50,
        "Continental qualification": 40,
        "Friendly": 20,
    }

    def _classify_tournament(tournament: str) -> str:
        t = tournament.lower()
        if "world cup" in t and "qualif" not in t:
            return "FIFA World Cup"
        if "world cup" in t and "qualif" in t:
            return "FIFA World Cup qualification"
        if any(x in t for x in ["euro ", "copa america", "african cup",
                                 "asian cup", "gold cup", "nations league"]):
            return "Continental championship"
        if "qualif" in t:
            return "Continental qualification"
        return "Friendly"

    def _goal_diff_multiplier(goal_diff: int) -> float:
        """Multiplier for margin of victory (World Football Elo formula)."""
        if goal_diff <= 1:
            return 1.0
        if goal_diff == 2:
            return 1.5
        return (11.0 + goal_diff) / 8.0

    records = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        if home not in elo:
            elo[home] = INITIAL_ELO
        if away not in elo:
            elo[away] = INITIAL_ELO

        elo_h = elo[home] + HOME_ADV
        elo_a = elo[away]

        # Expected scores
        exp_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
        exp_a = 1.0 - exp_h

        # Actual result
        gh, ga = int(row["home_score"]), int(row["away_score"])
        if gh > ga:
            w_h, w_a = 1.0, 0.0
        elif gh < ga:
            w_h, w_a = 0.0, 1.0
        else:
            w_h, w_a = 0.5, 0.5

        k_cat = _classify_tournament(str(row.get("tournament", "Friendly")))
        k = K_FACTORS.get(k_cat, 20)
        gd_mult = _goal_diff_multiplier(abs(gh - ga))

        delta_h = k * gd_mult * (w_h - exp_h)

        elo_before_h = elo[home]
        elo_before_a = elo[away]
        elo[home] += delta_h
        elo[away] -= delta_h

        records.append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "elo_home_before": round(elo_before_h, 1),
            "elo_away_before": round(elo_before_a, 1),
            "elo_home_after": round(elo[home], 1),
            "elo_away_after": round(elo[away], 1),
        })

    elo_df = pd.DataFrame(records)
    elo_df.to_csv(dest, index=False)
    logger.info("Computed Elo history: %d match records -> %s", len(elo_df), dest)
    return dest


# -- Convenience --------------------------------------------------------


def download_all(force: bool = False) -> dict[str, list[Path] | Path]:
    """Download all data sources. Returns dict of paths."""
    results = {}
    results["martj42"] = download_martj42(force=force)
    results["elo"] = download_elo_ratings(force=force)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_all()
