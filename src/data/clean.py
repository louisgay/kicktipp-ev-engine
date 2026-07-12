"""Data cleaning and normalisation.

Responsibilities
-----------------
- Parse dates, ensure correct dtypes.
- Normalise team names across sources (martj42, Elo, FIFA, WC 2026 roster).
- Classify matches: friendly / qualifier / finals.
- Mark venue type: home / away / neutral.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "config" / "config.yaml"

# -- Team-name mapping -------------------------------------------------
# Keys = variant spellings found in various sources.
# Values = canonical name used internally.
# Extend as needed when adding new data sources.

TEAM_NAME_MAP: dict[str, str] = {
    # Common discrepancies between martj42, Elo, FIFA
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Eswatini": "Swaziland",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Timor-Leste": "East Timor",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",  # Odds API format
    "Bosnien-Herzegowina": "Bosnia and Herzegovina",  # kicktipp German name
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Curacao": "Curaçao",  # Odds API uses unaccented
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Brunei Darussalam": "Brunei",
    "Kyrgyz Republic": "Kyrgyzstan",
}

# WC 2026 qualified teams (48 teams, as of 2025 knowledge).
# Used for validation / filtering. Will be updated.
WC_2026_QUALIFIED: set[str] = {
    "United States", "Mexico", "Canada",  # hosts
    "Brazil", "Argentina", "Uruguay", "Ecuador", "Colombia", "Paraguay",
    "Venezuela", "Peru", "Chile", "Bolivia",
    "Germany", "France", "Spain", "England", "Portugal", "Netherlands",
    "Belgium", "Italy", "Croatia", "Denmark", "Switzerland", "Austria",
    "Serbia", "Ukraine", "Scotland", "Slovenia", "Hungary", "Turkey",
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran", "Iraq",
    "Qatar", "Uzbekistan", "Jordan", "Indonesia", "China",
    "Morocco", "Senegal", "Nigeria", "Cameroon", "Egypt",
    "Ivory Coast", "South Africa", "DR Congo", "Mali",
    "New Zealand",
}


# -- Match-type classification -----------------------------------------

def classify_tournament(tournament: str) -> str:
    """Map tournament name to a broad category."""
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return "world_cup_finals"
    if "world cup" in t:
        return "world_cup_qualif"
    for comp in ("euro ", "european championship",
                 "copa america", "copa américa",
                 "african cup", "africa cup",
                 "asian cup", "gold cup",
                 "nations league", "confederations cup"):
        if comp in t:
            return "continental"
    if "qualif" in t:
        return "continental_qualif"
    return "friendly"


# -- Core cleaning -----------------------------------------------------

def normalise_team(name: str) -> str:
    """Return canonical team name."""
    return TEAM_NAME_MAP.get(name, name)


def clean_results(raw_path: Path | str) -> pd.DataFrame:
    """Load and clean the martj42 results.csv.

    Returns a DataFrame with columns:
        date, home_team, away_team, home_score, away_score,
        tournament, tournament_type, city, country, neutral
    """
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df["home_team"] = df["home_team"].map(normalise_team)
    df["away_team"] = df["away_team"].map(normalise_team)

    # Drop rows with missing scores
    n_before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    if len(df) < n_before:
        logger.info("Dropped %d rows with missing scores", n_before - len(df))

    # Classify tournament
    df["tournament_type"] = df["tournament"].map(classify_tournament)

    # Ensure score columns are int
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # neutral field: already present in martj42 data
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].fillna(False).astype(bool)

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Cleaned results: %d matches, %d teams, %s to %s",
                len(df), df["home_team"].nunique(),
                df["date"].min().date(), df["date"].max().date())
    return df


def save_processed(df: pd.DataFrame, dest: Path | str) -> Path:
    """Save cleaned DataFrame to processed directory."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Saved processed data: %s (%d rows)", dest, len(df))
    return dest


def clean_and_save(raw_dir: Path | None = None,
                   processed_dir: Path | None = None) -> pd.DataFrame:
    """Full cleaning pipeline: load raw -> clean -> save processed."""
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if raw_dir is None:
        raw_dir = _ROOT / cfg["paths"]["raw_data"]
    if processed_dir is None:
        processed_dir = _ROOT / cfg["paths"]["processed_data"]

    df = clean_results(raw_dir / "results.csv")
    save_processed(df, processed_dir / "results_clean.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_and_save()
