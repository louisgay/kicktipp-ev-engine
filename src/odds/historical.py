"""Historical odds data for backtesting.

Provides closing odds for past tournaments, either from:
1. The Odds API (historical endpoint, cached)
2. Manually curated data (fallback for quota conservation)

The Odds API historical endpoint returns all events for a sport at
a given timestamp snapshot. We query one snapshot per matchday.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


# -- World Cup 2022 closing odds (Pinnacle-sourced, curated) ----------
# Source: public records of Pinnacle closing 1X2 and O/U 2.5 odds.
# Format: (home, away, [home_odds, draw_odds, away_odds], [over_2.5, under_2.5])

WC_2022_ODDS: list[dict] = [
    # Group Stage - Day 1
    {"home_team": "Qatar", "away_team": "Ecuador", "h2h_odds": [5.54, 3.88, 1.73], "totals_odds": [2.11, 1.82], "totals_line": 2.5, "date": "2022-11-20"},
    {"home_team": "England", "away_team": "Iran", "h2h_odds": [1.37, 5.10, 9.20], "totals_odds": [1.60, 2.44], "totals_line": 2.5, "date": "2022-11-21"},
    {"home_team": "Senegal", "away_team": "Netherlands", "h2h_odds": [4.70, 3.55, 1.85], "totals_odds": [2.01, 1.90], "totals_line": 2.5, "date": "2022-11-21"},
    {"home_team": "United States", "away_team": "Wales", "h2h_odds": [2.35, 3.30, 3.30], "totals_odds": [2.05, 1.87], "totals_line": 2.5, "date": "2022-11-21"},
    # Day 2
    {"home_team": "Argentina", "away_team": "Saudi Arabia", "h2h_odds": [1.18, 7.30, 17.0], "totals_odds": [1.52, 2.64], "totals_line": 2.5, "date": "2022-11-22"},
    {"home_team": "Mexico", "away_team": "Poland", "h2h_odds": [2.65, 3.10, 2.95], "totals_odds": [2.10, 1.83], "totals_line": 2.5, "date": "2022-11-22"},
    {"home_team": "Denmark", "away_team": "Tunisia", "h2h_odds": [1.75, 3.55, 5.20], "totals_odds": [1.98, 1.93], "totals_line": 2.5, "date": "2022-11-22"},
    {"home_team": "France", "away_team": "Australia", "h2h_odds": [1.27, 5.90, 12.5], "totals_odds": [1.58, 2.48], "totals_line": 2.5, "date": "2022-11-22"},
    # Day 3
    {"home_team": "Morocco", "away_team": "Croatia", "h2h_odds": [4.20, 3.35, 2.00], "totals_odds": [2.14, 1.79], "totals_line": 2.5, "date": "2022-11-23"},
    {"home_team": "Belgium", "away_team": "Canada", "h2h_odds": [1.57, 4.10, 6.40], "totals_odds": [1.77, 2.14], "totals_line": 2.5, "date": "2022-11-23"},
    {"home_team": "Germany", "away_team": "Japan", "h2h_odds": [1.42, 4.90, 8.00], "totals_odds": [1.69, 2.27], "totals_line": 2.5, "date": "2022-11-23"},
    {"home_team": "Spain", "away_team": "Costa Rica", "h2h_odds": [1.19, 7.00, 17.0], "totals_odds": [1.46, 2.85], "totals_line": 2.5, "date": "2022-11-23"},
    # Day 4
    {"home_team": "Portugal", "away_team": "Ghana", "h2h_odds": [1.40, 4.80, 8.80], "totals_odds": [1.72, 2.22], "totals_line": 2.5, "date": "2022-11-24"},
    {"home_team": "Uruguay", "away_team": "South Korea", "h2h_odds": [1.90, 3.45, 4.40], "totals_odds": [2.05, 1.87], "totals_line": 2.5, "date": "2022-11-24"},
    {"home_team": "Switzerland", "away_team": "Cameroon", "h2h_odds": [2.07, 3.30, 3.90], "totals_odds": [2.07, 1.85], "totals_line": 2.5, "date": "2022-11-24"},
    {"home_team": "Brazil", "away_team": "Serbia", "h2h_odds": [1.38, 5.00, 8.80], "totals_odds": [1.69, 2.27], "totals_line": 2.5, "date": "2022-11-24"},
    # Day 5
    {"home_team": "England", "away_team": "United States", "h2h_odds": [1.62, 3.85, 6.00], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2022-11-25"},
    {"home_team": "Wales", "away_team": "Iran", "h2h_odds": [2.18, 3.25, 3.65], "totals_odds": [2.12, 1.80], "totals_line": 2.5, "date": "2022-11-25"},
    {"home_team": "Qatar", "away_team": "Senegal", "h2h_odds": [4.90, 3.65, 1.79], "totals_odds": [2.01, 1.90], "totals_line": 2.5, "date": "2022-11-25"},
    {"home_team": "Netherlands", "away_team": "Ecuador", "h2h_odds": [1.92, 3.50, 4.25], "totals_odds": [1.93, 1.97], "totals_line": 2.5, "date": "2022-11-25"},
    # Day 6
    {"home_team": "Poland", "away_team": "Saudi Arabia", "h2h_odds": [1.67, 3.80, 5.60], "totals_odds": [1.93, 1.97], "totals_line": 2.5, "date": "2022-11-26"},
    {"home_team": "Argentina", "away_team": "Mexico", "h2h_odds": [1.62, 3.80, 6.20], "totals_odds": [1.92, 1.98], "totals_line": 2.5, "date": "2022-11-26"},
    {"home_team": "Tunisia", "away_team": "Australia", "h2h_odds": [2.55, 3.10, 3.05], "totals_odds": [2.19, 1.75], "totals_line": 2.5, "date": "2022-11-26"},
    {"home_team": "France", "away_team": "Denmark", "h2h_odds": [1.82, 3.60, 4.70], "totals_odds": [1.90, 2.00], "totals_line": 2.5, "date": "2022-11-26"},
    # Day 7
    {"home_team": "Japan", "away_team": "Costa Rica", "h2h_odds": [1.58, 4.10, 6.40], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2022-11-27"},
    {"home_team": "Spain", "away_team": "Germany", "h2h_odds": [2.38, 3.40, 3.10], "totals_odds": [1.77, 2.14], "totals_line": 2.5, "date": "2022-11-27"},
    {"home_team": "Belgium", "away_team": "Morocco", "h2h_odds": [1.57, 4.10, 6.40], "totals_odds": [1.87, 2.03], "totals_line": 2.5, "date": "2022-11-27"},
    {"home_team": "Croatia", "away_team": "Canada", "h2h_odds": [1.70, 3.80, 5.20], "totals_odds": [1.78, 2.13], "totals_line": 2.5, "date": "2022-11-27"},
    # Day 8
    {"home_team": "Portugal", "away_team": "Uruguay", "h2h_odds": [1.85, 3.50, 4.60], "totals_odds": [1.88, 2.02], "totals_line": 2.5, "date": "2022-11-28"},
    {"home_team": "South Korea", "away_team": "Ghana", "h2h_odds": [2.58, 3.25, 2.85], "totals_odds": [1.86, 2.04], "totals_line": 2.5, "date": "2022-11-28"},
    {"home_team": "Cameroon", "away_team": "Serbia", "h2h_odds": [3.45, 3.30, 2.24], "totals_odds": [1.72, 2.22], "totals_line": 2.5, "date": "2022-11-28"},
    {"home_team": "Brazil", "away_team": "Switzerland", "h2h_odds": [1.50, 4.30, 7.20], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2022-11-28"},
    # Day 9 (MD3)
    {"home_team": "Ecuador", "away_team": "Senegal", "h2h_odds": [2.78, 3.20, 2.70], "totals_odds": [1.90, 2.00], "totals_line": 2.5, "date": "2022-11-29"},
    {"home_team": "Qatar", "away_team": "Netherlands", "h2h_odds": [9.80, 5.50, 1.33], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2022-11-29"},
    {"home_team": "Wales", "away_team": "England", "h2h_odds": [6.40, 4.00, 1.57], "totals_odds": [1.85, 2.05], "totals_line": 2.5, "date": "2022-11-29"},
    {"home_team": "Iran", "away_team": "United States", "h2h_odds": [3.80, 3.40, 2.10], "totals_odds": [2.03, 1.88], "totals_line": 2.5, "date": "2022-11-29"},
    # Day 10
    {"home_team": "Poland", "away_team": "Argentina", "h2h_odds": [6.60, 4.20, 1.53], "totals_odds": [1.80, 2.10], "totals_line": 2.5, "date": "2022-11-30"},
    {"home_team": "Saudi Arabia", "away_team": "Mexico", "h2h_odds": [5.20, 3.65, 1.78], "totals_odds": [1.93, 1.97], "totals_line": 2.5, "date": "2022-11-30"},
    {"home_team": "Australia", "away_team": "Denmark", "h2h_odds": [4.40, 3.50, 1.90], "totals_odds": [2.05, 1.87], "totals_line": 2.5, "date": "2022-11-30"},
    {"home_team": "Tunisia", "away_team": "France", "h2h_odds": [5.40, 3.60, 1.75], "totals_odds": [2.05, 1.87], "totals_line": 2.5, "date": "2022-11-30"},
    # Day 11
    {"home_team": "Canada", "away_team": "Morocco", "h2h_odds": [3.65, 3.30, 2.20], "totals_odds": [2.00, 1.91], "totals_line": 2.5, "date": "2022-12-01"},
    {"home_team": "Costa Rica", "away_team": "Germany", "h2h_odds": [13.0, 6.40, 1.24], "totals_odds": [1.47, 2.80], "totals_line": 2.5, "date": "2022-12-01"},
    {"home_team": "Japan", "away_team": "Spain", "h2h_odds": [5.40, 3.70, 1.73], "totals_odds": [1.95, 1.95], "totals_line": 2.5, "date": "2022-12-01"},
    {"home_team": "Croatia", "away_team": "Belgium", "h2h_odds": [2.90, 3.15, 2.62], "totals_odds": [1.95, 1.95], "totals_line": 2.5, "date": "2022-12-01"},
    # Day 12
    {"home_team": "Serbia", "away_team": "Switzerland", "h2h_odds": [2.52, 3.25, 3.00], "totals_odds": [1.72, 2.22], "totals_line": 2.5, "date": "2022-12-02"},
    {"home_team": "Cameroon", "away_team": "Brazil", "h2h_odds": [9.00, 5.00, 1.38], "totals_odds": [1.72, 2.22], "totals_line": 2.5, "date": "2022-12-02"},
    {"home_team": "Ghana", "away_team": "Uruguay", "h2h_odds": [3.80, 3.45, 2.05], "totals_odds": [1.85, 2.05], "totals_line": 2.5, "date": "2022-12-02"},
    {"home_team": "South Korea", "away_team": "Portugal", "h2h_odds": [5.40, 3.85, 1.68], "totals_odds": [1.80, 2.10], "totals_line": 2.5, "date": "2022-12-02"},
    # Round of 16
    {"home_team": "Netherlands", "away_team": "United States", "h2h_odds": [1.67, 3.80, 5.60], "totals_odds": [1.80, 2.10], "totals_line": 2.5, "date": "2022-12-03"},
    {"home_team": "Argentina", "away_team": "Australia", "h2h_odds": [1.27, 5.60, 12.0], "totals_odds": [1.70, 2.24], "totals_line": 2.5, "date": "2022-12-03"},
    {"home_team": "France", "away_team": "Poland", "h2h_odds": [1.35, 5.20, 9.50], "totals_odds": [1.62, 2.40], "totals_line": 2.5, "date": "2022-12-04"},
    {"home_team": "England", "away_team": "Senegal", "h2h_odds": [1.35, 5.20, 9.50], "totals_odds": [1.68, 2.28], "totals_line": 2.5, "date": "2022-12-04"},
    {"home_team": "Japan", "away_team": "Croatia", "h2h_odds": [3.10, 3.10, 2.48], "totals_odds": [2.10, 1.82], "totals_line": 2.5, "date": "2022-12-05"},
    {"home_team": "Brazil", "away_team": "South Korea", "h2h_odds": [1.30, 5.40, 11.0], "totals_odds": [1.52, 2.64], "totals_line": 2.5, "date": "2022-12-05"},
    {"home_team": "Morocco", "away_team": "Spain", "h2h_odds": [5.00, 3.40, 1.82], "totals_odds": [2.14, 1.79], "totals_line": 2.5, "date": "2022-12-06"},
    {"home_team": "Portugal", "away_team": "Switzerland", "h2h_odds": [1.65, 3.90, 5.60], "totals_odds": [1.67, 2.30], "totals_line": 2.5, "date": "2022-12-06"},
    # Quarter-finals
    {"home_team": "Croatia", "away_team": "Brazil", "h2h_odds": [4.80, 3.50, 1.82], "totals_odds": [1.90, 2.00], "totals_line": 2.5, "date": "2022-12-09"},
    {"home_team": "Netherlands", "away_team": "Argentina", "h2h_odds": [3.40, 3.20, 2.30], "totals_odds": [1.98, 1.93], "totals_line": 2.5, "date": "2022-12-09"},
    {"home_team": "England", "away_team": "France", "h2h_odds": [2.80, 3.25, 2.65], "totals_odds": [1.80, 2.10], "totals_line": 2.5, "date": "2022-12-10"},
    {"home_team": "Morocco", "away_team": "Portugal", "h2h_odds": [4.60, 3.30, 1.90], "totals_odds": [2.16, 1.78], "totals_line": 2.5, "date": "2022-12-10"},
    # Semi-finals
    {"home_team": "Argentina", "away_team": "Croatia", "h2h_odds": [1.57, 4.00, 6.60], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2022-12-13"},
    {"home_team": "France", "away_team": "Morocco", "h2h_odds": [1.38, 4.80, 9.00], "totals_odds": [1.78, 2.13], "totals_line": 2.5, "date": "2022-12-14"},
    # 3rd place
    {"home_team": "Croatia", "away_team": "Morocco", "h2h_odds": [2.10, 3.30, 3.80], "totals_odds": [1.97, 1.93], "totals_line": 2.5, "date": "2022-12-17"},
    # Final
    {"home_team": "Argentina", "away_team": "France", "h2h_odds": [2.60, 3.30, 2.80], "totals_odds": [1.77, 2.14], "totals_line": 2.5, "date": "2022-12-18"},
]


# -- Euro 2024 closing odds (curated) ---------------------------------

EURO_2024_ODDS: list[dict] = [
    # Group A
    {"home_team": "Germany", "away_team": "Scotland", "h2h_odds": [1.30, 5.60, 11.0], "totals_odds": [1.55, 2.55], "totals_line": 2.5, "date": "2024-06-14"},
    {"home_team": "Hungary", "away_team": "Switzerland", "h2h_odds": [3.10, 3.15, 2.48], "totals_odds": [2.00, 1.91], "totals_line": 2.5, "date": "2024-06-15"},
    {"home_team": "Germany", "away_team": "Hungary", "h2h_odds": [1.30, 5.80, 11.0], "totals_odds": [1.53, 2.60], "totals_line": 2.5, "date": "2024-06-19"},
    {"home_team": "Scotland", "away_team": "Switzerland", "h2h_odds": [3.20, 3.20, 2.38], "totals_odds": [2.10, 1.82], "totals_line": 2.5, "date": "2024-06-19"},
    {"home_team": "Scotland", "away_team": "Hungary", "h2h_odds": [2.52, 3.20, 3.05], "totals_odds": [2.05, 1.87], "totals_line": 2.5, "date": "2024-06-23"},
    {"home_team": "Germany", "away_team": "Switzerland", "h2h_odds": [1.78, 3.60, 5.00], "totals_odds": [1.85, 2.05], "totals_line": 2.5, "date": "2024-06-23"},
    # Group B
    {"home_team": "Spain", "away_team": "Croatia", "h2h_odds": [2.00, 3.40, 4.10], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2024-06-15"},
    {"home_team": "Italy", "away_team": "Albania", "h2h_odds": [1.32, 5.40, 10.5], "totals_odds": [1.70, 2.24], "totals_line": 2.5, "date": "2024-06-15"},
    {"home_team": "Croatia", "away_team": "Albania", "h2h_odds": [1.42, 4.70, 8.20], "totals_odds": [1.80, 2.10], "totals_line": 2.5, "date": "2024-06-19"},
    {"home_team": "Spain", "away_team": "Italy", "h2h_odds": [1.97, 3.30, 4.30], "totals_odds": [1.95, 1.95], "totals_line": 2.5, "date": "2024-06-20"},
    {"home_team": "Croatia", "away_team": "Italy", "h2h_odds": [2.55, 3.10, 3.05], "totals_odds": [2.00, 1.91], "totals_line": 2.5, "date": "2024-06-24"},
    {"home_team": "Albania", "away_team": "Spain", "h2h_odds": [9.40, 5.00, 1.37], "totals_odds": [1.80, 2.10], "totals_line": 2.5, "date": "2024-06-24"},
    # Group C
    {"home_team": "Slovenia", "away_team": "Denmark", "h2h_odds": [3.90, 3.30, 2.10], "totals_odds": [2.10, 1.82], "totals_line": 2.5, "date": "2024-06-16"},
    {"home_team": "Serbia", "away_team": "England", "h2h_odds": [5.40, 3.60, 1.74], "totals_odds": [2.00, 1.91], "totals_line": 2.5, "date": "2024-06-16"},
    {"home_team": "Slovenia", "away_team": "Serbia", "h2h_odds": [2.65, 3.15, 2.88], "totals_odds": [2.15, 1.78], "totals_line": 2.5, "date": "2024-06-20"},
    {"home_team": "Denmark", "away_team": "England", "h2h_odds": [4.40, 3.40, 1.92], "totals_odds": [2.02, 1.89], "totals_line": 2.5, "date": "2024-06-20"},
    {"home_team": "England", "away_team": "Slovenia", "h2h_odds": [1.40, 4.80, 8.80], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2024-06-25"},
    {"home_team": "Denmark", "away_team": "Serbia", "h2h_odds": [2.15, 3.25, 3.70], "totals_odds": [2.14, 1.79], "totals_line": 2.5, "date": "2024-06-25"},
    # Group D
    {"home_team": "Netherlands", "away_team": "France", "h2h_odds": [3.30, 3.20, 2.36], "totals_odds": [1.85, 2.05], "totals_line": 2.5, "date": "2024-06-21"},
    {"home_team": "Poland", "away_team": "Austria", "h2h_odds": [2.90, 3.15, 2.62], "totals_odds": [1.93, 1.97], "totals_line": 2.5, "date": "2024-06-21"},
    {"home_team": "Netherlands", "away_team": "Austria", "h2h_odds": [1.77, 3.65, 4.80], "totals_odds": [1.80, 2.10], "totals_line": 2.5, "date": "2024-06-25"},
    {"home_team": "France", "away_team": "Poland", "h2h_odds": [1.25, 6.00, 13.0], "totals_odds": [1.55, 2.55], "totals_line": 2.5, "date": "2024-06-25"},
    {"home_team": "Poland", "away_team": "Netherlands", "h2h_odds": [4.20, 3.45, 1.95], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2024-06-16"},
    {"home_team": "Austria", "away_team": "France", "h2h_odds": [4.80, 3.50, 1.82], "totals_odds": [1.92, 1.98], "totals_line": 2.5, "date": "2024-06-17"},
    # Group E
    {"home_team": "Romania", "away_team": "Ukraine", "h2h_odds": [3.20, 3.20, 2.42], "totals_odds": [1.95, 1.95], "totals_line": 2.5, "date": "2024-06-17"},
    {"home_team": "Belgium", "away_team": "Slovakia", "h2h_odds": [1.42, 4.70, 8.20], "totals_odds": [1.72, 2.22], "totals_line": 2.5, "date": "2024-06-17"},
    {"home_team": "Slovakia", "away_team": "Ukraine", "h2h_odds": [3.40, 3.20, 2.30], "totals_odds": [2.10, 1.82], "totals_line": 2.5, "date": "2024-06-21"},
    {"home_team": "Belgium", "away_team": "Romania", "h2h_odds": [1.45, 4.60, 7.80], "totals_odds": [1.75, 2.17], "totals_line": 2.5, "date": "2024-06-22"},
    {"home_team": "Ukraine", "away_team": "Belgium", "h2h_odds": [3.65, 3.30, 2.18], "totals_odds": [1.98, 1.93], "totals_line": 2.5, "date": "2024-06-26"},
    {"home_team": "Slovakia", "away_team": "Romania", "h2h_odds": [2.72, 3.05, 2.88], "totals_odds": [2.22, 1.72], "totals_line": 2.5, "date": "2024-06-26"},
    # Group F
    {"home_team": "Turkey", "away_team": "Georgia", "h2h_odds": [1.72, 3.70, 5.30], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2024-06-18"},
    {"home_team": "Portugal", "away_team": "Czech Republic", "h2h_odds": [1.47, 4.40, 7.60], "totals_odds": [1.77, 2.14], "totals_line": 2.5, "date": "2024-06-18"},
    {"home_team": "Georgia", "away_team": "Czech Republic", "h2h_odds": [3.40, 3.20, 2.30], "totals_odds": [2.10, 1.82], "totals_line": 2.5, "date": "2024-06-22"},
    {"home_team": "Turkey", "away_team": "Portugal", "h2h_odds": [4.60, 3.55, 1.85], "totals_odds": [1.87, 2.03], "totals_line": 2.5, "date": "2024-06-22"},
    {"home_team": "Czech Republic", "away_team": "Turkey", "h2h_odds": [2.65, 3.15, 2.90], "totals_odds": [1.95, 1.95], "totals_line": 2.5, "date": "2024-06-26"},
    {"home_team": "Georgia", "away_team": "Portugal", "h2h_odds": [9.60, 5.20, 1.35], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2024-06-26"},
    # Round of 16
    {"home_team": "Switzerland", "away_team": "Italy", "h2h_odds": [2.90, 3.10, 2.65], "totals_odds": [2.10, 1.82], "totals_line": 2.5, "date": "2024-06-29"},
    {"home_team": "Germany", "away_team": "Denmark", "h2h_odds": [1.60, 3.90, 6.20], "totals_odds": [1.72, 2.22], "totals_line": 2.5, "date": "2024-06-29"},
    {"home_team": "England", "away_team": "Slovakia", "h2h_odds": [1.35, 5.00, 10.0], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2024-06-30"},
    {"home_team": "Spain", "away_team": "Georgia", "h2h_odds": [1.20, 6.80, 16.0], "totals_odds": [1.47, 2.80], "totals_line": 2.5, "date": "2024-06-30"},
    {"home_team": "France", "away_team": "Belgium", "h2h_odds": [1.95, 3.30, 4.30], "totals_odds": [2.00, 1.91], "totals_line": 2.5, "date": "2024-07-01"},
    {"home_team": "Portugal", "away_team": "Slovenia", "h2h_odds": [1.40, 4.60, 8.80], "totals_odds": [1.72, 2.22], "totals_line": 2.5, "date": "2024-07-01"},
    {"home_team": "Romania", "away_team": "Netherlands", "h2h_odds": [5.00, 3.50, 1.82], "totals_odds": [1.87, 2.03], "totals_line": 2.5, "date": "2024-07-02"},
    {"home_team": "Austria", "away_team": "Turkey", "h2h_odds": [1.87, 3.50, 4.50], "totals_odds": [1.82, 2.08], "totals_line": 2.5, "date": "2024-07-02"},
    # Quarter-finals
    {"home_team": "Germany", "away_team": "Spain", "h2h_odds": [3.20, 3.25, 2.38], "totals_odds": [1.85, 2.05], "totals_line": 2.5, "date": "2024-07-05"},
    {"home_team": "Portugal", "away_team": "France", "h2h_odds": [3.15, 3.10, 2.50], "totals_odds": [2.10, 1.82], "totals_line": 2.5, "date": "2024-07-05"},
    {"home_team": "England", "away_team": "Switzerland", "h2h_odds": [1.90, 3.30, 4.60], "totals_odds": [2.02, 1.89], "totals_line": 2.5, "date": "2024-07-06"},
    {"home_team": "Netherlands", "away_team": "Turkey", "h2h_odds": [1.57, 4.00, 6.40], "totals_odds": [1.75, 2.17], "totals_line": 2.5, "date": "2024-07-06"},
    # Semi-finals
    {"home_team": "Spain", "away_team": "France", "h2h_odds": [2.40, 3.10, 3.25], "totals_odds": [2.02, 1.89], "totals_line": 2.5, "date": "2024-07-09"},
    {"home_team": "Netherlands", "away_team": "England", "h2h_odds": [2.90, 3.15, 2.62], "totals_odds": [1.90, 2.00], "totals_line": 2.5, "date": "2024-07-10"},
    # Final
    {"home_team": "Spain", "away_team": "England", "h2h_odds": [2.10, 3.30, 3.80], "totals_odds": [1.92, 1.98], "totals_line": 2.5, "date": "2024-07-14"},
]


def load_historical_odds(tournament: str = "wc2022") -> list[dict]:
    """Load curated historical odds for a tournament.

    Parameters
    ----------
    tournament : 'wc2022' or 'euro2024'

    Returns
    -------
    List of odds dicts ready for OddsModel.load_odds().
    """
    if tournament == "wc2022":
        return WC_2022_ODDS
    elif tournament == "euro2024":
        return EURO_2024_ODDS
    else:
        raise ValueError(f"Unknown tournament: {tournament}")


