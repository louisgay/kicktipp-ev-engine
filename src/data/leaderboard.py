"""Read-only scraper/parser for the kicktipp leaderboard ("tippuebersicht").

The leaderboard page exposes every player's prediction for every match
(once the match has kicked off - picks are hidden until each deadline), plus
the standings columns P / B / W / T.

This is the raw material for opponent modelling: see ``src.opponents`` for the
logger that accumulates these picks over time and the behavioural summaries.

Strictly read-only (GET only). Reuses the session/auth helpers from
``src.data.kicktipp_scrape``.

Page structure (as of 2026-06)
------------------------------
- ``table#spielplanSpiele`` - the fixture list (date, home, away, group,
  result), one row per match, in the same order as the pick columns.
- ``table#ranking.tippuebersicht`` - header row + one row per player. Each
  player row has cells ``td.ereignis{i}`` holding that player's pick for match
  ``i`` (text like ``"2-0 4"`` = pick 2-0 worth 4 pts, ``"0-1"`` = pick with 0
  pts, ``"---"`` = no visible pick), and the P/B/W/T total cells.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from src.data.kicktipp_scrape import BASE_URL, COMMUNITY, _fetch_page, _make_session

logger = logging.getLogger(__name__)


# -- Data classes -----------------------------------------------------


@dataclass
class Fixture:
    """A single match on the leaderboard fixture list."""

    index: int          # 0-based column index, aligns with td.ereignis{index}
    date: str
    home: str
    away: str
    group: str
    result: tuple[int, int] | None  # None if not played yet


@dataclass
class PlayerRow:
    """One player's standings row, including their visible picks."""

    rank: int | None
    name: str
    # match_index -> (pick, points). pick is None when no pick is visible.
    picks: dict[int, tuple[tuple[int, int] | None, int]] = field(default_factory=dict)
    matchday_points: float = 0.0
    bonus: float = 0.0
    wins: float = 0.0          # fractional matchday wins (the W column)
    total: float = 0.0


@dataclass
class Leaderboard:
    fixtures: list[Fixture]
    players: list[PlayerRow]
    spieltag: int | None = None  # matchday index this page was scraped from


# -- Parsing helpers --------------------------------------------------


def _parse_result(text: str) -> tuple[int, int] | None:
    """Parse a fixture result cell ('2 - 0' / '- - -')."""
    m = re.search(r"(\d+)\s*-\s*(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _parse_pick_cell(text: str) -> tuple[tuple[int, int] | None, int]:
    """Parse a player's pick cell.

    Examples: '2-0 4' -> ((2,0), 4); '0-1' -> ((0,1), 0); '---' -> (None, 0).
    """
    text = " ".join(text.split())
    if not text or text == "---":
        return None, 0
    m = re.match(r"^(\d+)-(\d+)(?:\s+(\d+))?$", text)
    if m:
        pick = (int(m.group(1)), int(m.group(2)))
        return pick, int(m.group(3)) if m.group(3) else 0
    # Fallback: any "a - b" anywhere in the cell
    m2 = re.search(r"(\d+)\s*-\s*(\d+)", text)
    return ((int(m2.group(1)), int(m2.group(2))), 0) if m2 else (None, 0)


def _to_float(text: str) -> float:
    text = text.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _ereignis_index(cell) -> int | None:
    """Return the match index N from a cell's 'ereignisN' class, else None."""
    for cls in cell.get("class", []):
        m = re.match(r"^ereignis(\d+)$", cls)
        if m:
            return int(m.group(1))
    return None


# -- Main parser ------------------------------------------------------


def parse_leaderboard(html: str, spieltag: int | None = None) -> Leaderboard:
    """Parse the leaderboard HTML into fixtures + player rows.

    ``spieltag`` is stamped onto the result (the matchday index this HTML came
    from) - it cannot be inferred from the page reliably, so the caller passes
    it through. Match column indices (``ereignis{i}``) reset per matchday, so
    ``spieltag`` is required to key picks globally.
    """
    soup = BeautifulSoup(html, "lxml")

    # -- Fixtures --
    fixtures: list[Fixture] = []
    fx_table = soup.find("table", id="spielplanSpiele")
    if fx_table is not None:
        for ri, tr in enumerate(fx_table.find_all("tr")):
            cells = tr.find_all(["th", "td"])
            # Group stage: [Date, Home, Away, Group, Result] (5 cols).
            # Knockout:    [Date, Home, Away, Result] (4 cols, no Group column;
            # the Result cell may carry an a.PSO suffix, e.g. "4-5a.PSO").
            if len(cells) < 4:
                continue
            txt = [c.get_text(" ", strip=True) for c in cells]
            if txt[0].lower() == "date":  # header
                continue
            has_group = len(cells) >= 5
            fixtures.append(Fixture(
                index=len(fixtures),
                date=txt[0],
                home=txt[1],
                away=txt[2],
                group=txt[3] if has_group else "",
                result=_parse_result(txt[-1]),
            ))
    logger.info("Parsed %d fixtures", len(fixtures))

    # -- Players --
    players: list[PlayerRow] = []
    rk_table = soup.find("table", id="ranking")
    if rk_table is None:
        logger.warning("No #ranking table found")
        return Leaderboard(fixtures, players, spieltag=spieltag)

    for tr in rk_table.find_all("tr"):
        classes = tr.get("class", []) or []
        if "headerErgebnis" in classes or "teilnehmer" not in classes:
            continue  # skip header / non-player rows

        cells = tr.find_all("td")
        by_class: dict[str, object] = {}
        picks: dict[int, tuple[tuple[int, int] | None, int]] = {}
        for c in cells:
            cls = c.get("class", []) or []
            idx = _ereignis_index(c)
            if idx is not None:
                picks[idx] = _parse_pick_cell(c.get_text(" ", strip=True))
                continue
            for key in ("position", "mg_class", "name", "spieltagspunkte",
                        "bonus", "siege", "gesamtpunkte"):
                if key in cls:
                    by_class[key] = c

        name_cell = by_class.get("mg_class") or by_class.get("name")
        if name_cell is None:
            continue
        name = name_cell.get_text(" ", strip=True)

        rank = None
        if "position" in by_class:
            rm = re.search(r"\d+", by_class["position"].get_text(strip=True))
            rank = int(rm.group()) if rm else None

        players.append(PlayerRow(
            rank=rank,
            name=name,
            picks=picks,
            matchday_points=_to_float(by_class["spieltagspunkte"].get_text(strip=True))
                if "spieltagspunkte" in by_class else 0.0,
            bonus=_to_float(by_class["bonus"].get_text(strip=True))
                if "bonus" in by_class else 0.0,
            wins=_to_float(by_class["siege"].get_text(strip=True))
                if "siege" in by_class else 0.0,
            total=_to_float(by_class["gesamtpunkte"].get_text(strip=True))
                if "gesamtpunkte" in by_class else 0.0,
        ))

    logger.info("Parsed %d players", len(players))
    return Leaderboard(fixtures, players, spieltag=spieltag)


# -- Live scrape ------------------------------------------------------

# Season ("Tippsaison") id - pool/season-specific and NOT shipped. Read it off
# any leaderboard URL (?tippsaisonId=...) and pass it via the ``tippsaison_id``
# argument or the --season-id flag. Left unset by default (fetches the pool's
# current season); no real pool identifier ships in this repo.
DEFAULT_TIPPSAISON_ID: int | None = None


def _leaderboard_url(
    community: str = COMMUNITY,
    spieltag: int | None = None,
    tippsaison_id: int | None = DEFAULT_TIPPSAISON_ID,
) -> str:
    """Build a leaderboard URL, optionally pinned to a season + matchday."""
    url = f"{BASE_URL}/{community}/leaderboard"
    params = []
    if tippsaison_id is not None:
        params.append(f"tippsaisonId={tippsaison_id}")
    if spieltag is not None:
        params.append(f"spieltagIndex={spieltag}")
    if params:
        url += "?" + "&".join(params)
    return url


def fetch_leaderboard_html(
    community: str = COMMUNITY,
    spieltag: int | None = None,
    tippsaison_id: int | None = DEFAULT_TIPPSAISON_ID,
) -> str:
    """GET a leaderboard page (requires KICKTIPP_SESSION). Returns raw HTML.

    With ``spieltag=None`` the site returns its default (current) matchday;
    pass an explicit index to target a specific matchday.
    """
    session = _make_session()
    url = _leaderboard_url(community, spieltag, tippsaison_id)
    soup = _fetch_page(session, url)
    return str(soup)


def scrape_leaderboard(
    community: str = COMMUNITY,
    spieltag: int | None = None,
    tippsaison_id: int | None = DEFAULT_TIPPSAISON_ID,
) -> Leaderboard:
    """Scrape and parse a single matchday's leaderboard."""
    html = fetch_leaderboard_html(community, spieltag, tippsaison_id)
    return parse_leaderboard(html, spieltag=spieltag)
