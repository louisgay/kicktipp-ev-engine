"""Read-only scraper for kicktipp.co.uk prediction pages.

Extracts:
1. Match odds (1X2) from the prediction page - these are pre-devigged
   fair probabilities displayed by kicktipp (sum ≈ 100%).
2. Bonus questions with options and deadlines from the bonus page.

NO form submission. Strictly read-only (GET requests only).

Authentication
--------------
Uses a persistent login cookie provided via the KICKTIPP_SESSION
environment variable (or .env file). No email/password login.

Set in .env:
    KICKTIPP_SESSION=<your-login-cookie-value>

To obtain this value:
1. Log in to kicktipp.co.uk in your browser
2. Open Developer Tools -> Application -> Cookies
3. Copy the value of the 'login' cookie
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

BASE_URL = "https://www.kicktipp.co.uk"


def _get_pool_slug() -> str:
    """Kicktipp community (pool) slug, read from the KICKTIPP_POOL env var or .env.

    The real pool slug is intentionally not shipped. Set your own to run live
    scrapes; offline tests use fixtures and never hit the network.
    """
    slug = os.environ.get("KICKTIPP_POOL", "")
    if not slug:
        env_path = _ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("KICKTIPP_POOL="):
                    slug = line.split("=", 1)[1].strip().strip("'\"")
                    break
    return slug or "your-pool-slug"


COMMUNITY = _get_pool_slug()
REQUEST_DELAY = 2.0  # seconds between requests


# -- Data classes -----------------------------------------------------


@dataclass
class MatchOdds:
    """Parsed 1X2 odds for a single match."""
    match_id: str
    datetime_str: str
    home_team: str
    away_team: str
    odds_home: float    # decimal odds displayed by kicktipp
    odds_draw: float
    odds_away: float
    prob_home: float    # renormalised probabilities (sum = 1)
    prob_draw: float
    prob_away: float
    overround: float    # sum of implied probs before renormalisation
    result: str | None = None  # "---" if not played yet
    a_pso: bool = False  # knockout (a.PSO): 2-way advance odds, prob_draw == 0


@dataclass
class BonusQuestion:
    """Parsed bonus question."""
    question_id: str
    question_text: str
    deadline: str
    options: list[dict]   # [{"value": "1", "label": "Argentina"}, ...]
    question_type: str    # "single" or "multi"
    select_count: int | None = None  # for multi-select: how many to pick


# -- Session / Authentication -----------------------------------------


def _get_session_cookie() -> str:
    """Read the kicktipp session cookie from environment."""
    cookie = os.environ.get("KICKTIPP_SESSION", "")
    if not cookie:
        env_path = _ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("KICKTIPP_SESSION="):
                    cookie = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not cookie:
        raise RuntimeError(
            "KICKTIPP_SESSION not set. Set it as an environment variable "
            "or in .env at the project root. See README for instructions."
        )
    return cookie


def _make_session() -> requests.Session:
    """Create a requests.Session with the kicktipp login cookie."""
    session = requests.Session()
    cookie = _get_session_cookie()
    session.cookies.set("login", cookie, domain="www.kicktipp.co.uk")
    session.headers.update({
        "User-Agent": "kicktipp-ev-engine-research/0.1 (read-only)",
    })
    return session


def _fetch_page(session: requests.Session, url: str) -> BeautifulSoup:
    """GET a page with rate limiting, return parsed soup."""
    time.sleep(REQUEST_DELAY)
    logger.info("GET %s", url)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


# -- Parsing: prediction page (1X2 odds) -----------------------------


def parse_prediction_page(html: str) -> list[MatchOdds]:
    """Parse a kicktipp prediction page and extract 1X2 odds.

    The odds displayed by kicktipp are pre-devigged (fair probabilities).
    We verify this by checking that the sum of implied probabilities ≈ 1.00.
    If overround > 1.03, a warning is logged.

    Parameters
    ----------
    html : raw HTML string of the prediction page

    Returns
    -------
    List of MatchOdds, one per match on the page.
    """
    soup = BeautifulSoup(html, "lxml")
    matches = []

    # Find the prediction table
    table = soup.find("table", class_=lambda c: c and "tippabgabe" in c)
    if table is None:
        # Fallback: look for any table inside kicktipp-content
        content = soup.find(id="kicktipp-content")
        if content:
            table = content.find("table")
    if table is None:
        logger.warning("No prediction table found in HTML")
        return matches

    tbody = table.find("tbody")
    if tbody is None:
        tbody = table

    for tr in tbody.find_all("tr"):
        # -- Extract odds: try legacy (td.kicktipp-wettquote) then new layout --
        odds_cells = tr.find_all("td", class_="kicktipp-wettquote")
        odds_h = odds_d = odds_a = None

        if len(odds_cells) >= 3:
            # Legacy layout: three separate <td class="kicktipp-wettquote">
            try:
                odds_h = float(odds_cells[0].get_text(strip=True))
                odds_d = float(odds_cells[1].get_text(strip=True))
                odds_a = float(odds_cells[2].get_text(strip=True))
            except (ValueError, IndexError):
                pass

        if odds_h is None:
            # New layout: <div class="tippabgabe-quoten"> with span.quote-text
            quoten_div = tr.find("div", class_="tippabgabe-quoten")
            if quoten_div:
                spans = quoten_div.find_all("span", class_="quote-text")
                if len(spans) >= 3:
                    try:
                        odds_h = float(spans[0].get_text(strip=True))
                        odds_d = float(spans[1].get_text(strip=True))
                        odds_a = float(spans[2].get_text(strip=True))
                    except (ValueError, IndexError):
                        pass

        # Knockout (a.PSO) layout: a single <td class="... quoten"> carries the
        # odds inline as "1 <oh> X <od> 2 <oa>" - and kicktipp prints the draw
        # price as 0.00 (a 2-way "who advances" market, no draw).
        a_pso = bool(tr.find(class_=lambda c: c and "spielabschnitt" in c)) \
            or "a.PSO" in tr.get_text()
        if odds_h is None:
            quoten_td = tr.find("td", class_=lambda c: c and "quoten" in c)
            if quoten_td:
                m = re.search(r"1\s+([\d.]+)\s+X\s+([\d.]+)\s+2\s+([\d.]+)",
                              quoten_td.get_text(" ", strip=True))
                if m:
                    try:
                        odds_h, odds_d, odds_a = (
                            float(m.group(1)), float(m.group(2)), float(m.group(3)))
                    except ValueError:
                        pass

        if odds_h is None:
            continue

        # -- Extract match info --
        all_tds = tr.find_all("td")
        datetime_str = ""
        home_team = ""
        away_team = ""
        result = None
        match_id = tr.get("id", "")

        # Try col-class based extraction (legacy)
        for td in all_tds:
            classes = td.get("class", [])
            if "col1" in classes:
                datetime_str = td.get_text(strip=True)
            elif "col3" in classes:
                home_team = td.get_text(strip=True)
            elif "col5" in classes:
                away_team = td.get_text(strip=True)
            elif "col6" in classes:
                result = td.get_text(strip=True)
                if result == "---":
                    result = None

        # Fallback: positional extraction for new layout
        # New layout: td[0]=time, td[1]=home, td[2]=away, td[3]=inputs, td[4]=quoten
        if not home_team:
            plain_tds = [
                td for td in all_tds
                if "quoten" not in (td.get("class") or [])
                and "kicktipp-tippabgabe" not in (td.get("class") or [])
            ]
            if len(plain_tds) >= 3:
                datetime_str = plain_tds[0].get_text(strip=True)
                home_team = plain_tds[1].get_text(strip=True)
                away_team = plain_tds[2].get_text(strip=True)
            elif len(all_tds) >= 6:
                datetime_str = all_tds[0].get_text(strip=True)
                home_team = all_tds[2].get_text(strip=True)
                away_team = all_tds[4].get_text(strip=True)
                res_text = all_tds[5].get_text(strip=True)
                result = None if res_text == "---" else res_text

        # Extract match_id from input fields if not on the tr
        if not match_id:
            inp = tr.find("input", attrs={"name": re.compile(r"spieltippForms\[")})
            if inp:
                m_id = re.search(r"spieltippForms\[(\d+)\]", inp.get("name", ""))
                if m_id:
                    match_id = m_id.group(1)

        # a.PSO knockout: 2-way "who advances" market - kicktipp prints the draw
        # price as 0.00. Devig over home+away only; prob_draw is 0 by construction.
        if a_pso and odds_d == 0:
            if odds_h <= 0 or odds_a <= 0:
                logger.debug("Skipping %s vs %s: non-positive a.PSO odds",
                             home_team, away_team)
                continue
            implied = [1.0 / odds_h, 1.0 / odds_a]
            overround = sum(implied)
            prob_h = implied[0] / overround
            prob_d = 0.0
            prob_a = implied[1] / overround
        else:
            # Compute probabilities - guard against odds-less matchdays: when no
            # market is posted (e.g. future spieltag) kicktipp shows quotes of 0,
            # which would divide by zero here and crash scrape_deadlines.
            if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
                logger.debug("Skipping %s vs %s: non-positive odds (no market posted)",
                             home_team, away_team)
                continue
            implied = [1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a]
            overround = sum(implied)

            if overround > 1.03:
                logger.warning(
                    "High overround (%.4f) for %s vs %s - odds may NOT be "
                    "pre-devigged. Check source.",
                    overround, home_team, away_team,
                )

            # Renormalise to sum = 1
            prob_h = implied[0] / overround
            prob_d = implied[1] / overround
            prob_a = implied[2] / overround

        matches.append(MatchOdds(
            match_id=match_id,
            datetime_str=datetime_str,
            home_team=home_team,
            away_team=away_team,
            odds_home=odds_h,
            odds_draw=odds_d,
            odds_away=odds_a,
            prob_home=round(prob_h, 6),
            prob_draw=round(prob_d, 6),
            prob_away=round(prob_a, 6),
            overround=round(overround, 6),
            result=result,
            a_pso=a_pso,
        ))

    logger.info("Parsed %d matches from prediction page", len(matches))
    return matches


# -- Parsing: bonus page ----------------------------------------------


def parse_bonus_page(html: str) -> list[BonusQuestion]:
    """Parse the kicktipp bonus predictions page.

    Extracts all bonus questions, their options, deadlines, and type
    (single-select or multi-select).

    Parameters
    ----------
    html : raw HTML string of the bonus page

    Returns
    -------
    List of BonusQuestion.
    """
    soup = BeautifulSoup(html, "lxml")
    questions = []

    # Find all bonus question rows
    table = soup.find("table", class_=lambda c: c and "bonustipps" in c)
    if table is None:
        content = soup.find(id="kicktipp-content")
        if content:
            # Prefer table containing fragebox elements (bonus questions)
            for t in content.find_all("table"):
                if t.find("div", class_="fragebox"):
                    table = t
                    break
            if table is None:
                table = content.find("table")
    if table is None:
        logger.warning("No bonus table found in HTML")
        return questions

    tbody = table.find("tbody") or table

    for tr in tbody.find_all("tr"):
        # Find question text
        q_span = tr.find("span", class_="bonusfrage")
        if q_span is None:
            # Fallback: look for any span with question-like text
            spans = tr.find_all("span")
            for s in spans:
                text = s.get_text(strip=True)
                if "?" in text or "which" in text.lower() or "who" in text.lower():
                    q_span = s
                    break
        # Fallback for new layout: question text in a plain <td>, fragebox present
        question_text = ""
        if q_span is not None:
            question_text = q_span.get_text(strip=True)
        elif tr.find("div", class_="fragebox"):
            # New layout: bonus rows have div.fragebox, question text in a plain td
            for td in tr.find_all("td"):
                td_text = td.get_text(strip=True)
                td_classes = td.get("class", [])
                if ("?" in td_text or "which" in td_text.lower()
                        or "who" in td_text.lower()):
                    if ("kicktipp-tippabgabe" not in td_classes
                            and "kicktipp-time" not in td_classes):
                        question_text = td_text
                        break
        if not question_text:
            continue

        question_id = tr.get("id", "")
        # Fallback: extract question_id from form input name
        if not question_id:
            inp = tr.find("select", attrs={"name": re.compile(r"fragetippForms\[")})
            if inp:
                m_id = re.search(r"fragetippForms\[(\d+)\]", inp.get("name", ""))
                if m_id:
                    question_id = m_id.group(1)

        # Find deadline
        deadline = ""
        dl_span = tr.find("span", class_="bonusdeadline")
        if dl_span:
            deadline_text = dl_span.get_text(strip=True)
            # Extract the date portion
            deadline = deadline_text.replace("Deadline:", "").strip()
        # Fallback: use time cell as deadline
        if not deadline:
            time_td = tr.find("td", class_="kicktipp-time")
            if time_td:
                deadline = time_td.get_text(strip=True)

        # Find options (select element)
        all_selects = tr.find_all("select")
        select = all_selects[0] if all_selects else None
        options = []
        question_type = "single"
        select_count = None

        if select:
            if select.get("multiple"):
                question_type = "multi"
                # Look for hint about how many to select
                hint = tr.find("span", class_="hint")
                if hint:
                    hint_text = hint.get_text(strip=True)
                    match = re.search(r"(\d+)", hint_text)
                    if match:
                        select_count = int(match.group(1))

            # Multiple single-selects in same row = multi-select (pick N)
            # e.g. semi-finalists: 4 dropdowns, same question ID
            if len(all_selects) > 1 and not select.get("multiple"):
                question_type = "multi"
                select_count = len(all_selects)

            for opt in select.find_all("option"):
                val = opt.get("value", "")
                label = opt.get_text(strip=True)
                if val and label and label not in ("-- Select --", "-- Not saved --"):
                    options.append({"value": val, "label": label})

        questions.append(BonusQuestion(
            question_id=question_id,
            question_text=question_text,
            deadline=deadline,
            options=options,
            question_type=question_type,
            select_count=select_count,
        ))

    logger.info("Parsed %d bonus questions", len(questions))
    return questions


# -- Live scraping (requires KICKTIPP_SESSION) ------------------------


def scrape_matchday(matchday: int = 1,
                    community: str = COMMUNITY) -> list[MatchOdds]:
    """Scrape odds for a specific matchday (requires authentication).

    Parameters
    ----------
    matchday : matchday index (1-based)
    community : kicktipp community slug

    Returns
    -------
    List of MatchOdds for that matchday.
    """
    session = _make_session()
    url = f"{BASE_URL}/{community}/tippabgabe?spieltagIndex={matchday}"
    soup = _fetch_page(session, url)
    return parse_prediction_page(str(soup))


def scrape_bonus(
    community: str = COMMUNITY,
    matchday: int = 1,
) -> list[BonusQuestion]:
    """Scrape bonus questions (requires authentication).

    Bonus questions are embedded inline on the prediction page,
    so this fetches the matchday prediction page and parses bonus
    questions from it.

    Returns
    -------
    List of BonusQuestion.
    """
    session = _make_session()
    url = f"{BASE_URL}/{community}/tippabgabe?spieltagIndex={matchday}"
    soup = _fetch_page(session, url)
    return parse_bonus_page(str(soup))


def scrape_all_matchdays(
    n_matchdays: int = 15,
    community: str = COMMUNITY,
) -> list[MatchOdds]:
    """Scrape odds for all matchdays."""
    session = _make_session()
    all_matches = []
    for md in range(1, n_matchdays + 1):
        url = f"{BASE_URL}/{community}/tippabgabe?spieltagIndex={md}"
        soup = _fetch_page(session, url)
        matches = parse_prediction_page(str(soup))
        if not matches:
            logger.info("No matches found for matchday %d, stopping", md)
            break
        all_matches.extend(matches)
        logger.info("Matchday %d: %d matches", md, len(matches))
    return all_matches


# -- Offline parsing (from saved HTML files) --------------------------


def parse_file(path: str | Path) -> list[MatchOdds] | list[BonusQuestion]:
    """Parse a saved HTML file. Auto-detects prediction vs bonus page."""
    path = Path(path)
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    if soup.find("table", class_=lambda c: c and "bonustipps" in c):
        return parse_bonus_page(html)
    else:
        return parse_prediction_page(html)


# -- Display helpers --------------------------------------------------


def format_odds_table(matches: list[MatchOdds]) -> str:
    """Format matches as a readable table."""
    lines = []
    header = (f"{'Date':>16}  {'Home':>20} vs {'Away':<20}  "
              f"{'1':>6} {'X':>6} {'2':>6}  "
              f"{'P(H)':>6} {'P(D)':>6} {'P(A)':>6}  {'OR':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for m in matches:
        lines.append(
            f"{m.datetime_str:>16}  {m.home_team:>20} vs {m.away_team:<20}  "
            f"{m.odds_home:6.2f} {m.odds_draw:6.2f} {m.odds_away:6.2f}  "
            f"{m.prob_home:6.3f} {m.prob_draw:6.3f} {m.prob_away:6.3f}  "
            f"{m.overround:6.4f}"
        )
    return "\n".join(lines)


def format_bonus_table(questions: list[BonusQuestion]) -> str:
    """Format bonus questions as a readable table."""
    lines = []
    for q in questions:
        lines.append(f"\n[{q.question_id}] {q.question_text}")
        lines.append(f"  Type: {q.question_type}"
                     + (f" (select {q.select_count})" if q.select_count else ""))
        lines.append(f"  Deadline: {q.deadline}")
        lines.append(f"  Options ({len(q.options)}):")
        for opt in q.options[:8]:  # show first 8
            lines.append(f"    - {opt['label']} (val={opt['value']})")
        if len(q.options) > 8:
            lines.append(f"    ... and {len(q.options) - 8} more")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        result = parse_file(path)
        if result and isinstance(result[0], MatchOdds):
            print(format_odds_table(result))
        elif result and isinstance(result[0], BonusQuestion):
            print(format_bonus_table(result))
    else:
        print("Usage: python -m src.data.kicktipp_scrape <html_file>")
        print("       or set KICKTIPP_SESSION and call scrape_matchday()")
