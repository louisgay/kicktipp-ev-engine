"""The Odds API client with disk-based caching.

Wraps the v4 API (https://the-odds-api.com/liveapi/guides/v4/).
Every response is cached to disk so we never re-fetch the same snapshot.

Environment variables
---------------------
ODDS_API_KEY : str
    API key for The Odds API (free tier: 500 credits/month).

Quota costs (historical endpoint)
---------------------------------
cost = 10 × len(markets) × len(regions)  per call
Empty responses (no data) are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "config" / "config.yaml"
_CACHE_DIR = _ROOT / "data" / "raw" / "odds_cache"

BASE_URL = "https://api.the-odds-api.com"


def _get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        # Try .env file
        env_path = _ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ODDS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY not set. Set it as an environment variable or "
            "in a .env file at the project root."
        )
    return key


def _cache_key(endpoint: str, params: dict) -> str:
    """Deterministic cache key from endpoint + params (excluding apiKey)."""
    filtered = {k: v for k, v in sorted(params.items()) if k != "apiKey"}
    raw = f"{endpoint}|{json.dumps(filtered, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(cache_key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{cache_key}.json"


def _read_cache(cache_key: str) -> dict | None:
    path = _cache_path(cache_key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _write_cache(cache_key: str, data: dict, meta: dict | None = None) -> None:
    path = _cache_path(cache_key)
    payload = {"data": data, "meta": meta or {}, "cached_at": time.time()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Cached response: %s", path.name)


# -- API methods ------------------------------------------------------


def get_sports(all_sports: bool = True) -> list[dict]:
    """GET /v4/sports - discover available sport keys.
    Free: does NOT count against quota.
    """
    endpoint = "/v4/sports"
    params = {"apiKey": _get_api_key()}
    if all_sports:
        params["all"] = "true"

    ck = _cache_key(endpoint, params)
    cached = _read_cache(ck)
    if cached:
        logger.info("Sports list from cache")
        return cached["data"]

    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    _log_quota(resp)
    _write_cache(ck, data)
    return data


def get_odds(
    sport: str,
    regions: str = "eu",
    markets: str = "h2h",
    odds_format: str = "decimal",
    bookmakers: str | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """GET /v4/sports/{sport}/odds - current (live) odds.

    Quota cost: 1 × n_markets × n_regions. ``force_refresh=True`` bypasses the
    disk cache (which has no TTL) so the freshest snapshot is fetched - used by
    the pre-kickoff refresh chain. Default False preserves the cached behaviour.
    """
    endpoint = f"/v4/sports/{sport}/odds"
    params = {
        "apiKey": _get_api_key(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    ck = _cache_key(endpoint, params)
    if not force_refresh:
        cached = _read_cache(ck)
        if cached:
            logger.info("Live odds from cache (%s)", sport)
            return cached["data"]

    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    n_markets = len(markets.split(","))
    n_regions = len(regions.split(","))
    cost = n_markets * n_regions
    meta = {"quota_cost": cost}
    _log_quota(resp, meta)
    _write_cache(ck, data, meta)
    return data


def _log_quota(resp: requests.Response, meta: dict | None = None) -> None:
    """Log quota usage from response headers."""
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    last = resp.headers.get("x-requests-last", "?")
    logger.info("Quota: remaining=%s, used=%s, this_call=%s", remaining, used, last)
    if meta is not None:
        meta["remaining"] = remaining
        meta["used"] = used


# -- Helpers ----------------------------------------------------------


def list_soccer_sports() -> list[dict]:
    """Return only soccer sport keys."""
    sports = get_sports(all_sports=True)
    return [s for s in sports if s.get("key", "").startswith("soccer")]


def find_sport_key(keyword: str) -> list[dict]:
    """Search sport keys by keyword."""
    sports = get_sports(all_sports=True)
    kw = keyword.lower()
    return [s for s in sports if kw in s.get("key", "").lower()
            or kw in s.get("title", "").lower()]
