"""Test configuration - seeds group match cache for offline tests."""

from __future__ import annotations

import src.bonus.group_sim as gs
from tests.fixtures.group_matches import KICKTIPP_GROUP_MATCHES


def pytest_configure(config):
    """Seed the group match cache so tests don't require live scraping."""
    gs._write_cache(KICKTIPP_GROUP_MATCHES)
