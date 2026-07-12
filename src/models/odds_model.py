"""Odds-based score predictor.

Wraps devigged bookmaker odds into a ScorePredictor that can be
used with the existing backtest and scoring infrastructure.

Unlike the Dixon-Coles model which is fitted on historical matches,
this model requires pre-computed odds data keyed by (home_team, away_team).

Models
------
OddsModel          : bookmaker h2h + O/U (devigged via Shin / normalise)
KicktippOddsModel  : kicktipp pre-devigged 1X2 + optional Odds API O/U
EnsembleModel      : weighted average of any ScorePredictor instances
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.data.clean import normalise_team
from src.models.base import ScorePredictor
from src.odds.devig import devig_1x2, devig_over_under
from src.odds.reconstruct import reconstruct_matrix

logger = logging.getLogger(__name__)


class OddsModel(ScorePredictor):
    """Score predictor that reconstructs P(i,j) from bookmaker odds.

    This model does NOT use fit() in the traditional sense. Instead,
    call load_odds() with a DataFrame of odds data before predict.
    fit() is a no-op (for interface compatibility).
    """

    def __init__(
        self,
        devig_method: str = "normalise",
        rho: float = -0.04,
        max_goals: int = 8,
    ):
        self.devig_method = devig_method
        self.rho = rho
        self.max_goals = max_goals
        # odds_data: keyed by (home_team, away_team) -> odds dict
        self._odds: dict[tuple[str, str], dict] = {}
        self.teams_: list[str] = []
        self.team_idx_: dict[str, int] = {}

    def load_odds(self, odds_records: list[dict]) -> OddsModel:
        """Load odds data.

        Parameters
        ----------
        odds_records : list of dicts, each with keys:
            home_team, away_team, h2h_odds (list[3]),
            optionally totals_odds (list[2]) and totals_line.
        """
        teams = set()
        for rec in odds_records:
            key = (rec["home_team"], rec["away_team"])
            self._odds[key] = rec
            teams.add(rec["home_team"])
            teams.add(rec["away_team"])

        self.teams_ = sorted(teams)
        self.team_idx_ = {t: i for i, t in enumerate(self.teams_)}
        logger.info("Loaded odds for %d matches, %d teams",
                    len(self._odds), len(self.teams_))
        return self

    def fit(self, matches: pd.DataFrame, **kwargs) -> OddsModel:
        """No-op: odds model doesn't learn from match results."""
        logger.info("OddsModel.fit() is a no-op; use load_odds() instead.")
        return self

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        match_date: date | str | None = None,
        neutral: bool = False,
        max_goals: int | None = None,
    ) -> np.ndarray:
        if max_goals is None:
            max_goals = self.max_goals

        key = (home_team, away_team)
        if key not in self._odds:
            raise KeyError(f"No odds data for {home_team} vs {away_team}")

        rec = self._odds[key]
        h2h = rec["h2h_odds"]

        # Devig 1X2
        p_home, p_draw, p_away = devig_1x2(
            h2h[0], h2h[1], h2h[2], method=self.devig_method
        )

        # Devig O/U if available
        p_over = None
        if rec.get("totals_odds") and rec.get("totals_line") == 2.5:
            p_over, _ = devig_over_under(
                rec["totals_odds"][0], rec["totals_odds"][1],
                method=self.devig_method,
            )

        matrix = reconstruct_matrix(
            p_home, p_draw, p_away,
            p_over_2_5=p_over,
            rho=self.rho,
            max_goals=max_goals,
        )
        return matrix

    def predict_lambda(
        self,
        home_team: str,
        away_team: str,
        match_date: date | str | None = None,
        neutral: bool = False,
    ) -> tuple[float, float]:
        key = (home_team, away_team)
        if key not in self._odds:
            raise KeyError(f"No odds data for {home_team} vs {away_team}")

        rec = self._odds[key]
        h2h = rec["h2h_odds"]

        p_home, p_draw, p_away = devig_1x2(
            h2h[0], h2h[1], h2h[2], method=self.devig_method
        )

        p_over = None
        if rec.get("totals_odds") and rec.get("totals_line") == 2.5:
            p_over, _ = devig_over_under(
                rec["totals_odds"][0], rec["totals_odds"][1],
                method=self.devig_method,
            )

        from src.odds.reconstruct import reconstruct_lambdas
        lam_h, lam_a, _ = reconstruct_lambdas(
            p_home, p_draw, p_away, p_over, self.rho, self.max_goals
        )
        return lam_h, lam_a


class KicktippOddsModel(ScorePredictor):
    """Score predictor using kicktipp pre-devigged 1X2 + Odds API O/U.

    Kicktipp displays odds with overround ≈ 1.00, so no devigging is
    needed on the 1X2 component - we just renormalise to sum = 1
    (already done by the parser).

    The total-goals constraint (O/U 2.5) comes from The Odds API and
    IS devigged here.  When O/U is unavailable, reconstruction uses
    1X2 alone (less constrained but still valid).

    Team names are normalised via ``normalise_team()`` on load so that
    lookups match the backtest / results data.
    """

    OVERROUND_WARN = 1.03  # alert if kicktipp odds have real margin

    def __init__(
        self,
        rho: float = -0.04,
        max_goals: int = 8,
        ou_devig_method: str = "normalise",
        require_ou: bool = True,
        blend_weight: float = 0.0,
    ):
        self.rho = rho
        self.max_goals = max_goals
        self.ou_devig_method = ou_devig_method
        self.require_ou = require_ou
        self.blend_weight = blend_weight
        # kicktipp 1X2: keyed by (home, away) -> (p_h, p_d, p_a)
        self._probs_1x2: dict[tuple[str, str], tuple[float, float, float]] = {}
        # devigged sharp 1X2 (Shin), keyed like _probs_1x2
        self._sharp_1x2: dict[tuple[str, str], tuple[float, float, float]] = {}
        # O/U: keyed by (home, away) -> p_over_2_5
        self._p_over: dict[tuple[str, str], float] = {}
        self.teams_: list[str] = []
        self.team_idx_: dict[str, int] = {}

    # -- Loading ------------------------------------------------------

    def load_kicktipp_odds(self, matches: list) -> KicktippOddsModel:
        """Load parsed kicktipp MatchOdds (pre-devigged 1X2).

        Parameters
        ----------
        matches : list of ``MatchOdds`` from kicktipp_scrape.parse_prediction_page
        """
        teams: set[str] = set()
        n_warn = 0
        for m in matches:
            h = normalise_team(m.home_team)
            a = normalise_team(m.away_team)
            key = (h, a)
            self._probs_1x2[key] = (m.prob_home, m.prob_draw, m.prob_away)
            teams.update([h, a])
            if m.overround > self.OVERROUND_WARN:
                n_warn += 1

        self.teams_ = sorted(teams)
        self.team_idx_ = {t: i for i, t in enumerate(self.teams_)}
        logger.info(
            "Loaded kicktipp 1X2 for %d matches (%d teams, %d high-overround)",
            len(self._probs_1x2), len(self.teams_), n_warn,
        )
        return self

    def load_totals(self, totals_records: list[dict]) -> KicktippOddsModel:
        """Load O/U 2.5 odds from The Odds API (will be devigged).

        Parameters
        ----------
        totals_records : list of dicts with keys
            home_team, away_team, totals_odds (list[2]: [over, under]),
            totals_line (must be 2.5).
        """
        for rec in totals_records:
            h = normalise_team(rec["home_team"])
            a = normalise_team(rec["away_team"])
            line = rec.get("totals_line")
            odds = rec.get("totals_odds")
            if line != 2.5 or not odds or len(odds) < 2:
                continue
            p_over, _ = devig_over_under(
                odds[0], odds[1], method=self.ou_devig_method,
            )
            self._p_over[(h, a)] = p_over

        logger.info("Loaded O/U 2.5 for %d matches", len(self._p_over))
        return self

    def load_sharp_1x2(self, records: list) -> "KicktippOddsModel":
        """Store devigged sharp h2h (Shin) for blending. No-op when blend_weight=0.

        records : dicts with home_team, away_team, h2h_odds=[home, draw, away].
        Sharp odds carry vig + favourite-longshot structure, so Shin is correct
        here; kicktipp's own 1X2 is left as-is (already ~vig-free).
        """
        for rec in records:
            o = rec.get("h2h_odds")
            if not o or len(o) < 3:
                continue
            h = normalise_team(rec["home_team"])
            a = normalise_team(rec["away_team"])
            self._sharp_1x2[(h, a)] = devig_1x2(o[0], o[1], o[2], method="shin")
        logger.info("Loaded sharp 1X2 for %d matches", len(self._sharp_1x2))
        return self

    def _blended_1x2(self, key: tuple[str, str]) -> tuple[float, float, float]:
        """1X2 used for reconstruction. blend_weight<=0 or no sharp => pure kicktipp."""
        kt = self._probs_1x2[key]
        sharp = self._sharp_1x2.get(key)
        if self.blend_weight <= 0.0 or sharp is None:
            return kt
        w = self.blend_weight
        b = [w * s + (1.0 - w) * k for s, k in zip(sharp, kt)]
        tot = sum(b)
        return tuple(x / tot for x in b) if tot > 0 else kt

    # -- ScorePredictor interface -------------------------------------

    def fit(self, matches: pd.DataFrame, **kwargs) -> KicktippOddsModel:
        """No-op: odds model doesn't learn from match results."""
        return self

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        match_date: date | str | None = None,
        neutral: bool = False,
        max_goals: int | None = None,
    ) -> np.ndarray:
        if max_goals is None:
            max_goals = self.max_goals

        key = (home_team, away_team)
        if key not in self._probs_1x2:
            raise KeyError(
                f"No kicktipp odds for {home_team} vs {away_team}"
            )

        p_h, p_d, p_a = self._blended_1x2(key)
        p_over = self._p_over.get(key)  # None if unavailable

        if self.require_ou and p_over is None:
            raise ValueError(
                f"O/U data required but missing for {home_team} vs {away_team}. "
                f"Pass require_ou=False to allow 1X2-only reconstruction."
            )

        return reconstruct_matrix(
            p_h, p_d, p_a,
            p_over_2_5=p_over,
            rho=self.rho,
            max_goals=max_goals,
        )


    def predict_lambdas(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """Return reconstructed (λ_home, λ_away) for a match.

        Useful for downstream Monte-Carlo simulation (e.g., group stage).
        """
        key = (home_team, away_team)
        if key not in self._probs_1x2:
            raise KeyError(f"No kicktipp odds for {home_team} vs {away_team}")

        p_h, p_d, p_a = self._blended_1x2(key)
        p_over = self._p_over.get(key)

        from src.odds.reconstruct import reconstruct_lambdas
        lam_h, lam_a, _ = reconstruct_lambdas(
            p_h, p_d, p_a, p_over, self.rho, self.max_goals,
        )
        return lam_h, lam_a


class EnsembleModel(ScorePredictor):
    """Weighted average of multiple score probability matrices.

    Combines matrices from different models (e.g., Dixon-Coles + odds)
    with configurable weights.
    """

    def __init__(
        self,
        models: list[ScorePredictor],
        weights: list[float] | None = None,
        max_goals: int = 8,
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.max_goals = max_goals

        if len(self.weights) != len(self.models):
            raise ValueError("weights must have same length as models")
        # Normalise
        w_sum = sum(self.weights)
        self.weights = [w / w_sum for w in self.weights]

    @property
    def team_idx_(self) -> dict[str, int]:
        """Union of all models' team indices."""
        all_teams: dict[str, int] = {}
        for m in self.models:
            if hasattr(m, "team_idx_"):
                all_teams.update(m.team_idx_)
        return all_teams

    def fit(self, matches: pd.DataFrame, **kwargs) -> EnsembleModel:
        """Fit all underlying models."""
        for model in self.models:
            model.fit(matches, **kwargs)
        return self

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        match_date: date | str | None = None,
        neutral: bool = False,
        max_goals: int | None = None,
    ) -> np.ndarray:
        if max_goals is None:
            max_goals = self.max_goals

        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for model, weight in zip(self.models, self.weights):
            try:
                m = model.predict_score_matrix(
                    home_team, away_team, match_date, neutral, max_goals
                )
                matrix += weight * m
            except (KeyError, RuntimeError) as e:
                logger.warning("Model %s failed for %s vs %s: %s",
                               type(model).__name__, home_team, away_team, e)
                # Redistribute weight to other models
                pass

        # Renormalise
        if matrix.sum() > 0:
            matrix /= matrix.sum()
        return matrix
