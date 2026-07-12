"""Independent double-Poisson model.

Each team's goals are modelled as independent Poisson random variables:
    home_goals ~ Poisson(lambda_home)
    away_goals ~ Poisson(lambda_away)

where:
    log(lambda_home) = mu + home_adv + attack_home - defence_away
    log(lambda_away) = mu              + attack_away - defence_home

Parameters (attack_i, defence_i for each team, plus mu and home_adv)
are estimated by maximising the Poisson log-likelihood over historical
matches, optionally with exponential time-weighting.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

from src.models.base import ScorePredictor

logger = logging.getLogger(__name__)


class DoublePoissonModel(ScorePredictor):
    """Independent double-Poisson model for football scores."""

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.teams_: list[str] = []
        self.team_idx_: dict[str, int] = {}
        self.params_: np.ndarray | None = None
        self._n_teams: int = 0

    # -- Parameter layout ----------------------------------------------
    # params = [attack_0, ..., attack_{n-1},
    #           defence_0, ..., defence_{n-1},
    #           mu, home_adv]

    def _unpack(self, params: np.ndarray):
        n = self._n_teams
        attack = params[:n]
        defence = params[n:2*n]
        mu = params[2*n]
        home_adv = params[2*n + 1]
        return attack, defence, mu, home_adv

    def _lambda_pair(self, params: np.ndarray,
                     home_idx: int, away_idx: int,
                     neutral: bool = False) -> tuple[float, float]:
        attack, defence, mu, home_adv = self._unpack(params)
        ha = 0.0 if neutral else home_adv
        lam_h = np.exp(mu + ha + attack[home_idx] - defence[away_idx])
        lam_a = np.exp(mu + attack[away_idx] - defence[home_idx])
        return float(lam_h), float(lam_a)

    def _neg_log_likelihood(self, params: np.ndarray,
                            home_idx: np.ndarray,
                            away_idx: np.ndarray,
                            home_goals: np.ndarray,
                            away_goals: np.ndarray,
                            weights: np.ndarray,
                            neutral: np.ndarray) -> float:
        attack, defence, mu, home_adv = self._unpack(params)

        ha = home_adv * (~neutral).astype(float)
        lam_h = np.exp(mu + ha + attack[home_idx] - defence[away_idx])
        lam_a = np.exp(mu + attack[away_idx] - defence[home_idx])

        # Poisson log-likelihood
        ll_h = home_goals * np.log(lam_h + 1e-10) - lam_h
        ll_a = away_goals * np.log(lam_a + 1e-10) - lam_a
        return -np.sum(weights * (ll_h + ll_a))

    def fit(self, matches: pd.DataFrame, **kwargs) -> DoublePoissonModel:
        """Fit model parameters via MLE.

        Parameters
        ----------
        matches : DataFrame with columns date, home_team, away_team,
                  home_score, away_score, and optionally neutral.
        **kwargs : optional 'weights' array (same length as matches).
        """
        teams = sorted(
            set(matches["home_team"].unique()) | set(matches["away_team"].unique())
        )
        self.teams_ = teams
        self.team_idx_ = {t: i for i, t in enumerate(teams)}
        self._n_teams = len(teams)

        home_idx = matches["home_team"].map(self.team_idx_).values
        away_idx = matches["away_team"].map(self.team_idx_).values
        home_goals = matches["home_score"].values.astype(float)
        away_goals = matches["away_score"].values.astype(float)
        neutral = matches.get("neutral", pd.Series(False, index=matches.index)).values.astype(bool)

        weights = kwargs.get("weights", np.ones(len(matches)))

        n = self._n_teams
        # Initial params: attack=0, defence=0, mu=0.3, home_adv=0.25
        x0 = np.zeros(2 * n + 2)
        x0[2*n] = 0.3       # mu
        x0[2*n + 1] = 0.25  # home_adv

        # Constraint: sum of attack params = 0 (identifiability)
        constraints = [
            {"type": "eq", "fun": lambda p: np.sum(p[:n])},
            {"type": "eq", "fun": lambda p: np.sum(p[n:2*n])},
        ]

        result = optimize.minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights, neutral),
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning("Optimisation did not converge: %s", result.message)

        self.params_ = result.x
        attack, defence, mu, home_adv = self._unpack(self.params_)
        logger.info("Fitted DoublePoissonModel: mu=%.3f, home_adv=%.3f, "
                    "%d teams", mu, home_adv, n)
        return self

    def predict_lambda(
        self,
        home_team: str,
        away_team: str,
        match_date: date | str | None = None,
        neutral: bool = False,
    ) -> tuple[float, float]:
        if self.params_ is None:
            raise RuntimeError("Model not fitted")
        hi = self.team_idx_[home_team]
        ai = self.team_idx_[away_team]
        return self._lambda_pair(self.params_, hi, ai, neutral)

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
        lam_h, lam_a = self.predict_lambda(home_team, away_team,
                                           match_date, neutral)
        goals = np.arange(max_goals + 1)
        prob_h = poisson.pmf(goals, lam_h)
        prob_a = poisson.pmf(goals, lam_a)
        # Outer product gives joint probability (independence assumption)
        matrix = np.outer(prob_h, prob_a)
        # Normalise to account for truncation
        matrix /= matrix.sum()
        return matrix
