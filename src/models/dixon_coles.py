"""Dixon-Coles model for football score prediction.

Extends the double-Poisson model with:
1. A dependence parameter rho (ρ) that corrects the joint probability
   of the four low-scoring outcomes (0-0, 1-0, 0-1, 1-1).
   The independent Poisson systematically underestimates draws.
2. Exponential time-weighting so recent matches count more.

Reference: Dixon & Coles (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market".
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


def _dc_tau(home_goals: int, away_goals: int,
            lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles correction factor tau for a single (i, j) outcome.

    Only modifies probabilities for (0,0), (1,0), (0,1), (1,1).
    For all other scores, tau = 1 (no correction).
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam_h * lam_a * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lam_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam_h * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel(ScorePredictor):
    """Dixon-Coles model with time-weighting."""

    def __init__(self, max_goals: int = 8, half_life_days: int = 1095):
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.teams_: list[str] = []
        self.team_idx_: dict[str, int] = {}
        self.params_: np.ndarray | None = None
        self._n_teams: int = 0

    # -- Parameter layout ----------------------------------------------
    # params = [attack_0, ..., attack_{n-1},
    #           defence_0, ..., defence_{n-1},
    #           mu, home_adv, rho]

    def _unpack(self, params: np.ndarray):
        n = self._n_teams
        attack = params[:n]
        defence = params[n:2*n]
        mu = params[2*n]
        home_adv = params[2*n + 1]
        rho = params[2*n + 2]
        return attack, defence, mu, home_adv, rho

    def _lambda_pair(self, params: np.ndarray,
                     home_idx: int, away_idx: int,
                     neutral: bool = False) -> tuple[float, float]:
        attack, defence, mu, home_adv, _ = self._unpack(params)
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
        attack, defence, mu, home_adv, rho = self._unpack(params)

        ha = home_adv * (~neutral).astype(float)
        lam_h = np.exp(mu + ha + attack[home_idx] - defence[away_idx])
        lam_a = np.exp(mu + attack[away_idx] - defence[home_idx])

        # Clip lambdas for numerical safety
        lam_h = np.clip(lam_h, 1e-6, 20.0)
        lam_a = np.clip(lam_a, 1e-6, 20.0)

        # Poisson log-probabilities
        ll_h = home_goals * np.log(lam_h) - lam_h
        ll_a = away_goals * np.log(lam_a) - lam_a

        # Dixon-Coles tau correction (vectorised)
        tau = np.ones(len(home_goals))
        m00 = (home_goals == 0) & (away_goals == 0)
        m10 = (home_goals == 1) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m11 = (home_goals == 1) & (away_goals == 1)

        tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
        tau[m10] = 1.0 + lam_a[m10] * rho
        tau[m01] = 1.0 + lam_h[m01] * rho
        tau[m11] = 1.0 - rho

        # Ensure tau > 0 for log
        tau = np.clip(tau, 1e-10, None)

        nll = -np.sum(weights * (ll_h + ll_a + np.log(tau)))
        return nll

    def _compute_time_weights(self, dates: pd.Series,
                              reference_date: pd.Timestamp) -> np.ndarray:
        """Exponential decay weights: w(t) = exp(-ln2 * days_ago / half_life)."""
        days_ago = (reference_date - dates).dt.days.values.astype(float)
        decay = np.log(2) / self.half_life_days
        return np.exp(-decay * days_ago)

    def fit(self, matches: pd.DataFrame, **kwargs) -> DixonColesModel:
        """Fit Dixon-Coles model parameters via MLE with time-weighting.

        Parameters
        ----------
        matches : DataFrame with date, home_team, away_team,
                  home_score, away_score, and optionally neutral.
        """
        matches = matches.copy()
        matches["date"] = pd.to_datetime(matches["date"])

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

        # Time weights
        ref_date = matches["date"].max()
        weights = self._compute_time_weights(matches["date"], ref_date)
        logger.info("Time weights: min=%.4f, median=%.4f, max=%.4f",
                    weights.min(), np.median(weights), weights.max())

        n = self._n_teams
        # Initial params
        x0 = np.zeros(2 * n + 3)
        x0[2*n] = 0.3       # mu
        x0[2*n + 1] = 0.25  # home_adv
        x0[2*n + 2] = -0.05 # rho (typically small negative)

        # Identifiability constraints
        constraints = [
            {"type": "eq", "fun": lambda p: np.sum(p[:n])},
            {"type": "eq", "fun": lambda p: np.sum(p[n:2*n])},
        ]

        # Bounds: rho should be in a sensible range
        bounds = [(None, None)] * (2*n + 2) + [(-0.5, 0.5)]

        result = optimize.minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights, neutral),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning("Optimisation did not fully converge: %s",
                           result.message)

        self.params_ = result.x
        attack, defence, mu, home_adv, rho = self._unpack(self.params_)
        logger.info("Fitted DixonColesModel: mu=%.3f, home_adv=%.3f, "
                    "rho=%.4f, %d teams", mu, home_adv, rho, n)
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
        if self.params_ is None:
            raise RuntimeError("Model not fitted")
        if max_goals is None:
            max_goals = self.max_goals

        lam_h, lam_a = self.predict_lambda(home_team, away_team,
                                           match_date, neutral)
        _, _, _, _, rho = self._unpack(self.params_)

        goals = np.arange(max_goals + 1)
        prob_h = poisson.pmf(goals, lam_h)
        prob_a = poisson.pmf(goals, lam_a)

        # Start with independent Poisson
        matrix = np.outer(prob_h, prob_a)

        # Apply Dixon-Coles correction to (0,0), (1,0), (0,1), (1,1)
        matrix[0, 0] *= _dc_tau(0, 0, lam_h, lam_a, rho)
        matrix[1, 0] *= _dc_tau(1, 0, lam_h, lam_a, rho)
        matrix[0, 1] *= _dc_tau(0, 1, lam_h, lam_a, rho)
        matrix[1, 1] *= _dc_tau(1, 1, lam_h, lam_a, rho)

        # Renormalise
        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()
        return matrix
