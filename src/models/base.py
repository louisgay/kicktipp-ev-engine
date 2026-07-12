"""Abstract base class for score prediction models.

All models expose a common interface so the scoring/decision layer
is decoupled from the model implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np


class ScorePredictor(ABC):
    """Interface for score prediction models.

    Subclasses must implement ``fit`` and ``predict_score_matrix``.
    """

    @abstractmethod
    def fit(self, matches: "pd.DataFrame", **kwargs) -> "ScorePredictor":
        """Train the model on historical match data.

        Parameters
        ----------
        matches : DataFrame with at least columns
            date, home_team, away_team, home_score, away_score
        """
        ...

    @abstractmethod
    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        match_date: date | str | None = None,
        neutral: bool = False,
        max_goals: int = 8,
    ) -> np.ndarray:
        """Return a (max_goals+1) x (max_goals+1) matrix of P(home=i, away=j).

        The matrix must sum to ~1.0 (up to truncation error).
        """
        ...

    def predict_lambda(
        self,
        home_team: str,
        away_team: str,
        match_date: date | str | None = None,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) expected goals.

        Optional; not all models expose this directly.
        """
        raise NotImplementedError
