"""
reputation/module.py
Reputation mechanism from the paper.

Two components:
  Truthfulness  T  = N_truthful / N_answers
  Consistency   C  = 1 - N_backtrack / N_moves

Episode reputation:
  Rep = α · T + (1 − α) · C

Running reputation (exponential smoothing):
  R_{t+1} = (1 − β) · R_t + β · Rep

Default hyperparameters (paper Section 5.1):
  α = 0.5  (equal weight to truthfulness and consistency)
  β = 0.2  (slow update — reputation does not shift drastically after one episode)
"""
from __future__ import annotations
from core.schema import EpisodeResult
from core.dialogue import DialogueHistory


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ReputationTracker:
    """
    Maintains a running reputation score for one agent across episodes.

    Usage
    -----
    tracker = ReputationTracker(alpha=0.5, beta=0.2)
    for result in episode_results:
        score = tracker.update(result, role="seller")
    current_score = tracker.score
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.2, initial: float = 0.5):
        """
        Parameters
        ----------
        alpha   : weight of truthfulness vs consistency ∈ [0,1]
        beta    : learning rate for running average ∈ [0,1]
        initial : starting reputation (neutral = 0.5)
        """
        self.alpha = alpha
        self.beta = beta
        self._score = initial

    @property
    def score(self) -> float:
        return round(self._score, 4)

    def update(self, result: EpisodeResult, role: str, history: DialogueHistory) -> float:
        """
        Compute episode-level reputation from an EpisodeResult and update the
        running score.  Returns the new running score.

        Parameters
        ----------
        result  : completed episode
        role    : "buyer" or "seller" — whose reputation to update
        history : dialogue history from the same episode (for consistency calculation)
        """
        T = _compute_truthfulness(result, role)
        C = _compute_consistency(history, role)
        rep_episode = self.alpha * T + (1.0 - self.alpha) * C

        self._score = (1.0 - self.beta) * self._score + self.beta * rep_episode
        return self.score


# ---------------------------------------------------------------------------
# Component calculations (match paper formulae exactly)
# ---------------------------------------------------------------------------

def _compute_truthfulness(result: EpisodeResult, role: str) -> float:
    """
    T = N_truthful / N_answers

    N_answers  : number of questions this agent answered
    N_truthful : answers that matched ground truth within 5 %
    """
    answers = [sa for sa in result.structured_answers if sa.answering_agent == role]
    if not answers:
        return 1.0   # No questions asked → assume truthful (no evidence of lying)

    n_truthful = sum(1 for sa in answers if sa.is_truthful)
    return n_truthful / len(answers)


def _compute_consistency(history: DialogueHistory, role: str) -> float:
    """
    C = 1 - N_backtrack / N_moves

    A backtrack happens when an agent reverses the direction of its price concession.
    """
    n_moves = history.count_offers(role)
    if n_moves < 2:
        return 1.0   # Not enough offers to detect a reversal

    n_backtrack = history.count_backtracks(role)
    denominator = max(n_moves - 1, 1)   # direction changes need at least 2 offers
    return max(0.0, 1.0 - n_backtrack / denominator)


# ---------------------------------------------------------------------------
# Convenience function for one-shot scoring (used in tests)
# ---------------------------------------------------------------------------

def compute_episode_reputation(
    result: EpisodeResult,
    history: DialogueHistory,
    role: str,
    alpha: float = 0.5,
) -> float:
    """Return the raw episode reputation score (before smoothing)."""
    T = _compute_truthfulness(result, role)
    C = _compute_consistency(history, role)
    return alpha * T + (1.0 - alpha) * C
