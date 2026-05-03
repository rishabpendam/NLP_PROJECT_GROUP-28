"""
agents/base_agent.py
Abstract base class that all buyer/seller agents must implement.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from core.schema import Scenario
from core.dialogue import DialogueHistory


class BaseAgent(ABC):
    """
    Abstract negotiation agent.

    Subclasses must implement `act()`.  The return value is a raw string in the
    Thought–Talk–Action format which engine.py's parser will interpret.
    """

    def __init__(self, role: str):
        assert role in ("buyer", "seller"), f"role must be 'buyer' or 'seller', got {role!r}"
        self.role = role

    @abstractmethod
    def act(
        self,
        scenario: Scenario,
        history: DialogueHistory,
        rep_score: Optional[float],
        opponent_answer: Optional[str],
    ) -> str:
        """
        Observe the current state and return a raw Thought/Talk/Action string.

        Parameters
        ----------
        scenario        : full scenario (including this agent's private parameters)
        history         : dialogue history visible to both agents
        rep_score       : opponent's reputation score ∈ [0,1] or None if hidden
        opponent_answer : answer to the last ASK question (if any)
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(role={self.role!r})"
