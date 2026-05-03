"""
core/dialogue.py
Dialogue history manager.
Keeps track of all turns and provides formatted context strings for each agent.
"""
from __future__ import annotations
from typing import List, Optional
from core.schema import AgentTurn, Offer


class DialogueHistory:
    """
    Stores all AgentTurns in order and renders history for prompt injection.
    """

    def __init__(self):
        self._turns: List[AgentTurn] = []

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_turn(self, turn: AgentTurn) -> None:
        self._turns.append(turn)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_turns(self) -> List[AgentTurn]:
        return list(self._turns)

    def get_last_offer(self, role: Optional[str] = None) -> Optional[Offer]:
        """
        Return the most recent offer made by `role` (or by anyone if role is None).
        Returns None if no offer has been made yet.
        """
        for turn in reversed(self._turns):
            if turn.action == "OFFER" and turn.offer is not None:
                if role is None or turn.role == role:
                    return turn.offer
        return None

    def get_last_complete_offer(self) -> Optional[Offer]:
        """
        Return the most recent offer that has all three fields populated.
        Used by the engine when processing a DEAL action.
        """
        for turn in reversed(self._turns):
            if turn.action == "OFFER" and turn.offer is not None:
                o = turn.offer
                if o.price > 0 and o.delivery_days > 0 and o.warranty_months > 0:
                    return o
        return None

    def format_for_agent(self, viewer_role: str) -> str:
        """
        Return a human-readable dialogue history string for injection into a prompt.
        Both THOUGHT sections are hidden (internal reasoning is private).
        Each turn is labelled as YOU or OPPONENT so the agent never loses track of its role.
        """
        lines = []
        for t in self._turns:
            if t.role == viewer_role:
                label = f"YOU ({t.role.upper()})"
            else:
                label = f"OPPONENT ({t.role.upper()})"
            lines.append(f"[Turn {t.turn_number}] {label}:")
            if t.talk:
                lines.append(f"  TALK: {t.talk}")
            if t.action == "OFFER" and t.offer is not None:
                o = t.offer
                lines.append(
                    f"  ACTION: OFFER price={o.price:.2f} "
                    f"delivery_days={o.delivery_days:.0f} "
                    f"warranty_months={o.warranty_months:.0f}"
                )
            elif t.action in ("DEAL", "QUIT"):
                lines.append(f"  ACTION: {t.action}")
            elif t.action == "ASK":
                q = t.action_payload.get("question", "")
                lines.append(f"  ACTION: ASK {q}")
        return "\n".join(lines) if lines else "(No dialogue yet)"

    def count_turns(self) -> int:
        return len(self._turns)

    def count_offers(self, role: str) -> int:
        return sum(1 for t in self._turns if t.role == role and t.action == "OFFER")

    def count_backtracks(self, role: str) -> int:
        """
        Count how many times `role` reversed the direction of their price concessions.
        A backtrack = going from lowering to raising (seller) or raising to lowering (buyer).
        """
        offers = [t.offer for t in self._turns
                  if t.role == role and t.action == "OFFER" and t.offer is not None]
        if len(offers) < 3:
            return 0

        backtracks = 0
        for i in range(2, len(offers)):
            prev_delta = offers[i - 1].price - offers[i - 2].price
            curr_delta = offers[i].price - offers[i - 1].price
            if prev_delta * curr_delta < 0:   # sign flip = reversal
                backtracks += 1
        return backtracks
