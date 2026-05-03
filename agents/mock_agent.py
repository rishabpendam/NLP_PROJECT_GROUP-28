"""
agents/mock_agent.py
Scripted (non-LLM) agents for smoke tests and pipeline validation.
No API key required.

MockBuyer  : starts low, increments by $10 each turn, accepts if price ≤ WTP.
MockSeller : starts at initial_offer, decrements by $8 each turn, floors at min_price.
"""
from __future__ import annotations
import re
from typing import Optional
from agents.base_agent import BaseAgent
from core.schema import Scenario
from core.dialogue import DialogueHistory


class MockBuyer(BaseAgent):
    """
    Scripted buyer for testing.
    Starts at (WTP - 40), increases offer by $10 each turn.
    Accepts the deal if the seller's standing price ≤ WTP.
    """

    def __init__(self):
        super().__init__(role="buyer")

    def act(
        self,
        scenario: Scenario,
        history: DialogueHistory,
        rep_score: Optional[float],
        opponent_answer: Optional[str],
    ) -> str:
        wtp = scenario.buyer.willingness_to_pay
        n_offers = history.count_offers("buyer")
        my_price = min(wtp, (wtp - 40) + n_offers * 10)

        # Check if seller's last offer is acceptable
        seller_offer = history.get_last_offer(role="seller")
        if seller_offer is not None and seller_offer.price <= wtp:
            return (
                "THOUGHT: Seller's price is within my budget. Accepting.\n"
                f"TALK: That works for me. Deal!\n"
                "ACTION: DEAL"
            )

        my_price = round(my_price, 2)
        return (
            f"THOUGHT: I'll offer ${my_price:.2f}.\n"
            f"TALK: I can offer ${my_price:.2f} with 7 day delivery and 12 month warranty.\n"
            f"ACTION: OFFER price={my_price} delivery_days=7 warranty_months=12"
        )


class MockSeller(BaseAgent):
    """
    Scripted seller for testing.
    Starts at initial_offer_price, decreases by $8 each turn, floors at min_acceptable_price.
    """

    def __init__(self):
        super().__init__(role="seller")

    def act(
        self,
        scenario: Scenario,
        history: DialogueHistory,
        rep_score: Optional[float],
        opponent_answer: Optional[str],
    ) -> str:
        floor = scenario.seller.min_acceptable_price
        n_offers = history.count_offers("seller")
        my_price = max(floor, scenario.initial_offer_price - n_offers * 8)
        my_price = round(my_price, 2)

        # Check if buyer's last offer is above our floor
        buyer_offer = history.get_last_offer(role="buyer")
        if buyer_offer is not None and buyer_offer.price >= floor:
            return (
                "THOUGHT: Buyer's price covers my floor. Accepting.\n"
                f"TALK: Agreed! Let's close at ${buyer_offer.price:.2f}.\n"
                "ACTION: DEAL"
            )

        return (
            f"THOUGHT: I'll offer ${my_price:.2f}.\n"
            f"TALK: My price is ${my_price:.2f}, delivery in 7 days, 12 month warranty.\n"
            f"ACTION: OFFER price={my_price} delivery_days=7 warranty_months=12"
        )
