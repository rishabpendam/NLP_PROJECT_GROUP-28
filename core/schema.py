"""
core/schema.py
Shared dataclasses used across the entire negotiation framework.
All modules should import from here — do not duplicate definitions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# Negotiation scenario (private info for each agent + environment params)
# ---------------------------------------------------------------------------

@dataclass
class BuyerParams:
    """Private parameters for the buyer agent."""
    willingness_to_pay: float          # Upper bound on price the buyer will accept
    weight_price: float                # Importance weight for price      (sums to 1)
    weight_delivery: float             # Importance weight for delivery time
    weight_warranty: float             # Importance weight for warranty

    def __post_init__(self):
        total = self.weight_price + self.weight_delivery + self.weight_warranty
        assert abs(total - 1.0) < 1e-4, f"Buyer weights must sum to 1, got {total}"


@dataclass
class SellerParams:
    """Private parameters for the seller agent."""
    production_cost: float             # Internal cost (never disclosed)
    min_acceptable_price: float        # Hard floor — will QUIT below this
    weight_price: float                # Importance weight for price      (sums to 1)
    weight_delivery: float             # Importance weight for delivery time
    weight_warranty: float             # Importance weight for warranty

    def __post_init__(self):
        total = self.weight_price + self.weight_delivery + self.weight_warranty
        assert abs(total - 1.0) < 1e-4, f"Seller weights must sum to 1, got {total}"


@dataclass
class Scenario:
    """
    A single negotiation scenario.
    Contains both agents' private parameters plus environment settings.
    """
    scenario_id: str
    buyer: BuyerParams
    seller: SellerParams

    # Environment parameters
    initial_offer_price: float         # Seller's opening price
    max_turns: int = 16                # Maximum dialogue turns before timeout
    noise: float = 0.2                 # η — probability an answer is misleading
    rep_visible: bool = False          # Whether reputation scores are shown to agents

    # Typical market ranges (used for utility normalisation)
    price_min: float = 50.0
    price_max: float = 200.0
    delivery_min: float = 1.0          # days
    delivery_max: float = 30.0
    warranty_min: float = 1.0          # months
    warranty_max: float = 24.0


# ---------------------------------------------------------------------------
# Per-turn agent output
# ---------------------------------------------------------------------------

@dataclass
class Offer:
    """A concrete offer with all three negotiation dimensions."""
    price: float
    delivery_days: float
    warranty_months: float


@dataclass
class AgentTurn:
    """
    Structured output from one agent on one turn.
    Produced by action_parser.py from raw LLM text.
    """
    role: str                          # "buyer" or "seller"
    turn_number: int
    raw_text: str                      # Original LLM output (for debugging)
    thought: str = ""
    talk: str = ""
    action: str = "QUIT"               # OFFER | ASK | DEAL | QUIT
    action_payload: Dict[str, Any] = field(default_factory=dict)
    # Populated only when action == "OFFER"
    offer: Optional[Offer] = None


# ---------------------------------------------------------------------------
# Episode-level result
# ---------------------------------------------------------------------------

@dataclass
class StructuredAnswer:
    """Record of an ASK/answer exchange for reputation scoring."""
    turn_number: int
    answering_agent: str               # "buyer" or "seller"
    question: str
    answer_given: str
    ground_truth: str
    field: str                         # e.g. "min_price", "wtp", "production_cost"
    is_truthful: bool = False


@dataclass
class EpisodeResult:
    """Full result of one negotiation episode."""
    scenario_id: str
    outcome: str                       # "deal" | "quit" | "timeout"
    final_deal: Optional[Offer] = None
    turns: List[AgentTurn] = field(default_factory=list)
    structured_answers: List[StructuredAnswer] = field(default_factory=list)
    buyer_reputation_score: float = 0.5
    seller_reputation_score: float = 0.5
    buyer_utility: float = 0.0
    deal: bool = False
