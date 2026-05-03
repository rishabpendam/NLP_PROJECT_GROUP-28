"""
agents/prompts.py
Shared prompt-building utilities.

All LLM agents (NoRep and Rep, regardless of provider) use these builders so
prompt wording is identical across conditions — the only difference is whether
a reputation block is appended.
"""
from __future__ import annotations
from typing import Optional
from core.schema import Scenario
from core.dialogue import DialogueHistory


# ---------------------------------------------------------------------------
# System prompts (Appendix A of paper)
# ---------------------------------------------------------------------------

BUYER_SYSTEM = """You are a buyer in a multi-issue negotiation.
Your goal is to obtain the best possible deal while staying within your constraints.

You negotiate over three issues: price (lower is better), delivery_days (fewer is better), warranty_months (more is better).

Negotiation Rules:
- NEVER exceed your willingness to pay (WTP).
- Start with a low offer — roughly 55-65% of your WTP — to leave room for negotiation.
- Gradually increase your offer when the seller makes concessions.
- Use ASK at most once per negotiation to ask the seller about their constraints.
- After 3 or more offers have been exchanged, start converging toward a deal.
- IMPORTANT: If the seller's latest offer is within 5% of your own last offer, play DEAL immediately — do NOT keep counter-offering.
- Accept the deal if it satisfies your constraints and is below your WTP.

Response Format — you MUST use exactly this structure every turn:
THOUGHT: <your internal reasoning — never shown to the opponent>
TALK: <natural language message to the seller>
ACTION: <one of: OFFER price=X delivery_days=Y warranty_months=Z | ASK <question> | DEAL | QUIT>

OFFER format requirement: every OFFER must include ALL THREE fields:
  ACTION: OFFER price=<number> delivery_days=<number> warranty_months=<number>

Do not include any text after ACTION."""

SELLER_SYSTEM = """You are a seller in a multi-issue negotiation.
Your goal is to reach a profitable agreement while staying above your minimum price floor.

You negotiate over three issues: price (higher is better for you), delivery_days (longer is acceptable), warranty_months (shorter is preferable for you).

Negotiation Rules:
- NEVER go below your minimum acceptable price.
- Your FIRST offer MUST be at or above the Initial Offer Price shown in your private information — this is your opening anchor.
- Keep delivery_days between 3 and 30. Do NOT propose 90, 120, or 180 day delivery.
- Gradually decrease your price when the buyer makes concessions.
- After 3 or more offers have been exchanged, start converging toward a deal.
- IMPORTANT: If the buyer's latest offer is within 5% of your own last offer, play DEAL immediately — do NOT keep counter-offering.
- Accept the deal if the buyer's offer covers your floor.

Response Format — you MUST use exactly this structure every turn:
THOUGHT: <your internal reasoning — never shown to the opponent>
TALK: <natural language message to the buyer>
ACTION: <one of: OFFER price=X delivery_days=Y warranty_months=Z | ASK <question> | DEAL | QUIT>

OFFER format requirement: every OFFER must include ALL THREE fields:
  ACTION: OFFER price=<number> delivery_days=<number> warranty_months=<number>

Do not include any text after ACTION."""


# ---------------------------------------------------------------------------
# User-turn message builders
# ---------------------------------------------------------------------------

def build_buyer_user_message(
    scenario: Scenario,
    history: DialogueHistory,
    rep_score: Optional[float],
    opponent_answer: Optional[str],
    include_rep_reasoning: bool = False,
) -> str:
    """Build the user-turn message injected into the buyer's prompt."""
    bp = scenario.buyer
    suggested_first_offer = round(bp.willingness_to_pay * 0.60, 2)
    is_first_turn = not history.get_last_offer(role="buyer")
    lines = [
        "=== YOUR ROLE ===",
        "You are the BUYER. Do NOT copy the seller's messages. Generate your OWN response.",
        "",
        "=== YOUR PRIVATE INFORMATION ===",
        f"Willingness to Pay (WTP): ${bp.willingness_to_pay:.2f}  ← NEVER exceed this",
        f"Suggested first offer: ${suggested_first_offer:.2f}  (≈60% of WTP — start here to leave negotiation room)",
        f"Preference weights: price={bp.weight_price:.2f}, delivery={bp.weight_delivery:.2f}, warranty={bp.weight_warranty:.2f}",
        "",
        "=== UNCERTAINTY ===",
        f"Noise level η = {scenario.noise:.1f}. The seller's answers may be partially incorrect or misleading.",
        "Do not fully trust all information you receive.",
        "",
    ]
    if is_first_turn:
        lines += [
            f"⚠ This is your FIRST turn. Open with an OFFER around ${suggested_first_offer:.2f} (or lower) — never at or above WTP.",
            "",
        ]

    # Standing offer
    last_seller_offer = history.get_last_offer(role="seller")
    if last_seller_offer:
        o = last_seller_offer
        lines += [
            "=== SELLER'S STANDING OFFER ===",
            f"price=${o.price:.2f}, delivery={o.delivery_days:.0f} days, warranty={o.warranty_months:.0f} months",
            "",
        ]

    # Opponent answer
    if opponent_answer:
        lines += [
            "=== SELLER'S ANSWER TO YOUR QUESTION ===",
            opponent_answer,
            "",
        ]

    # Reputation block (LLM-Rep condition only)
    if rep_score is not None:
        lines += [
            "=== OPPONENT REPUTATION SCORE ===",
            f"Seller reputation: {rep_score:.2f} (0 = untrustworthy, 1 = fully trustworthy)",
        ]
        if include_rep_reasoning:
            if rep_score < 0.35:
                lines.append("⚠ Low reputation — be cautious; seller may be deceptive or inconsistent.")
            elif rep_score < 0.65:
                lines.append("ℹ Moderate reputation — verify important claims before accepting.")
            else:
                lines.append("✓ High reputation — seller has been honest and consistent in the past.")
        lines.append("")

    # Dialogue history
    lines += [
        "=== DIALOGUE HISTORY ===",
        history.format_for_agent("buyer"),
        "",
        "=== YOUR TURN ===",
        "Respond with THOUGHT, TALK, and ACTION.",
        "Remember: OFFER must include ALL THREE fields: price, delivery_days, warranty_months.",
    ]

    return "\n".join(lines)


def build_seller_user_message(
    scenario: Scenario,
    history: DialogueHistory,
    rep_score: Optional[float],
    opponent_answer: Optional[str],
    include_rep_reasoning: bool = False,
) -> str:
    """Build the user-turn message injected into the seller's prompt."""
    sp = scenario.seller
    is_first_turn = not history.get_last_offer(role="seller")
    lines = [
        "=== YOUR ROLE ===",
        "You are the SELLER. Do NOT copy the buyer's messages. Generate your OWN response.",
        "",
        "=== YOUR PRIVATE INFORMATION ===",
        f"Production cost: ${sp.production_cost:.2f}",
        f"Minimum acceptable price: ${sp.min_acceptable_price:.2f}  ← NEVER go below this",
        f"Initial Offer Price: ${scenario.initial_offer_price:.2f}  ← your FIRST offer MUST be at or above this",
        f"Preference weights: price={sp.weight_price:.2f}, delivery={sp.weight_delivery:.2f}, warranty={sp.weight_warranty:.2f}",
        "",
        "=== UNCERTAINTY ===",
        f"Noise level η = {scenario.noise:.1f}. The buyer's answers may be partially incorrect or misleading.",
        "",
    ]
    if is_first_turn:
        lines += [
            f"⚠ This is your FIRST turn. Start with an OFFER at ${scenario.initial_offer_price:.2f} or higher.",
            "",
        ]

    # Standing offer
    last_buyer_offer = history.get_last_offer(role="buyer")
    if last_buyer_offer:
        o = last_buyer_offer
        lines += [
            "=== BUYER'S STANDING OFFER ===",
            f"price=${o.price:.2f}, delivery={o.delivery_days:.0f} days, warranty={o.warranty_months:.0f} months",
            "",
        ]

    # Opponent answer
    if opponent_answer:
        lines += [
            "=== BUYER'S ANSWER TO YOUR QUESTION ===",
            opponent_answer,
            "",
        ]

    # Reputation block
    if rep_score is not None:
        lines += [
            "=== OPPONENT REPUTATION SCORE ===",
            f"Buyer reputation: {rep_score:.2f} (0 = untrustworthy, 1 = fully trustworthy)",
        ]
        if include_rep_reasoning:
            if rep_score < 0.35:
                lines.append("⚠ Low reputation — buyer may be deceptive; negotiate more cautiously.")
            elif rep_score < 0.65:
                lines.append("ℹ Moderate reputation — proceed with standard caution.")
            else:
                lines.append("✓ High reputation — buyer has been reliable; consider conceding faster.")
        lines.append("")

    lines += [
        "=== DIALOGUE HISTORY ===",
        history.format_for_agent("seller"),
        "",
        "=== YOUR TURN ===",
        "Respond with THOUGHT, TALK, and ACTION.",
        "Remember: OFFER must include ALL THREE fields: price, delivery_days, warranty_months.",
        "Keep delivery_days between 3 and 30.",
    ]

    return "\n".join(lines)
