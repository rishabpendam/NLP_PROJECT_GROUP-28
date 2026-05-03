"""
core/action_parser.py
Parse raw LLM text (Thought / Talk / Action) into a structured AgentTurn.
Handles messy formatting gracefully — the LLM does not always comply.
"""
from __future__ import annotations
import re
from core.schema import AgentTurn, Offer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_agent_turn(raw_text: str, role: str, turn_number: int) -> AgentTurn:
    """
    Convert raw LLM output into a structured AgentTurn.

    Expected format:
        THOUGHT: <reasoning>
        TALK: <message to opponent>
        ACTION: OFFER price=120 delivery_days=7 warranty_months=12
                | ASK <question>
                | DEAL
                | QUIT
    """
    thought = _extract_section(raw_text, "THOUGHT")
    talk = _extract_section(raw_text, "TALK")
    action_line = _extract_section(raw_text, "ACTION").strip()

    action, payload, offer = _parse_action(action_line)

    return AgentTurn(
        role=role,
        turn_number=turn_number,
        raw_text=raw_text,
        thought=thought,
        talk=talk,
        action=action,
        action_payload=payload,
        offer=offer,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_section(text: str, label: str) -> str:
    """
    Extract the content following a label like 'THOUGHT:' up to the next label or end.
    Case-insensitive. Returns empty string if the label is not found.
    """
    pattern = rf"(?i){label}\s*:\s*(.*?)(?=\n(?:THOUGHT|TALK|ACTION)\s*:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: everything after the label on the same line
    pattern2 = rf"(?i){label}\s*:\s*(.*)"
    match2 = re.search(pattern2, text)
    if match2:
        return match2.group(1).strip()
    return ""


def _parse_action(action_line: str):
    """
    Parse the ACTION line and return (action_name, payload_dict, offer_or_None).
    Defaults to QUIT if nothing recognisable is found.
    """
    action_line = action_line.strip().upper()

    # ---- DEAL ----
    if action_line.startswith("DEAL"):
        return "DEAL", {}, None

    # ---- QUIT ----
    if action_line.startswith("QUIT"):
        return "QUIT", {}, None

    # ---- ASK ----
    ask_match = re.match(r"ASK\s+(.*)", action_line, re.IGNORECASE | re.DOTALL)
    if ask_match:
        question = ask_match.group(1).strip()
        return "ASK", {"question": question}, None

    # ---- OFFER ----
    if "OFFER" in action_line:
        price = _extract_float(action_line, r"PRICE\s*=\s*([0-9.]+)")
        delivery = _extract_float(action_line, r"DELIVERY_DAYS\s*=\s*([0-9.]+)")
        warranty = _extract_float(action_line, r"WARRANTY_MONTHS\s*=\s*([0-9.]+)")

        if price is not None and delivery is not None and warranty is not None:
            offer = Offer(price=price, delivery_days=delivery, warranty_months=warranty)
            payload = {
                "price": price,
                "delivery_days": delivery,
                "warranty_months": warranty,
            }
            return "OFFER", payload, offer

        # Partial offer — treat as QUIT (engine will warn)
        return "QUIT", {}, None

    # Fallback
    return "QUIT", {}, None


def _extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
