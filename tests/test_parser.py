"""
tests/test_parser.py
Unit tests for core/action_parser.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.action_parser import parse_agent_turn


def _parse(raw, role="buyer", turn=0):
    return parse_agent_turn(raw, role, turn)


# ---------------------------------------------------------------------------
# OFFER parsing
# ---------------------------------------------------------------------------

def test_offer_all_fields():
    raw = (
        "THOUGHT: Let me offer a reasonable price.\n"
        "TALK: How about this deal?\n"
        "ACTION: OFFER price=120 delivery_days=7 warranty_months=12"
    )
    t = _parse(raw)
    assert t.action == "OFFER"
    assert t.offer is not None
    assert t.offer.price == 120.0
    assert t.offer.delivery_days == 7.0
    assert t.offer.warranty_months == 12.0


def test_offer_decimal_values():
    raw = (
        "THOUGHT: x\n"
        "TALK: x\n"
        "ACTION: OFFER price=134.50 delivery_days=10 warranty_months=6"
    )
    t = _parse(raw)
    assert t.action == "OFFER"
    assert t.offer.price == 134.50


def test_offer_missing_field_becomes_quit():
    raw = (
        "THOUGHT: x\nTALK: x\n"
        "ACTION: OFFER price=120 delivery_days=7"   # missing warranty_months
    )
    t = _parse(raw)
    assert t.action == "QUIT"


# ---------------------------------------------------------------------------
# DEAL / QUIT
# ---------------------------------------------------------------------------

def test_deal():
    raw = "THOUGHT: x\nTALK: Agreed!\nACTION: DEAL"
    t = _parse(raw)
    assert t.action == "DEAL"
    assert t.offer is None


def test_quit():
    raw = "THOUGHT: x\nTALK: Walking away.\nACTION: QUIT"
    t = _parse(raw)
    assert t.action == "QUIT"


# ---------------------------------------------------------------------------
# ASK
# ---------------------------------------------------------------------------

def test_ask():
    raw = (
        "THOUGHT: I want to know their floor.\n"
        "TALK: Can you tell me your minimum?\n"
        "ACTION: ASK What is your minimum acceptable price?"
    )
    t = _parse(raw)
    assert t.action == "ASK"
    assert "minimum acceptable price" in t.action_payload.get("question", "").lower()


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def test_thought_extracted():
    raw = "THOUGHT: This is my plan.\nTALK: Hi\nACTION: QUIT"
    t = _parse(raw)
    assert "plan" in t.thought.lower()


def test_talk_extracted():
    raw = "THOUGHT: x\nTALK: Hello seller!\nACTION: QUIT"
    t = _parse(raw)
    assert "hello seller" in t.talk.lower()


def test_role_preserved():
    raw = "THOUGHT: x\nTALK: x\nACTION: DEAL"
    t = parse_agent_turn(raw, role="seller", turn_number=3)
    assert t.role == "seller"
    assert t.turn_number == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string_defaults_quit():
    t = _parse("")
    assert t.action == "QUIT"


def test_garbage_defaults_quit():
    t = _parse("I don't know what to say.")
    assert t.action == "QUIT"


def test_case_insensitive_action():
    raw = "thought: x\ntalk: x\naction: deal"
    t = _parse(raw)
    assert t.action == "DEAL"
