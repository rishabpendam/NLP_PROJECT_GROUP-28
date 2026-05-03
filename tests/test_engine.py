"""
tests/test_engine.py
Unit tests for engine, dialogue history, scenario generator, metrics, and reputation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.schema import Offer, AgentTurn
from core.dialogue import DialogueHistory
from core.engine import NegotiationEngine
from scenario.generator import generate_scenario, make_simple_scenario
from agents.mock_agent import MockBuyer, MockSeller
from reputation.module import ReputationTracker, compute_episode_reputation
from evaluation.metrics import (
    compute_buyer_utility, deal_rate, average_buyer_utility,
    delta_utility, delta_deal_rate,
)


# ---------------------------------------------------------------------------
# DialogueHistory
# ---------------------------------------------------------------------------

def _make_offer_turn(role, turn_num, price, delivery=7, warranty=12):
    offer = Offer(price=price, delivery_days=delivery, warranty_months=warranty)
    return AgentTurn(
        role=role, turn_number=turn_num, raw_text="",
        action="OFFER", offer=offer,
        action_payload={"price": price, "delivery_days": delivery, "warranty_months": warranty},
    )


def test_dialogue_get_last_offer():
    hist = DialogueHistory()
    hist.add_turn(_make_offer_turn("seller", 0, 160.0))
    hist.add_turn(_make_offer_turn("buyer", 1, 110.0))
    hist.add_turn(_make_offer_turn("seller", 2, 140.0))

    assert hist.get_last_offer("seller").price == 140.0
    assert hist.get_last_offer("buyer").price == 110.0
    assert hist.get_last_offer().price == 140.0


def test_dialogue_get_last_complete_offer():
    hist = DialogueHistory()
    # Offer with all fields — should be returned
    hist.add_turn(_make_offer_turn("seller", 0, 150.0, delivery=10, warranty=12))
    assert hist.get_last_complete_offer() is not None
    assert hist.get_last_complete_offer().price == 150.0


def test_dialogue_no_offer_returns_none():
    hist = DialogueHistory()
    assert hist.get_last_offer() is None
    assert hist.get_last_complete_offer() is None


def test_dialogue_count_offers():
    hist = DialogueHistory()
    hist.add_turn(_make_offer_turn("seller", 0, 160))
    hist.add_turn(_make_offer_turn("buyer", 1, 110))
    hist.add_turn(_make_offer_turn("seller", 2, 140))
    assert hist.count_offers("seller") == 2
    assert hist.count_offers("buyer") == 1


def test_dialogue_count_backtracks():
    hist = DialogueHistory()
    # Seller: 160 → 140 → 150 (backtrack) → 130
    hist.add_turn(_make_offer_turn("seller", 0, 160))
    hist.add_turn(_make_offer_turn("seller", 2, 140))
    hist.add_turn(_make_offer_turn("seller", 4, 150))   # reversal here
    hist.add_turn(_make_offer_turn("seller", 6, 130))
    assert hist.count_backtracks("seller") >= 1


def test_dialogue_format_no_crash():
    hist = DialogueHistory()
    hist.add_turn(_make_offer_turn("buyer", 0, 120))
    text = hist.format_for_agent("seller")
    assert "BUYER" in text or "buyer" in text.lower()


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def test_simple_scenario_weights_sum_to_one():
    s = make_simple_scenario()
    bp = s.buyer
    sp = s.seller
    assert abs(bp.weight_price + bp.weight_delivery + bp.weight_warranty - 1.0) < 1e-6
    assert abs(sp.weight_price + sp.weight_delivery + sp.weight_warranty - 1.0) < 1e-6


def test_random_scenario_reproducible():
    s1 = generate_scenario(seed=42)
    s2 = generate_scenario(seed=42)
    assert s1.buyer.willingness_to_pay == s2.buyer.willingness_to_pay
    assert s1.seller.min_acceptable_price == s2.seller.min_acceptable_price


def test_random_scenario_varies():
    s1 = generate_scenario(seed=1)
    s2 = generate_scenario(seed=2)
    assert s1.buyer.willingness_to_pay != s2.buyer.willingness_to_pay


# ---------------------------------------------------------------------------
# Engine integration (mock agents)
# ---------------------------------------------------------------------------

def test_engine_mock_produces_result():
    scenario = make_simple_scenario(wtp=150, cost=80, noise=0.0)
    engine = NegotiationEngine(MockBuyer(), MockSeller(), verbose=False)
    result = engine.run_episode(scenario)
    assert result.outcome in ("deal", "quit", "timeout")


def test_engine_mock_deal_has_offer():
    scenario = make_simple_scenario(wtp=150, cost=80, noise=0.0)
    engine = NegotiationEngine(MockBuyer(), MockSeller(), verbose=False)
    result = engine.run_episode(scenario)
    if result.outcome == "deal":
        assert result.final_deal is not None
        assert result.final_deal.price > 0


def test_engine_turn_count_within_limit():
    scenario = make_simple_scenario(wtp=150, cost=80, noise=0.0, max_turns=16)
    engine = NegotiationEngine(MockBuyer(), MockSeller(), verbose=False)
    result = engine.run_episode(scenario)
    assert len(result.turns) <= scenario.max_turns


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_buyer_utility_no_deal():
    from core.schema import EpisodeResult
    scenario = make_simple_scenario()
    result = EpisodeResult(scenario_id="x", outcome="quit", deal=False, final_deal=None)
    u = compute_buyer_utility(result, scenario)
    assert u == 0.0


def test_compute_buyer_utility_at_wtp():
    """If price == WTP, price satisfaction == 0 → utility is low."""
    from core.schema import EpisodeResult
    scenario = make_simple_scenario(wtp=150)
    offer = Offer(price=150.0, delivery_days=1.0, warranty_months=24.0)
    result = EpisodeResult(
        scenario_id="x", outcome="deal", deal=True, final_deal=offer, buyer_utility=0.0
    )
    u = compute_buyer_utility(result, scenario)
    # price satisfaction = 0 → utility driven by delivery + warranty weights
    assert 0.0 <= u <= 1.0


def test_compute_buyer_utility_great_deal():
    """Buyer gets price=50 (below WTP=150), fast delivery, long warranty → high utility."""
    from core.schema import EpisodeResult
    scenario = make_simple_scenario(wtp=150)
    offer = Offer(price=50.0, delivery_days=1.0, warranty_months=24.0)
    result = EpisodeResult(
        scenario_id="x", outcome="deal", deal=True, final_deal=offer, buyer_utility=0.0
    )
    u = compute_buyer_utility(result, scenario)
    assert u > 0.7


def test_deal_rate_calculation():
    from core.schema import EpisodeResult
    results = [
        EpisodeResult("a", "deal", deal=True, final_deal=None),
        EpisodeResult("b", "deal", deal=True, final_deal=None),
        EpisodeResult("c", "quit", deal=False, final_deal=None),
        EpisodeResult("d", "timeout", deal=False, final_deal=None),
    ]
    assert deal_rate(results) == 0.5


def test_delta_utility():
    from core.schema import EpisodeResult

    def _r(u, deal=True):
        r = EpisodeResult("x", "deal" if deal else "quit", deal=deal, final_deal=None)
        r.buyer_utility = u
        return r

    hidden = [_r(0.59), _r(0.59)]
    visible = [_r(0.65), _r(0.65)]
    assert abs(delta_utility(hidden, visible) - 0.06) < 1e-4


# ---------------------------------------------------------------------------
# Reputation
# ---------------------------------------------------------------------------

def test_reputation_tracker_initialises():
    t = ReputationTracker()
    assert t.score == 0.5


def test_reputation_tracker_update():
    from core.schema import EpisodeResult
    tracker = ReputationTracker(alpha=0.5, beta=0.2, initial=0.5)
    scenario = make_simple_scenario()
    engine = NegotiationEngine(MockBuyer(), MockSeller(), verbose=False)
    result = engine.run_episode(scenario)

    hist = DialogueHistory()
    for t in result.turns:
        hist.add_turn(t)

    new_score = tracker.update(result, "seller", hist)
    assert 0.0 <= new_score <= 1.0


def test_reputation_stays_in_bounds():
    from core.schema import EpisodeResult
    tracker = ReputationTracker(alpha=0.5, beta=0.2, initial=0.5)
    scenario = make_simple_scenario()
    engine = NegotiationEngine(MockBuyer(), MockSeller(), verbose=False)

    for _ in range(5):
        result = engine.run_episode(scenario)
        hist = DialogueHistory()
        for t in result.turns:
            hist.add_turn(t)
        tracker.update(result, "seller", hist)
        assert 0.0 <= tracker.score <= 1.0
