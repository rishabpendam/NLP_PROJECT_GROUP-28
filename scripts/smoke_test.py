"""
scripts/smoke_test.py
Single-episode sanity check — no API key required.
Runs mock agents through the full engine pipeline and prints results.

Usage:
    python scripts/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario.generator import make_simple_scenario
from agents.mock_agent import MockBuyer, MockSeller
from core.engine import NegotiationEngine
from core.dialogue import DialogueHistory
from reputation.module import ReputationTracker, compute_episode_reputation
from evaluation.metrics import compute_buyer_utility


def main():
    print("=" * 60)
    print("  NLP Project — Smoke Test")
    print("  (Mock agents, no API key needed)")
    print("=" * 60)

    scenario = make_simple_scenario(
        wtp=150.0,
        cost=80.0,
        noise=0.0,
        rep_visible=False,
        scenario_id="smoke_test",
        max_turns=16,
    )

    print(f"\nScenario: {scenario.scenario_id}")
    print(f"  Buyer WTP       : ${scenario.buyer.willingness_to_pay:.2f}")
    print(f"  Seller min price: ${scenario.seller.min_acceptable_price:.2f}")
    print(f"  Initial offer   : ${scenario.initial_offer_price:.2f}")
    print(f"  Max turns       : {scenario.max_turns}")
    print(f"  Noise η         : {scenario.noise}")

    buyer = MockBuyer()
    seller = MockSeller()
    engine = NegotiationEngine(buyer, seller, verbose=True)

    result = engine.run_episode(scenario)

    # Compute utility
    utility = compute_buyer_utility(result, scenario)
    result.buyer_utility = utility

    # Compute reputation (using dialogue history from result)
    from core.dialogue import DialogueHistory as DH
    history = DH()
    for t in result.turns:
        history.add_turn(t)

    buyer_rep = compute_episode_reputation(result, history, "buyer")
    seller_rep = compute_episode_reputation(result, history, "seller")

    print("\n" + "=" * 60)
    print("  EPISODE RESULT")
    print("=" * 60)
    print(f"  Outcome       : {result.outcome.upper()}")
    if result.final_deal:
        d = result.final_deal
        print(f"  Final deal    : price=${d.price:.2f}, delivery={d.delivery_days:.0f}d, warranty={d.warranty_months:.0f}mo")
    print(f"  Buyer utility : {utility:.4f}")
    print(f"  Total turns   : {len(result.turns)}")
    print(f"  Buyer rep (episode)  : {buyer_rep:.4f}")
    print(f"  Seller rep (episode) : {seller_rep:.4f}")
    print("=" * 60)

    if result.outcome == "deal":
        print("\n✅  Smoke test PASSED — engine, parser, dialogue, and mock agents all work.")
    else:
        print("\n⚠️   No deal reached — check agent logic if unexpected.")

    return result.outcome == "deal"


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
