"""
scripts/run_experiments.py
Main experiment runner.

Runs N negotiation episodes with the chosen provider and saves JSON logs to
the output directory. Supports all four experimental conditions from the paper.

Usage examples
--------------
# Mock (no API key):
  python scripts/run_experiments.py --mock --episodes 10 --verbose

# Groq (free):
  python scripts/run_experiments.py --provider groq --episodes 50 --noise 0.2

# Ollama (local Llama 3):
  python scripts/run_experiments.py --provider ollama --episodes 50 --noise 0.2

# Rep-visible condition:
  python scripts/run_experiments.py --provider groq --episodes 50 \\
      --noise 0.2 --rep-visible --use-rep-agent

# Full paper sweep (requires separate calls):
  python scripts/run_experiments.py --provider groq --episodes 100 \\
      --noise 0.2 --rep-visible --use-rep-agent --out-dir results/rep_rep
"""
import sys
import os
import json
import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from scenario.generator import generate_scenario, make_simple_scenario
from core.engine import NegotiationEngine
from core.dialogue import DialogueHistory
from reputation.module import ReputationTracker
from evaluation.metrics import (
    compute_buyer_utility, deal_rate, average_buyer_utility, print_summary
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run LLM negotiation experiments (paper replication)"
    )
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of negotiation episodes to run (default: 5)")
    p.add_argument("--mock", action="store_true",
                   help="Use scripted mock agents — no API key needed")
    p.add_argument("--provider", type=str, default="groq",
                   choices=["groq", "ollama", "anthropic"],
                   help="LLM provider (default: groq)")
    p.add_argument("--noise", type=float, default=0.2,
                   help="Noise level η ∈ [0,1] — probability of distorted answers (default: 0.2)")
    p.add_argument("--rep-visible", action="store_true",
                   help="Make opponent reputation visible to agents")
    p.add_argument("--use-rep-agent", action="store_true",
                   help="Use LLM-Rep (reputation-aware) agent instead of LLM-NoRep")
    p.add_argument("--random-scenarios", action="store_true",
                   help="Generate varied random scenarios (seeded by episode index)")
    p.add_argument("--verbose", action="store_false" if "--no-verbose" in sys.argv else "store_true",
                   help="Print each turn to the console")
    p.add_argument("--out-dir", type=str, default="results",
                   help="Directory to save episode JSON logs (default: results/)")
    p.add_argument("--max-turns", type=int, default=16,
                   help="Maximum turns per episode (default: 16)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agents(args, role_buyer, role_seller):
    if args.mock:
        from agents.mock_agent import MockBuyer, MockSeller
        return MockBuyer(), MockSeller()

    if args.provider == "groq":
        if args.use_rep_agent:
            from agents.llm_rep_groq import LLMRepGroqAgent
            return LLMRepGroqAgent("buyer"), LLMRepGroqAgent("seller")
        else:
            from agents.llm_norep_groq import LLMNoRepGroqAgent
            return LLMNoRepGroqAgent("buyer"), LLMNoRepGroqAgent("seller")

    elif args.provider == "ollama":
        if args.use_rep_agent:
            from agents.llm_rep_ollama import LLMRepOllamaAgent
            return LLMRepOllamaAgent("buyer"), LLMRepOllamaAgent("seller")
        else:
            from agents.llm_norep_ollama import LLMNoRepOllamaAgent
            return LLMNoRepOllamaAgent("buyer"), LLMNoRepOllamaAgent("seller")

    elif args.provider == "anthropic":
        if args.use_rep_agent:
            from agents.llm_rep_anthropic import LLMRepAnthropicAgent
            return LLMRepAnthropicAgent("buyer"), LLMRepAnthropicAgent("seller")
        else:
            from agents.llm_norep_anthropic import LLMNoRepAnthropicAgent
            return LLMNoRepAnthropicAgent("buyer"), LLMNoRepAnthropicAgent("seller")

    raise ValueError(f"Unknown provider: {args.provider}")


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def result_to_dict(result, scenario):
    """Convert EpisodeResult to a JSON-serialisable dict."""
    def offer_to_dict(o):
        if o is None:
            return None
        return {"price": o.price, "delivery_days": o.delivery_days, "warranty_months": o.warranty_months}

    turns = []
    for t in result.turns:
        turns.append({
            "role": t.role,
            "turn_number": t.turn_number,
            "thought": t.thought,
            "talk": t.talk,
            "action": t.action,
            "action_payload": t.action_payload,
            "offer": offer_to_dict(t.offer),
        })

    answers = []
    for sa in result.structured_answers:
        answers.append({
            "turn_number": sa.turn_number,
            "answering_agent": sa.answering_agent,
            "question": sa.question,
            "answer_given": sa.answer_given,
            "ground_truth": sa.ground_truth,
            "field": sa.field,
            "is_truthful": sa.is_truthful,
        })

    return {
        "scenario_id": result.scenario_id,
        "outcome": result.outcome,
        "deal": result.deal,
        "final_deal": offer_to_dict(result.final_deal),
        "buyer_utility": result.buyer_utility,
        "buyer_reputation_score": result.buyer_reputation_score,
        "seller_reputation_score": result.seller_reputation_score,
        "turns": turns,
        "structured_answers": answers,
        "scenario": {
            "buyer_wtp": scenario.buyer.willingness_to_pay,
            "buyer_weights": {
                "price": scenario.buyer.weight_price,
                "delivery": scenario.buyer.weight_delivery,
                "warranty": scenario.buyer.weight_warranty,
            },
            "seller_min_price": scenario.seller.min_acceptable_price,
            "seller_production_cost": scenario.seller.production_cost,
            "seller_initial_offer": scenario.initial_offer_price,
            "seller_weights": {
                "price": scenario.seller.weight_price,
                "delivery": scenario.seller.weight_delivery,
                "warranty": scenario.seller.weight_warranty,
            },
            "noise": scenario.noise,
            "rep_visible": scenario.rep_visible,
            "max_turns": scenario.max_turns,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buyer_agent, seller_agent = make_agents(args, "buyer", "seller")
    engine = NegotiationEngine(buyer_agent, seller_agent, verbose=args.verbose)

    buyer_tracker = ReputationTracker(alpha=0.5, beta=0.2, initial=0.5)
    seller_tracker = ReputationTracker(alpha=0.5, beta=0.2, initial=0.5)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition = "rep" if args.use_rep_agent else "norep"
    visibility = "visible" if args.rep_visible else "hidden"

    print(f"\n{'═' * 60}")
    print(f"  NLP Negotiation Experiment")
    print(f"  Provider   : {'mock' if args.mock else args.provider}")
    print(f"  Agent type : {condition}")
    print(f"  Reputation : {visibility}")
    print(f"  Noise η    : {args.noise}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Max turns  : {args.max_turns}")
    print(f"  Output dir : {out_dir}/")
    print(f"{'═' * 60}\n")

    for ep_idx in range(args.episodes):
        ep_id = f"ep{ep_idx + 1:04d}"

        if args.random_scenarios:
            scenario = generate_scenario(
                scenario_id=ep_id,
                seed=ep_idx,
                noise=args.noise,
                rep_visible=args.rep_visible,
                max_turns=args.max_turns,
            )
        else:
            scenario = make_simple_scenario(
                wtp=150.0, cost=80.0,
                noise=args.noise,
                rep_visible=args.rep_visible,
                scenario_id=ep_id,
                max_turns=args.max_turns,
            )

        # Run episode with current reputation scores
        result = engine.run_episode(
            scenario,
            buyer_rep_score=buyer_tracker.score,
            seller_rep_score=seller_tracker.score,
        )

        # Compute buyer utility
        utility = compute_buyer_utility(result, scenario)
        result.buyer_utility = utility
        result.buyer_reputation_score = buyer_tracker.score
        result.seller_reputation_score = seller_tracker.score

        # Update reputation trackers
        history = DialogueHistory()
        for t in result.turns:
            history.add_turn(t)
        buyer_tracker.update(result, "buyer", history)
        seller_tracker.update(result, "seller", history)

        all_results.append(result)

        # Save episode JSON
        ep_file = out_dir / f"{timestamp}_{ep_id}_{condition}_{visibility}.json"
        with open(ep_file, "w") as f:
            json.dump(result_to_dict(result, scenario), f, indent=2)

        # Progress
        status = "✓ DEAL" if result.deal else "✗ " + result.outcome.upper()
        print(f"  [{ep_id}] {status}  utility={utility:.4f}  "
              f"buyer_rep={buyer_tracker.score:.3f}  seller_rep={seller_tracker.score:.3f}")

    print_summary(all_results, label=f"{condition.upper()} | rep={visibility} | η={args.noise}")

    # Save aggregate summary JSON
    summary = {
        "condition": condition,
        "rep_visible": args.rep_visible,
        "noise": args.noise,
        "n_episodes": args.episodes,
        "provider": "mock" if args.mock else args.provider,
        "deal_rate": deal_rate(all_results),
        "avg_buyer_utility": average_buyer_utility(all_results),
        "final_buyer_rep": buyer_tracker.score,
        "final_seller_rep": seller_tracker.score,
    }
    summary_file = out_dir / f"{timestamp}_summary_{condition}_{visibility}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Logs saved to: {out_dir}/")
    print(f"  Summary file : {summary_file.name}")


if __name__ == "__main__":
    main()
