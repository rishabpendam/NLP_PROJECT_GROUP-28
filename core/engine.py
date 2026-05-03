"""
core/engine.py
Turn-loop engine for one negotiation episode.
Orchestrates buyer and seller agents, tracks structured answers,
and returns a complete EpisodeResult.
"""
from __future__ import annotations
import random
from typing import Optional

from core.schema import (
    Scenario, EpisodeResult, AgentTurn, StructuredAnswer, Offer
)
from core.dialogue import DialogueHistory
from core.action_parser import parse_agent_turn


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NegotiationEngine:
    """
    Runs one negotiation episode between two agents.

    Parameters
    ----------
    buyer_agent   : any object with a .act(scenario, history, rep_score) method
    seller_agent  : same interface
    verbose       : if True, print each turn to stdout
    """

    def __init__(self, buyer_agent, seller_agent, verbose: bool = False):
        self.buyer = buyer_agent
        self.seller = seller_agent
        self.verbose = verbose

    def run_episode(
        self,
        scenario: Scenario,
        buyer_rep_score: float = 0.5,
        seller_rep_score: float = 0.5,
    ) -> EpisodeResult:
        """
        Execute one full episode and return the result.
        """
        history = DialogueHistory()
        structured_answers: list[StructuredAnswer] = []
        pending_question: Optional[dict] = None   # {asker_role, question}

        result = EpisodeResult(
            scenario_id=scenario.scenario_id,
            outcome="timeout",
            buyer_reputation_score=buyer_rep_score,
            seller_reputation_score=seller_rep_score,
        )

        for turn_num in range(scenario.max_turns):
            role = "buyer" if turn_num % 2 == 0 else "seller"
            agent = self.buyer if role == "buyer" else self.seller
            rep_score = seller_rep_score if role == "buyer" else buyer_rep_score

            # Build the opponent's answer to any pending question
            opponent_answer = None
            if pending_question and pending_question["asker_role"] != role:
                opponent_answer = _generate_answer(
                    scenario, role, pending_question["question"], scenario.noise
                )
                # Record structured answer for reputation scoring
                sa = _make_structured_answer(
                    scenario, role, pending_question["question"],
                    opponent_answer, turn_num
                )
                structured_answers.append(sa)
                pending_question = None

            # Agent acts
            raw_text = agent.act(
                scenario=scenario,
                history=history,
                rep_score=rep_score if scenario.rep_visible else None,
                opponent_answer=opponent_answer,
            )

            turn = parse_agent_turn(raw_text, role, turn_num)
            history.add_turn(turn)

            if self.verbose:
                _print_turn(turn, opponent_answer)

            # --- Process action ---
            if turn.action == "DEAL":
                complete_offer = history.get_last_complete_offer()
                if complete_offer is None:
                    # No valid offer to accept — treat as QUIT
                    if self.verbose:
                        print(f"  [ENGINE] {role} played DEAL but no complete offer exists — treating as QUIT")
                    result.outcome = "quit"
                    break
                result.outcome = "deal"
                result.final_deal = complete_offer
                result.deal = True
                break

            elif turn.action == "QUIT":
                result.outcome = "quit"
                break

            elif turn.action == "ASK":
                pending_question = {
                    "asker_role": role,
                    "question": turn.action_payload.get("question", ""),
                }

            # OFFER — continue to next turn

        result.turns = history.get_turns()
        result.structured_answers = structured_answers
        return result


# ---------------------------------------------------------------------------
# Answer generation with noise
# ---------------------------------------------------------------------------

_ANSWER_FIELDS = {
    "min_price": ("seller", "min_acceptable_price"),
    "wtp": ("buyer", "willingness_to_pay"),
    "production_cost": ("seller", "production_cost"),
    "delivery": ("seller", None),
    "warranty": ("seller", None),
}


def _generate_answer(
    scenario: Scenario,
    answering_role: str,
    question: str,
    noise: float,
) -> str:
    """
    Generate a (possibly noisy) answer to a question.
    With probability (1-η) the answer is truthful; with probability η it is perturbed.
    """
    # Determine what to answer based on keywords in the question
    q_lower = question.lower()

    if answering_role == "seller":
        if "minimum" in q_lower or "floor" in q_lower or "lowest" in q_lower:
            truth = scenario.seller.min_acceptable_price
            field = "min_price"
        elif "cost" in q_lower:
            truth = scenario.seller.production_cost
            field = "production_cost"
        else:
            truth = scenario.seller.min_acceptable_price
            field = "min_price"
    else:  # buyer
        if "maximum" in q_lower or "budget" in q_lower or "most" in q_lower or "willing" in q_lower:
            truth = scenario.buyer.willingness_to_pay
            field = "wtp"
        else:
            truth = scenario.buyer.willingness_to_pay
            field = "wtp"

    # Apply noise
    if random.random() < noise:
        perturbed = truth * random.uniform(0.85, 1.20)
        return f"{perturbed:.1f}"
    return f"{truth:.1f}"


def _make_structured_answer(
    scenario: Scenario,
    answering_role: str,
    question: str,
    answer_given: str,
    turn_num: int,
) -> StructuredAnswer:
    q_lower = question.lower()

    if answering_role == "seller":
        if "cost" in q_lower:
            gt = scenario.seller.production_cost
            field = "production_cost"
        else:
            gt = scenario.seller.min_acceptable_price
            field = "min_price"
    else:
        gt = scenario.buyer.willingness_to_pay
        field = "wtp"

    try:
        is_truthful = abs(float(answer_given) - gt) / max(gt, 1e-9) < 0.05
    except ValueError:
        is_truthful = False

    return StructuredAnswer(
        turn_number=turn_num,
        answering_agent=answering_role,
        question=question,
        answer_given=answer_given,
        ground_truth=str(gt),
        field=field,
        is_truthful=is_truthful,
    )


# ---------------------------------------------------------------------------
# Debug printing
# ---------------------------------------------------------------------------

def _print_turn(turn: AgentTurn, opponent_answer: Optional[str]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Turn {turn.turn_number:02d}  [{turn.role.upper()}]")
    if opponent_answer:
        print(f"  (Opponent answered: {opponent_answer})")
    if turn.thought:
        print(f"  THOUGHT: {turn.thought[:120]}")
    if turn.talk:
        print(f"  TALK:    {turn.talk[:120]}")
    if turn.action == "OFFER" and turn.offer:
        o = turn.offer
        print(f"  ACTION:  OFFER price={o.price:.2f} delivery={o.delivery_days:.0f}d warranty={o.warranty_months:.0f}mo")
    else:
        print(f"  ACTION:  {turn.action}")
