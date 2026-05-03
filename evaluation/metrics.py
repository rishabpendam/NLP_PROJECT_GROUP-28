"""
evaluation/metrics.py
Evaluation metrics from the paper (Section 4.7).

Primary metric: buyer utility
  U = w_p · s_p + w_d · s_d + w_w · s_w

  where each satisfaction score s ∈ [0,1] is normalised so that:
    - the best possible outcome scores 1.0
    - the worst acceptable outcome scores 0.0

Secondary metric: deal rate
  D = #deals / #episodes

Comparative metrics:
  ΔU = U_visible − U_hidden
  ΔD = D_visible − D_hidden
"""
from __future__ import annotations
from typing import List
from core.schema import EpisodeResult, Scenario, Offer


# ---------------------------------------------------------------------------
# Per-episode buyer utility
# ---------------------------------------------------------------------------

def compute_buyer_utility(result: EpisodeResult, scenario: Scenario) -> float:
    """
    Compute buyer utility for a completed episode.
    Returns 0.0 if no deal was reached.
    """
    if not result.deal or result.final_deal is None:
        return 0.0

    offer = result.final_deal
    bp = scenario.buyer

    sp = _price_satisfaction(offer.price, scenario)
    sd = _delivery_satisfaction(offer.delivery_days, scenario)
    sw = _warranty_satisfaction(offer.warranty_months, scenario)

    utility = bp.weight_price * sp + bp.weight_delivery * sd + bp.weight_warranty * sw
    return round(max(0.0, min(1.0, utility)), 4)


def _price_satisfaction(price: float, scenario: Scenario) -> float:
    """Higher satisfaction when price is lower (buyer prefers cheaper)."""
    wtp = scenario.buyer.willingness_to_pay
    p_min = scenario.price_min
    # Normalise: price == p_min → 1.0; price == wtp → 0.0
    if wtp <= p_min:
        return 1.0
    sat = (wtp - price) / (wtp - p_min)
    return max(0.0, min(1.0, sat))


def _delivery_satisfaction(delivery_days: float, scenario: Scenario) -> float:
    """Higher satisfaction when delivery is faster (buyer prefers shorter)."""
    d_min = scenario.delivery_min
    d_max = scenario.delivery_max
    if d_max <= d_min:
        return 1.0
    sat = (d_max - delivery_days) / (d_max - d_min)
    return max(0.0, min(1.0, sat))


def _warranty_satisfaction(warranty_months: float, scenario: Scenario) -> float:
    """Higher satisfaction when warranty is longer (buyer prefers more coverage)."""
    w_min = scenario.warranty_min
    w_max = scenario.warranty_max
    if w_max <= w_min:
        return 1.0
    sat = (warranty_months - w_min) / (w_max - w_min)
    return max(0.0, min(1.0, sat))


# ---------------------------------------------------------------------------
# Aggregate metrics over a list of episodes
# ---------------------------------------------------------------------------

def deal_rate(results: List[EpisodeResult]) -> float:
    """D = #deals / #episodes"""
    if not results:
        return 0.0
    return round(sum(1 for r in results if r.deal) / len(results), 4)


def average_buyer_utility(results: List[EpisodeResult]) -> float:
    """Average buyer_utility field across all episodes."""
    if not results:
        return 0.0
    return round(sum(r.buyer_utility for r in results) / len(results), 4)


def delta_utility(
    results_hidden: List[EpisodeResult],
    results_visible: List[EpisodeResult],
) -> float:
    """ΔU = U_visible − U_hidden"""
    return round(
        average_buyer_utility(results_visible) - average_buyer_utility(results_hidden), 4
    )


def delta_deal_rate(
    results_hidden: List[EpisodeResult],
    results_visible: List[EpisodeResult],
) -> float:
    """ΔD = D_visible − D_hidden"""
    return round(deal_rate(results_visible) - deal_rate(results_hidden), 4)


# ---------------------------------------------------------------------------
# Reputation-grouped deal rate (Table 3 in paper)
# ---------------------------------------------------------------------------

def deal_rate_by_opponent_reputation(
    results: List[EpisodeResult],
    rep_scores: List[float],
) -> dict:
    """
    Compute deal rate broken down by opponent reputation tier.

    Returns:
        {"low": float, "medium": float, "high": float}
    """
    buckets: dict[str, list[bool]] = {"low": [], "medium": [], "high": []}

    for result, rep in zip(results, rep_scores):
        if rep < 0.3:
            tier = "low"
        elif rep < 0.7:
            tier = "medium"
        else:
            tier = "high"
        buckets[tier].append(result.deal)

    return {
        tier: round(sum(deals) / len(deals), 4) if deals else 0.0
        for tier, deals in buckets.items()
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(
    results: List[EpisodeResult],
    label: str = "Results",
) -> None:
    u = average_buyer_utility(results)
    d = deal_rate(results)
    n = len(results)
    print(f"\n{'═' * 50}")
    print(f"  {label}  ({n} episodes)")
    print(f"{'─' * 50}")
    print(f"  Buyer Utility (U) : {u:.4f}")
    print(f"  Deal Rate     (D) : {d:.4f}  ({sum(r.deal for r in results)}/{n} deals)")
    print(f"{'═' * 50}")
