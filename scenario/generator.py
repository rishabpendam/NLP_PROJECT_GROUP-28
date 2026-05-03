"""
scenario/generator.py
Creates negotiation scenarios — either randomly (seeded) or as fixed examples.

Random generation ensures a diverse set of negotiation environments while still
knowing the ground-truth private values (needed for reputation scoring).
"""
from __future__ import annotations
import random
from core.schema import Scenario, BuyerParams, SellerParams


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def generate_scenario(
    scenario_id: str = "ep0001",
    seed: int | None = None,
    noise: float = 0.2,
    rep_visible: bool = False,
    max_turns: int = 16,
) -> Scenario:
    """
    Generate a random-but-reproducible negotiation scenario.

    Parameters
    ----------
    scenario_id  : human-readable label (e.g. "ep0001")
    seed         : random seed for reproducibility; None = fully random
    noise        : η — probability an answer is misleading ∈ [0, 1]
    rep_visible  : whether reputation scores are injected into agent prompts
    max_turns    : maximum dialogue turns before episode times out
    """
    rng = random.Random(seed)

    # ---- Buyer ----
    wtp = rng.uniform(110, 180)
    bp_raw = _random_weights(rng)
    buyer = BuyerParams(
        willingness_to_pay=round(wtp, 2),
        weight_price=bp_raw[0],
        weight_delivery=bp_raw[1],
        weight_warranty=bp_raw[2],
    )

    # ---- Seller ----
    cost = rng.uniform(60, 100)
    min_price = cost * rng.uniform(1.05, 1.20)     # floor is cost + margin
    sp_raw = _random_weights(rng)
    seller = SellerParams(
        production_cost=round(cost, 2),
        min_acceptable_price=round(min_price, 2),
        weight_price=sp_raw[0],
        weight_delivery=sp_raw[1],
        weight_warranty=sp_raw[2],
    )

    # ---- Environment ----
    # Seller's opening anchor must sit ABOVE the buyer's likely first offer.
    # Buyer opens at ~55-65% of WTP (per prompt). Seller must open higher than that
    # to leave a proper ZOPA to negotiate into.
    # Anchor: 90-110% of WTP ensures seller opens above buyer regardless of WTP/cost ratio.
    initial_offer = rng.uniform(wtp * 0.90, wtp * 1.10)
    # Hard floor: never let it slip below min_price + meaningful margin
    initial_offer = max(initial_offer, min_price * 1.30)

    return Scenario(
        scenario_id=scenario_id,
        buyer=buyer,
        seller=seller,
        initial_offer_price=round(initial_offer, 2),
        max_turns=max_turns,
        noise=noise,
        rep_visible=rep_visible,
        price_min=50.0,
        price_max=200.0,
        delivery_min=1.0,
        delivery_max=30.0,
        warranty_min=1.0,
        warranty_max=24.0,
    )


def make_simple_scenario(
    wtp: float = 150.0,
    cost: float = 80.0,
    noise: float = 0.0,
    rep_visible: bool = False,
    scenario_id: str = "simple",
    max_turns: int = 16,
) -> Scenario:
    """
    Create a deterministic scenario with equal preference weights.
    Useful for unit tests and smoke tests.
    """
    buyer = BuyerParams(
        willingness_to_pay=wtp,
        weight_price=0.5,
        weight_delivery=0.3,
        weight_warranty=0.2,
    )
    seller = SellerParams(
        production_cost=cost,
        min_acceptable_price=cost * 1.1,
        weight_price=0.5,
        weight_delivery=0.3,
        weight_warranty=0.2,
    )
    return Scenario(
        scenario_id=scenario_id,
        buyer=buyer,
        seller=seller,
        initial_offer_price=160.0,
        max_turns=max_turns,
        noise=noise,
        rep_visible=rep_visible,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_weights(rng: random.Random) -> tuple[float, float, float]:
    """Sample three non-negative weights that sum to 1.0."""
    a = rng.random()
    b = rng.random()
    vals = sorted([0.0, a, b, 1.0])
    w = [round(vals[i + 1] - vals[i], 4) for i in range(3)]
    # Normalise to fix floating-point drift
    total = sum(w)
    w = [x / total for x in w]
    return (round(w[0], 4), round(w[1], 4), round(w[2], 4))
