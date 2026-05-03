"""
scripts/analyse_results.py
Load JSON episode logs from a results directory and print evaluation metrics.

Usage
-----
# Summary of one directory:
  python scripts/analyse_results.py --dir results/

# Side-by-side ΔU and ΔD comparison:
  python scripts/analyse_results.py --compare results/norep_hidden results/rep_visible

# Noise sweep analysis (Table 4 in paper):
  python scripts/analyse_results.py --noise-sweep results/noise_0 results/noise_0.2 results/noise_0.5 results/noise_0.8
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results_dir(directory: str) -> list[dict]:
    """Load all episode JSON files (not summary files) from a directory."""
    p = Path(directory)
    if not p.exists():
        print(f"[ERROR] Directory not found: {directory}")
        return []

    episodes = []
    for f in sorted(p.glob("*.json")):
        if "summary" in f.name:
            continue
        with open(f) as fh:
            try:
                episodes.append(json.load(fh))
            except json.JSONDecodeError:
                print(f"  [WARN] Could not parse {f.name}")
    return episodes


def compute_metrics(episodes: list[dict]) -> dict:
    if not episodes:
        return {"n": 0, "deal_rate": 0.0, "avg_utility": 0.0}

    n_deals = sum(1 for e in episodes if e.get("deal", False))
    utilities = [e.get("buyer_utility", 0.0) for e in episodes]
    avg_u = sum(utilities) / len(utilities)
    deal_r = n_deals / len(episodes)

    return {
        "n": len(episodes),
        "deal_rate": round(deal_r, 4),
        "avg_utility": round(avg_u, 4),
        "n_deals": n_deals,
    }


def print_metrics(label: str, m: dict) -> None:
    print(f"\n{'─' * 52}")
    print(f"  {label}  (n={m['n']})")
    print(f"{'─' * 52}")
    print(f"  Deal Rate     (D) : {m['deal_rate']:.4f}  ({m['n_deals']}/{m['n']} deals)")
    print(f"  Buyer Utility (U) : {m['avg_utility']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyse negotiation experiment results")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dir", type=str,
                     help="Single results directory to summarise")
    grp.add_argument("--compare", nargs=2, metavar=("HIDDEN_DIR", "VISIBLE_DIR"),
                     help="Compare two conditions — prints ΔU and ΔD (hidden vs visible)")
    grp.add_argument("--noise-sweep", nargs="+", metavar="DIR",
                     help="Analyse multiple noise-level directories in order")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 52)
    print("  NLP Project — Results Analysis")
    print("═" * 52)

    if args.dir:
        episodes = load_results_dir(args.dir)
        m = compute_metrics(episodes)
        print_metrics(args.dir, m)

    elif args.compare:
        hidden_dir, visible_dir = args.compare
        ep_hidden = load_results_dir(hidden_dir)
        ep_visible = load_results_dir(visible_dir)

        m_h = compute_metrics(ep_hidden)
        m_v = compute_metrics(ep_visible)

        print_metrics(f"HIDDEN  — {hidden_dir}", m_h)
        print_metrics(f"VISIBLE — {visible_dir}", m_v)

        delta_u = round(m_v["avg_utility"] - m_h["avg_utility"], 4)
        delta_d = round(m_v["deal_rate"] - m_h["deal_rate"], 4)

        print(f"\n{'═' * 52}")
        print(f"  REPUTATION EFFECT (paper Table 1, Section 5.2)")
        print(f"{'─' * 52}")
        print(f"  ΔU = U_visible − U_hidden = {delta_u:+.4f}")
        print(f"  ΔD = D_visible − D_hidden = {delta_d:+.4f}")
        print(f"{'═' * 52}")

    elif args.noise_sweep:
        print(f"\n  {'Noise η':<12} {'Deal Rate':>12} {'Avg Utility':>14}")
        print(f"  {'─' * 40}")
        for i, d in enumerate(args.noise_sweep):
            eps = load_results_dir(d)
            m = compute_metrics(eps)
            noise_label = f"η={i * 0.2:.1f}" if i < 5 else d
            noise_val = _infer_noise(eps)
            print(f"  {noise_val:<12} {m['deal_rate']:>12.4f} {m['avg_utility']:>14.4f}  (n={m['n']})")

    print()


def _infer_noise(episodes: list[dict]) -> str:
    """Try to read noise from the first episode's scenario field."""
    if episodes and "scenario" in episodes[0]:
        return str(episodes[0]["scenario"].get("noise", "?"))
    return "?"


if __name__ == "__main__":
    main()
