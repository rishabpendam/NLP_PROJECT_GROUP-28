# LLM Negotiation — Reputation Effects in Multi-Issue Bargaining

NLP Project · USC · Evaluating Reputation Effects in Multi-Issue LLM Negotiation

---

## Project Overview

Two LLM agents (buyer and seller) negotiate over **price**, **delivery time**, and **warranty**. Each agent has private parameters (willingness-to-pay, production cost, preference weights) hidden from the other. Agents can ask questions about each other's constraints, but answers may be noisy. Across repeated episodes, agents accumulate a **reputation score** based on their truthfulness and offer consistency.

We evaluate two experimental conditions:
- **LLM-NoRep** (baseline): agent sees only its own private info and dialogue history
- **LLM-Rep**: agent additionally receives the opponent's reputation score ∈ [0,1]

**Key results (paper Table 1):**

| Method     | Buyer Utility (U) | Deal Rate (D) |
|------------|-------------------|---------------|
| LLM-NoRep  | 0.59              | 0.64          |
| LLM-Rep    | 0.65              | 0.70          |
| **ΔU / ΔD**| **+0.06**         | **+0.06**     |

---

## Repository Structure

```
nlp_project/
│
├── core/
│   ├── schema.py          ← shared dataclasses (Scenario, AgentTurn, EpisodeResult, …)
│   ├── engine.py          ← turn-loop engine, episode runner
│   ├── action_parser.py   ← parse raw LLM text → AgentTurn (Thought–Talk–Action)
│   └── dialogue.py        ← dialogue history manager
│
├── agents/
│   ├── base_agent.py         ← abstract interface all agents implement
│   ├── mock_agent.py         ← scripted MockBuyer + MockSeller (no API key)
│   ├── prompts.py            ← shared system prompts and user-message builders
│   ├── llm_norep_ollama.py   ← LLM-NoRep using local Ollama / Llama 3 (paper model)
│   ├── llm_rep_ollama.py     ← LLM-Rep  using local Ollama / Llama 3 (paper model)
│   ├── llm_norep_groq.py     ← LLM-NoRep using Groq (optional)
│   ├── llm_rep_groq.py       ← LLM-Rep  using Groq (optional)
│   ├── llm_norep_anthropic.py← LLM-NoRep using Anthropic Claude (optional)
│   └── llm_rep_anthropic.py  ← LLM-Rep  using Anthropic Claude (optional)
│
├── scenario/
│   └── generator.py       ← random + fixed scenario creation
│
├── reputation/
│   └── module.py          ← truthfulness + consistency → reputation score
│
├── evaluation/
│   └── metrics.py         ← buyer utility, deal rate, ΔU, ΔD
│
├── scripts/
│   ├── smoke_test.py         ← 1-episode sanity check (no API key)
│   ├── run_experiments.py    ← main experiment runner
│   └── analyse_results.py    ← load JSON logs and print stats
│
├── tests/
│   ├── test_parser.py     ← unit tests for action parser
│   └── test_engine.py     ← unit tests for engine, dialogue, metrics, reputation
│
├── results/               ← JSON episode logs (git-ignored except .gitkeep)
├── .env.example           ← only needed for optional cloud providers
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## System / Device Used

The experiments in the paper were run using:
- **Model**: Llama 3.1 (Meta AI, 2024) via Ollama running locally
- **OS**: macOS 14 (Apple Silicon) — also works on Ubuntu 22.04 and any POSIX system
- **Python**: 3.10 or 3.11
- **RAM**: ≥ 8 GB — the MacBook Air M1 with 8 GB runs the Llama 3 8B model via Ollama
- **GPU**: Not required, but recommended

---

## Environment Setup

### 1. Clone the repo

```bash
git clone https://github.com/rishabpendam/NLP_PROJECT_GROUP-28
cd nlp_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model

Ollama runs Llama 3 fully locally — no API key or internet connection needed during inference.

```bash
# 1. Install Ollama from https://ollama.com

# 2. Pull the model (one-time download):
ollama pull llama3

# 3. Start the server — keep this running in a separate terminal for all experiments:
ollama serve
```

No `.env` file is needed when using Ollama.

---

## Running the Code

### Step 0 — Smoke test (no Ollama needed)

Verifies that the engine, parser, dialogue manager, and mock agents all work correctly:

```bash
python scripts/smoke_test.py
```

Expected output: a single episode with scripted mock agents, ending in `✅ Smoke test PASSED`.

### Step 1 — Unit tests (no Ollama needed)

```bash
pytest -v
```

Expected: all 32 tests in `tests/test_parser.py` and `tests/test_engine.py` pass.

### Step 2 — Mock pipeline test (no Ollama needed)

```bash
python scripts/run_experiments.py --mock --episodes 10 --verbose --random-scenarios
```

Runs 10 scripted episodes end-to-end and saves JSON logs to `results/`. Good for verifying the full pipeline before starting real LLM runs.

### Step 3 — Real experiments with Ollama

Make sure `ollama serve` is running in a separate terminal, then:

```bash
# Baseline condition: LLM-NoRep, reputation hidden
python scripts/run_experiments.py \
    --provider ollama --episodes 1 \
    --noise 0.2 --random-scenarios \
    --out-dir results/norep_hidden

# Reputation-visible condition: LLM-Rep
python scripts/run_experiments.py \
    --provider ollama --episodes 1 \
    --noise 0.2 --rep-visible --use-rep-agent --random-scenarios \
    --out-dir results/rep_visible
```

---

## Full Experimental Sweep (Replicating the Paper)

Run all conditions from the paper. Make sure `ollama serve` is running first.

```bash
mkdir -p results/{norep_hidden,rep_visible,noise_00,noise_02,noise_05,noise_08}

# Table 1 — Condition 1: LLM-NoRep, reputation hidden (baseline)
python scripts/run_experiments.py --provider ollama --episodes 100 \
    --noise 0.2 --random-scenarios --out-dir results/norep_hidden

# Table 1 — Condition 2: LLM-Rep, reputation visible
python scripts/run_experiments.py --provider ollama --episodes 100 \
    --noise 0.2 --rep-visible --use-rep-agent --random-scenarios \
    --out-dir results/rep_visible

# Table 4 — Noise sweep: η = 0.0, 0.2, 0.5, 0.8
for NOISE in 0.0 0.2 0.5 0.8; do
    python scripts/run_experiments.py --provider ollama --episodes 50 \
        --noise $NOISE --rep-visible --use-rep-agent --random-scenarios \
        --out-dir results/noise_${NOISE/./}
done
```

---

## Analysing Results

```bash
# Summary of one directory
python scripts/analyse_results.py --dir results/rep_visible

# Side-by-side ΔU and ΔD (replicates Table 1)
python scripts/analyse_results.py \
    --compare results/norep_hidden results/rep_visible

# Noise sweep (replicates Table 4)
python scripts/analyse_results.py \
    --noise-sweep results/noise_00 results/noise_02 results/noise_05 results/noise_08
```

---

## CLI Reference — `run_experiments.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | `5` | Number of negotiation episodes to run |
| `--mock` | off | Use scripted mock agents — no Ollama needed |
| `--provider NAME` | `ollama` | LLM backend: `ollama`, `groq`, `anthropic` |
| `--noise FLOAT` | `0.2` | Noise level η ∈ [0,1] — probability of distorted answers |
| `--rep-visible` | off | Inject opponent reputation score into agent prompts |
| `--use-rep-agent` | off | Use LLM-Rep (reputation-aware) instead of LLM-NoRep |
| `--random-scenarios` | off | Generate varied scenarios seeded by episode index |
| `--verbose` | off | Print each turn to the console |
| `--out-dir PATH` | `results` | Directory to save episode JSON logs |
| `--max-turns N` | `16` | Maximum turns per episode before timeout |

---

## How Results Are Generated

### 1. Scenario generation

Each episode samples a `Scenario` with:
- **Buyer**: `willingness_to_pay` ∈ [110, 180], random preference weights (sum to 1)
- **Seller**: `production_cost` ∈ [60, 100], `min_acceptable_price` = cost × [1.05, 1.20]
- **Seller initial offer**: sampled from WTP × [0.90, 1.10], with a hard floor of `min_acceptable_price` × 1.30. This ensures the seller always opens above the buyer's expected first offer (~60% of WTP), creating a proper zone of possible agreement (ZOPA) to negotiate into.
- **Noise** η (constant across main experiments at 0.2)
- **rep_visible** flag (controls whether reputation is injected into prompts)

### 2. Episode loop (engine.py)

Each episode runs up to `max_turns` turns alternating buyer/seller. Every turn:
1. The acting agent's `act()` method is called with its private info + dialogue history
2. The raw text response is parsed into a `Thought–Talk–Action` structure
3. The action is executed: OFFER continues the dialogue, DEAL/QUIT end the episode

Dialogue history is presented to each agent with explicit `YOU (BUYER)` / `OPPONENT (SELLER)` labels so agents cannot lose track of their own role across long conversations.

When an ASK is played, the opponent generates a (possibly noisy) answer on their next turn.

### 3. Reputation scoring (reputation/module.py)

After each episode, reputation is updated for both agents:

```
Truthfulness  T  = N_truthful / N_answers        (fraction of honest answers)
Consistency   C  = 1 − N_backtrack / N_moves     (fraction of non-reversing offers)
Rep_episode      = α·T + (1−α)·C                 (α = 0.5)
R_{t+1}          = (1−β)·R_t + β·Rep_episode     (β = 0.2)
```

### 4. Utility calculation (evaluation/metrics.py)

After a deal is reached, buyer utility is computed as:

```
U = w_p · s_p + w_d · s_d + w_w · s_w

s_p = (WTP − price) / (WTP − price_min)          [0,1]
s_d = (delivery_max − days) / (delivery_max − delivery_min)
s_w = (months − warranty_min) / (warranty_max − warranty_min)
```

No deal → U = 0.

### 5. Output

Each episode saves one JSON file to the output directory. `analyse_results.py` loads all JSON files in a directory and computes aggregate metrics. The `--compare` flag prints ΔU and ΔD between two conditions.

---

## Agent Action Format

Every LLM response must follow this exact format:

```
THOUGHT: <internal reasoning — not shown to opponent>
TALK: <natural language message to opponent>
ACTION: OFFER price=120 delivery_days=7 warranty_months=12
```

Valid actions:
- `OFFER price=X delivery_days=Y warranty_months=Z` — all three fields required
- `ASK <question>` — ask opponent about their hidden parameters (at most once)
- `DEAL` — accept the current standing offer
- `QUIT` — walk away from the negotiation

---

## Reputation Model (Section 4.6)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α | 0.5 | Equal weight to truthfulness and consistency |
| β | 0.2 | Slow update — reputation doesn't shift drastically after one episode |
| initial | 0.5 | Neutral starting point — no prior information |

Reputation ∈ [0, 1]. Values < 0.3 = low (untrustworthy), 0.3–0.7 = medium, > 0.7 = high.


---

## Provider Reference

| Provider | Cost | Setup | Model |
|----------|------|-------|-------|
| **Ollama (local)** | **Free** | **Install from ollama.com, `ollama pull llama3`** | **`llama3`** |
| Groq *(optional)* | Free tier | `pip install groq`, add `GROQ_API_KEY` to `.env` | `llama-3.3-70b-versatile` |
| Anthropic *(optional)* | Paid | `pip install anthropic`, add `ANTHROPIC_API_KEY` to `.env` | `claude-haiku-4-5-20251001` |

---