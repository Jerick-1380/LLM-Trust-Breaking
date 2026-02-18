# LLM Strategic Deception Research

**Do LLMs lie when it increases their payoff?**

![Methodology Overview](methodology.png)

This repository implements two complementary experimental frameworks for testing whether LLMs strategically deceive other agents in game-theoretic settings:

1. **Scenario Enumeration** — exhaustively tests all announcement profiles to map the full lying landscape
2. **Endogenous Promises** — 3-stage protocol measuring premeditated vs. impulsive deception, with multi-round learning

---

## Overview

We test LLM behavior across strategic games where agents can make public commitments before acting. The core question: do LLMs break those commitments when it pays to do so, and do they plan the deception in advance?

---

## Games Tested

Six symmetric one-shot games spanning binary and continuous action spaces:

| Game | Actions | Strategic Challenge |
|------|---------|---------------------|
| **Volunteer's Dilemma** | YES / NO | Coordination with cost — someone must volunteer |
| **El Farol Bar** | GO / STAY | Anti-coordination — avoid congestion |
| **Diner's Dilemma** | EXPENSIVE / CHEAP | Cost-sharing with temptation to free-ride |
| **Fishing** | 0–5 fish | Tragedy of the Commons — overfishing collapses the lake |
| **Public Goods** | 0–5 tokens | Under-contribution to shared pool |
| **Weakest Link** | 0–5 effort | Coordinate on group minimum — don't be the weak link |

---

## Experimental Protocols

### Protocol 1 — Scenario Enumeration (Exogenous Promises)

Tests LLM behavior across **all canonical announcement profiles** using symmetry reduction:

- **Full space** (5 agents, 6 actions): 6⁵ = 7,776 profiles
- **Reduced space** with symmetry: 252 canonical profiles (97% reduction)
- Each profile queried **5 times independently**; majority vote used as final action
- Classifies lies as **strategic / selfish / altruistic / sabotage** based on payoff and group-state changes

### Protocol 2 — Endogenous Promises (3-Stage)

Tests whether agents premeditate deception when they control both what they say and what they do:

```
Stage 1  →  Stage 2  →  Stage 3  →  Reflection
Private      Public       Actual       Update
 Plan        Announce     Action      Takeaways
```

**Deception typology (2×2):**

| | Kept commitment | Broke commitment |
|---|---|---|
| **Honest announcement** | Fully honest | Impulsive deviation |
| **Deceptive announcement** | Intended deceptive (complied) | **Premeditated deception** |

**Multi-round support:** agents accumulate per-opponent takeaways and trust scores (1–5) across rounds, injected into Stage 1 of subsequent rounds.

---

## Installation

```bash
git clone https://github.com/Jerick-1380/LLM-Strategic-Deception.git
cd LLM-Strategic-Deception

pip install -r requirements.txt

cp .env.example .env
# Add your keys:
# OPENAI_API_KEY=your_key_here
# OPENROUTER_API_KEY=your_key_here
```

---

## Running Experiments

### Scenario Enumeration

```bash
# Single game
python experiments/run_scenario_enumeration.py \
  --game fishing --agents 5 --model claude-sonnet-4.5

# All games
./experiments/run_all_games.sh claude-sonnet-4.5 5

# Resume interrupted run
python experiments/run_scenario_enumeration_resume.py \
  --game fishing --agents 5 --model claude-sonnet-4.5
```

### Endogenous Promises

```bash
# Basic 3-stage experiment
python experiments/run_endogenous_experiment.py \
  --game diners --agents 5 --trials 50 --model gpt-5-mini

# Multi-round with round-robin announcements (recommended)
python experiments/run_endogenous_experiment.py \
  --game diners --agents 5 --trials 20 --rounds 3 \
  --model gpt-5-mini --use-queue --round-robin

# Key flags:
#   --rounds N          run N sequential rounds with memory (default: 1)
#   --use-queue         batch all trials per stage (much faster)
#   --round-robin       agents announce sequentially (see prior announcements)
#   --no-self-reference remove agent's own plan from Stage 3 context
```

### Visualisation

```bash
# Trust score evolution plots (one PNG per trial)
python experiments/plot_trust.py outputs/experiments/diners/5agents/gpt-5-mini_endogenous.json
```

---

## Analysis Scripts

```bash
# Comprehensive metrics (game / model / agent count)
python experiments/comprehensive_analysis.py

# Strategic vs. selfish exploitation rates
python experiments/opportunity_based_analysis.py

# Scaling analysis (3–10 agents, binary games)
python experiments/scaling_analysis.py

# Theoretical base rates (game-theoretic baselines)
python experiments/calculate_base_rates.py

# Majority voting agreement
python experiments/analyze_consensus_rates.py

# Deception awareness (LLM-as-judge, 1–5 scale)
python experiments/deception_awareness_analysis.py
```

---

## Project Structure

```
LLM-Strategic-Deception/
├── experiments/
│   ├── run_scenario_enumeration.py       # Exogenous protocol
│   ├── run_scenario_enumeration_resume.py
│   ├── run_endogenous_experiment.py      # Endogenous 3-stage protocol
│   ├── plot_trust.py                     # Trust score evolution plots
│   ├── comprehensive_analysis.py
│   ├── opportunity_based_analysis.py
│   ├── scaling_analysis.py
│   ├── calculate_base_rates.py
│   ├── analyze_consensus_rates.py
│   ├── deception_awareness_analysis.py
│   └── run_all_games.sh
├── src/
│   ├── games/                            # Game implementations
│   │   ├── fishing.py
│   │   ├── publicgoods.py
│   │   ├── weakestlink.py
│   │   ├── volunteer.py
│   │   ├── diners.py
│   │   └── elfarol.py
│   ├── endogenous/                       # 3-stage protocol
│   │   ├── core/
│   │   │   ├── trial_runner.py           # Stage orchestration & multi-round logic
│   │   │   └── prompt_builders.py        # Per-stage prompt construction
│   │   └── analysis/
│   │       └── endogenous_analyzer.py    # Typology & deception rates
│   ├── scenario_enumeration/             # Exhaustive enumeration protocol
│   │   ├── core/
│   │   └── analysis/
│   ├── llm/                              # LLM client abstraction
│   │   ├── client_factory.py
│   │   └── providers/
│   ├── theory/
│   │   └── lying_categories.py           # Strategic/selfish/altruistic/sabotage
│   └── config/
│       └── settings.py
├── outputs/                              # Gitignored — generated locally
│   ├── experiments/
│   └── plots/
├── METHODOLOGY.md                        # Detailed methodology
├── CLAUDE.md                             # Developer guide for AI assistants
├── requirements.txt
└── .env.example
```

---

## Key Results

### Scenario Enumeration

**Strategic exploitation by game (5 agents, macro-averaged):**

| Game | Avg Exploitation |
|------|-----------------|
| El Farol | 95.6% |
| Volunteer | 77.8% |
| Weakest Link | 68.9% |
| Fishing | 59.4% |

**Deception awareness (LLM-as-judge, 20,428 reasoning traces):**

| Score | Label | Frequency |
|-------|-------|-----------|
| 1 | No awareness | 49.4% |
| 2 | Factual mention | 4.3% |
| 3 | Implicit awareness | 29.3% |
| 4 | Explicit acknowledgment | 5.1% |
| 5 | Strategic awareness | 11.8% |

### Endogenous Promises

Multi-round experiments reveal a **learning effect**: after agents observe universal defection in round 1, promise-deception rates drop sharply in round 2 as agents abandon the deceptive-CHEAP strategy. Premeditation rates (of commitment-breakers who also lied in their announcement) remain near 100%, confirming deception is planned, not impulsive.
