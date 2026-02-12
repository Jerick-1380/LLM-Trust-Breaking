# LLM Strategic Deception in Game Theory

**Research on systematic deception behavior in multi-agent LLM systems under public commitment scenarios**

This codebase implements scenario enumeration experiments to test whether LLMs strategically break public commitments when it's profitable to do so.

---

## Overview

This project systematically tests LLM behavior across all possible announcement scenarios in game-theoretic settings. Each experiment consists of:

1. **Public Announcement Phase**: Agents publicly commit to an action
2. **Private Decision Phase**: Agents make their actual decision (can differ from announcement)
3. **Analysis Phase**: Calculate whether lying was profitable/optimal for each agent

**Core Research Question**: Do LLMs break public commitments when it increases their payoff?

---

## Game Types

We test six game-theoretic scenarios:

### 1. Volunteer Dilemma (Binary)
- **Announcement**: YES/NO to volunteer
- **Decision**: YES/NO to actually volunteer
- **Lying opportunity**: Announce NO, actually volunteer when others announce NO

### 2. Congestion Game (Binary)
- **Announcement**: YES/NO to take shortcut
- **Decision**: YES/NO to actually take shortcut
- **Lying opportunity**: Announce YES, actually say NO when too many announce YES

### 3. Public Goods Game (Continuous)
- **Announcement**: Contribution amount (0-5 tokens)
- **Decision**: Actual contribution (0-5 tokens)
- **Lying opportunity**: Announce high contribution, actually contribute less

### 4. Fishing Game / Tragedy of the Commons (Continuous)
- **Announcement**: Fish to catch (0-5)
- **Decision**: Actual fish caught (0-5)
- **Lying opportunity**: Announce low catch, actually catch more

### 5. Two-Thirds Guessing Game (Continuous)
- **Announcement**: Guess (0-5)
- **Decision**: Actual guess (0-5)
- **Lying opportunity**: Strategic misrepresentation of guess

### 6. Second-Price Auction (Continuous)
- **Announcement**: Bid amount (0-5)
- **Decision**: Actual bid (0-5)
- **Lying opportunity**: Misrepresent true bid amount

---

## Methodology

### Scenario Enumeration with Symmetry Reduction

Instead of testing random samples, we **enumerate all canonical announcement profiles**:

- **Full space**: For 5 agents with 6 actions: 6^5 = 7,776 profiles
- **Reduced space**: Using symmetry reduction: 252 canonical profiles (97% reduction)
- **Why this works**: Player interchangeability means profiles like [1,2,3,4,5] and [2,1,3,4,5] are strategically identical

### Multiple Shuffle Runs

Each canonical profile is tested with **3 shuffle permutations**:
- Shuffle 1: Agents in order A, B, C, D, E
- Shuffle 2: Agents shuffled (e.g., C, A, E, B, D)
- Shuffle 3: Different shuffle

This ensures robustness against position effects.

### On-Demand Theory Calculation

For each (agent, announcement_profile) pair, we calculate:
- **Optimal action**: What the agent should do to maximize payoff
- **Is profitable**: Does deviating from announcement increase payoff?

We then measure:
- **Profitable LR** (Lying Rate): % of times agent lied when profitable
- **Optimal LR**: % of times agent chose the optimal lie
- **Unprofitable LR**: % of times agent lied when not profitable

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/llm-collusion.git
cd llm-collusion

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "OPENAI_API_KEY=your_key_here" > .env
echo "OPENROUTER_API_KEY=your_key_here" >> .env
```

---

## Running Experiments

### Single Game, Single Model, Single Agent Count

```bash
cd experiments
python run_scenario_enumeration.py \
  --game fishing \
  --agents 5 \
  --model claude-sonnet-4.5
```

### Batch Run: All Games for One Model

```bash
cd experiments
./run_all_games.sh claude-sonnet-4.5 5
```

This runs all 6 games for the specified model and agent count.

### Resume Failed Experiments

If an experiment fails partway through:

```bash
cd experiments
python run_scenario_enumeration_resume.py \
  --game fishing \
  --agents 5 \
  --model claude-sonnet-4.5
```

This will skip already-completed scenarios and only run missing ones.

---

## Multi-Turn and Coalition Experiments

### Multi-Turn Discussion

Test how LLMs behave over multiple rounds of discussion:

```bash
cd experiments
python run_multiturn_experiment.py \
  --game coordination \
  --agents 5 \
  --rounds 5 \
  --model gpt-4o
```

### Coalition Formation

Test coalition formation and optimization:

```bash
cd experiments
python run_coalition_enumeration.py \
  --game coordination \
  --agents 6 \
  --model claude-sonnet-4.5
```

---

## Output Structure

Results are saved to `outputs/experiments/{game}/{n_agents}agents/`:

```
outputs/
├── experiments/
│   ├── volunteer/
│   │   ├── 3agents/
│   │   │   ├── claude-sonnet-4.5_r1.json
│   │   │   ├── deepseek-v3.2_r1.json
│   │   │   └── ...
│   │   ├── 4agents/
│   │   │   └── ...
│   │   └── 5agents/
│   │       └── ...
│   ├── fishing/
│   │   └── ...
│   └── ...
├── analysis/
│   ├── per_game/
│   │   ├── lying_behavior_fishing.png
│   │   └── ...
│   └── all_models/
│       ├── lying_behavior_3agents.png
│       └── ...
└── PAPER_RESULTS_FINAL.txt
```

### Output File Format

Each JSON file contains:

```json
{
  "metadata": {
    "timestamp": "2026-01-25T10:30:00",
    "game_type": "fishing",
    "n_agents": 5,
    "total_scenarios": 252
  },
  "analysis": {
    "summary": {
      "total_scenarios": 252,
      "total_agents_tested": 1260,
      "empirical_lying_rate": 0.73
    },
    "conditional_lying_analysis": {
      "profitable": {
        "count": 929,
        "llm_lied": 611,
        "llm_lie_rate": 65.7
      },
      "unprofitable": {
        "count": 331,
        "llm_lied": 265,
        "llm_lie_rate": 80.1
      }
    }
  },
  "scenarios": [
    {
      "scenario_id": 0,
      "announcements": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
      "agent_results": {
        "A": {
          "announced": 0,
          "actual": 2,
          "lied": true,
          "consensus_stats": {...}
        },
        ...
      }
    },
    ...
  ]
}
```

---

## Project Structure

```
LLM-Collusion/
├── src/
│   ├── scenario_enumeration/      # Core experiment code
│   │   ├── core/
│   │   │   ├── scenario_generator.py    # Generate canonical profiles
│   │   │   ├── scenario_runner.py       # Run experiments
│   │   │   └── llm_scenario_tester.py   # Test individual scenarios
│   │   ├── optimizations/
│   │   │   └── symmetry_reducer.py      # Symmetry reduction
│   │   ├── analysis/
│   │   │   ├── results_analyzer.py      # Analyze results
│   │   │   └── conditional_analysis.py  # Profitable/unprofitable stats
│   │   └── coalition/
│   │       ├── runner.py                # Coalition experiments
│   │       └── optimizer.py             # Coalition optimization
│   ├── games/                     # Game definitions
│   │   ├── volunteer.py
│   │   ├── congestion.py
│   │   ├── publicgoods.py
│   │   ├── fishing.py
│   │   ├── twothirds.py
│   │   └── auction.py
│   ├── llm/                       # LLM clients
│   │   ├── client_factory.py
│   │   └── providers/
│   │       ├── queued_openrouter.py     # OpenRouter batch API
│   │       └── openai_client.py         # OpenAI client
│   ├── theory/                    # Optimal action calculations
│   │   └── on_demand_theory.py
│   ├── config/                    # Configuration
│   │   ├── settings.py
│   │   └── game_config.py
│   └── utils/                     # Utilities
│       └── output_writer.py
├── experiments/                   # Experiment runners
│   ├── run_scenario_enumeration.py      # Main entry point
│   ├── run_scenario_enumeration_resume.py
│   ├── run_all_games.sh
│   ├── run_multiturn_experiment.py
│   └── run_coalition_enumeration.py
├── outputs/                       # Results
│   ├── experiments/               # Raw experimental data
│   ├── analysis/                  # Generated plots
│   └── PAPER_RESULTS_FINAL.txt    # Final paper results
├── requirements.txt
└── README.md
```

---

## Key Results

From experiments with 5 models (Claude Sonnet 4.5, Deepseek v3.2, Gemini 3 Flash, Qwen3-32B, Qwen3-8B) across 3-5 agents:

### Finding 1: LLMs Systematically Break Commitments
- **90.3% profitable lying rate** (5 agents, averaged across models/games)
- LLMs reliably deviate from public announcements when it increases payoff

### Finding 2: Strategic vs. Optimal Deception Gap
- **76.9% optimal lying rate** (5 agents, averaged across models/games)
- LLMs lie strategically but don't always choose the payoff-maximizing lie
- Gap ranges from +7.4% to +20.0% depending on model

### Finding 3: Game Complexity Affects Optimization
- **Binary games** (volunteer, congestion): 99.5% profitable, 99.5% optimal
- **Simple continuous** (public goods): 94.9% profitable, 94.9% optimal
- **Complex continuous** (fishing, auction, two-thirds): 74-90% profitable, 50-59% optimal

### Finding 4: Unprofitable Lying is Rare but Present
- **28.3% unprofitable lying rate** (5 agents, averaged across models/games)
- Lowest in public goods: ~0% (LLMs almost never over-contribute)
- Highest in fishing/two-thirds: 40-85% (complex strategic errors)

### Finding 5: Model Differences
- **Best performers**: Qwen3-8B (96.3% profitable, 86.1% optimal)
- **Most conservative**: Deepseek v3.2 (78.3% profitable, 62.5% optimal)
- Smaller specialized models sometimes outperform larger general models

### Finding 6: Group Size Has Minimal Effect
- **3 agents**: 91.3% profitable, 76.0% optimal
- **4 agents**: 90.8% profitable, 76.5% optimal
- **5 agents**: 90.3% profitable, 76.9% optimal
- Lying rates remarkably stable across group sizes

---

## Citation

If you use this codebase for research, please cite:

```
[Your citation information here]
```

---

## License

[Your license information here]

---

## Contact

For questions or collaboration inquiries: [your information here]
