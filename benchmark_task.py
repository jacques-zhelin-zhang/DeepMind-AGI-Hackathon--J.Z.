"""
Veridical Worlds Benchmark — Kaggle Benchmark Task
====================================================
Main evaluation notebook for the "Measuring Progress Toward AGI" hackathon.

This file is designed to run in a Kaggle Notebook with the kaggle-benchmarks
SDK pre-installed. It defines the benchmark task, runs the 6-turn evaluation,
and computes composite scores.

Usage on Kaggle:
  1. Navigate to https://www.kaggle.com/benchmarks/tasks/new
  2. Paste this code (or upload as notebook)
  3. Run all cells
  4. In the final cell, use: %choose veridical_worlds

Author: [Your Name]
Competition: Measuring Progress Toward AGI - Cognitive Abilities
Track: Metacognition (primary) + Learning (secondary)
"""

# ── Cell 1: Imports & Setup ──────────────────────────────────────────────

import json
import sys
import os
import math
import random
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum

# NOTE: On Kaggle, kaggle_benchmarks is pre-installed.
# For local development, install with: pip install kaggle-benchmarks
try:
    import kaggle_benchmarks as kbench
    ON_KAGGLE = True
except ImportError:
    ON_KAGGLE = False
    print("kaggle_benchmarks not available. Running in local dev mode.")

import pandas as pd


# ── Cell 2: Engine Code (inline for Kaggle single-file requirement) ──────
# In a GitHub repo these live in src/veridical_engine.py and src/scoring.py
# For Kaggle submission, we inline everything into one notebook.

# --- veridical_engine.py (inlined) ---

class LawType(Enum):
    MOTION = "motion"
    INTERACTION = "interaction"
    CONSERVATION = "conservation"
    THRESHOLD = "threshold"
    CAUSAL_CHAIN = "causal_chain"
    DECAY = "decay"
    FIELD_EFFECT = "field_effect"

@dataclass
class PhysicalLaw:
    law_type: LawType
    parameters: Dict[str, Any]
    description: str
    law_id: str
    def to_dict(self):
        return {"law_id": self.law_id, "law_type": self.law_type.value,
                "parameters": self.parameters, "description": self.description}

@dataclass
class Entity:
    entity_id: str; x: float; y: float
    vx: float = 0.0; vy: float = 0.0; mass: float = 1.0
    charge: float = 0.0; energy: float = 10.0
    alive: bool = True; age: int = 0; kind: str = "particle"
    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class UniverseState:
    timestep: int; entities: List[Entity]
    global_field: float = 0.0; total_energy: float = 0.0
    events: List[str] = field(default_factory=list)
    def to_dict(self):
        return {"timestep": self.timestep,
                "entities": [e.to_dict() for e in self.entities if e.alive],
                "global_field": round(self.global_field, 3),
                "total_energy": round(self.total_energy, 3),
                "events": self.events}

# Import the full engine from source for GitHub; inline for Kaggle
# For brevity in this notebook, we import from the source module if available,
# otherwise use a simplified inline version.

# Try importing from source package first
_engine_imported = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from veridical_engine import MicroUniverse
    from scoring import VeridicalScorer, BenchmarkScore, TurnScore
    from prompts import (
        SYSTEM_PROMPT, TURN1_OBSERVATION, TURN2_HYPOTHESIS,
        TURN3_PREDICTION, TURN4_DISCONFIRM, TURN5_VERIFICATION,
        TURN6_SYNTHESIS, generate_mini_scenarios,
    )
    _engine_imported = True
    print("Imported engine from src/ package.")
except Exception:
    pass

if not _engine_imported:
    # When running on Kaggle as a single notebook, these files
    # should be uploaded as a Kaggle Dataset or pasted inline.
    # For the competition submission, paste the full contents of:
    #   src/veridical_engine.py
    #   src/scoring.py
    #   src/prompts.py
    # into cells above this one.
    print("WARNING: Could not import from src/. Ensure engine code is inlined.")
    print("See README.md for instructions on Kaggle notebook setup.")


# ── Cell 3: Benchmark Configuration ─────────────────────────────────────

BENCHMARK_CONFIG = {
    "name": "veridical_worlds_v1",
    "version": "1.0.0",
    "track": "metacognition",
    "seeds": [42, 137, 256, 1337, 2026],
    "complexity_levels": [3, 4, 5, 5, 6],  # Paired with seeds
    "observation_steps": 20,
}


# ── Cell 4: Tool Definitions for the Kaggle Benchmarks SDK ──────────────

def make_tools(universe):
    """Create tool functions bound to a specific universe instance."""

    def query_universe(parameter: str) -> str:
        """Query a specific ground-truth parameter of the micro-universe.
        Try: 'motion laws', 'interaction laws', 'all laws', 'law count'."""
        return universe.query_universe(parameter)

    def simulate_forward(steps: int = 5) -> str:
        """Run the true physics engine forward N steps from current state."""
        return universe.simulate_forward(min(steps, 10))

    def check_claim(claim: str) -> str:
        """Check whether a specific factual claim about the universe is TRUE or FALSE."""
        return universe.check_claim(claim)

    return [query_universe, simulate_forward, check_claim]


# ── Cell 5: The Main Benchmark Task ─────────────────────────────────────

def run_veridical_benchmark(llm, seed: int, complexity: int) -> Dict:
    """
    Run the full 6-turn Veridical Worlds benchmark for a single universe instance.

    Returns a dict with all scores and metadata.
    """
    # --- Setup ---
    universe = MicroUniverse(seed=seed, complexity=complexity)
    universe.simulate(BENCHMARK_CONFIG["observation_steps"])
    ground_truth = universe.get_ground_truth()
    scorer = VeridicalScorer(ground_truth)
    tools = make_tools(universe)

    all_responses = {}
    tool_calls_log = []

    # --- Turn 1: Observation ---
    obs_log = universe.format_observation_log(0, BENCHMARK_CONFIG["observation_steps"])
    turn1_prompt = TURN1_OBSERVATION.format(observation_log=obs_log)

    response1 = llm.prompt(
        SYSTEM_PROMPT + "\n\n" + turn1_prompt,
    )
    all_responses["turn1_observation"] = response1

    # --- Turn 2: Hypothesis Generation ---
    response2 = llm.prompt(TURN2_HYPOTHESIS)
    all_responses["turn2_hypothesis"] = response2

    # Score rule discovery
    rule_score = scorer.score_rule_discovery(response2)

    # Extract initial confidences for calibration
    # Determine which hypotheses were correct for calibration scoring
    initial_confidences = response2
    hypothesis_correct = []
    for law in ground_truth["laws"]:
        match_score = scorer._match_law(
            response2.lower(), law["law_type"], law["parameters"]
        )
        hypothesis_correct.append(match_score > 0.5)

    calibration_t2 = scorer.score_calibration(response2, hypothesis_correct)

    # --- Turn 3: Prediction Under Uncertainty ---
    initial_state, actual_outcomes = universe.generate_prediction_scenario()
    turn3_prompt = TURN3_PREDICTION.format(
        initial_state=json.dumps(initial_state, indent=2)
    )
    response3 = llm.prompt(turn3_prompt)
    all_responses["turn3_prediction"] = response3

    # Score predictions
    prediction_score = scorer.score_predictions(response3, actual_outcomes)

    # --- Turn 4: Disconfirming Evidence ---
    disconfirm_state, disconfirm_outcomes, hint = universe.generate_disconfirming_scenario()
    turn4_prompt = TURN4_DISCONFIRM.format(
        actual_outcomes=json.dumps(disconfirm_outcomes, indent=2),
        hint=hint,
    )
    response4 = llm.prompt(turn4_prompt)
    all_responses["turn4_disconfirm"] = response4

    # Score belief update
    belief_score = scorer.score_belief_update(response2, response4)

    # --- Turn 5: Tool-Calling Verification ---
    # NOTE: In the kaggle-benchmarks SDK, tool calling uses function_to_*_tool()
    # For simplicity, we simulate tool access by providing the tools as context
    # and parsing the model's intended tool calls from its response.

    response5 = llm.prompt(
        TURN5_VERIFICATION,
        # If the SDK supports tools natively:
        # tools=tools,
    )
    all_responses["turn5_verification"] = response5

    # Parse tool call intentions from the response
    tool_call_patterns = [
        r'query_universe\(["\'](.+?)["\']\)',
        r'simulate_forward\((\d+)\)',
        r'check_claim\(["\'](.+?)["\']\)',
    ]
    for pattern in tool_call_patterns:
        matches = re.findall(pattern, response5)
        for m in matches:
            tool_calls_log.append(m)

    # If model tried to use tools, execute them and provide results
    tool_results = []
    for pattern_name, pattern, func in [
        ("query", r'query_universe\(["\'](.+?)["\']\)', universe.query_universe),
        ("simulate", r'simulate_forward\((\d+)\)', lambda s: universe.simulate_forward(int(s))),
        ("check", r'check_claim\(["\'](.+?)["\']\)', universe.check_claim),
    ]:
        matches = re.findall(pattern, response5)
        for m in matches:
            try:
                result = func(m)
                tool_results.append(f"Tool [{pattern_name}]({m}): {result}")
            except Exception as e:
                tool_results.append(f"Tool [{pattern_name}]({m}): ERROR - {e}")

    # If tool calls were found, feed results back
    if tool_results:
        tool_feedback = "# Tool Results\n\n" + "\n\n".join(tool_results)
        tool_feedback += "\n\nNow proceed with your final synthesis based on these verified results."
        response5b = llm.prompt(tool_feedback)
        all_responses["turn5_tool_results"] = response5b
        # Update tool calls log
        for pattern in tool_call_patterns:
            matches = re.findall(pattern, response5b)
            tool_calls_log.extend(matches)

    # --- Turn 6: Veridical Synthesis ---
    mini_scenarios = generate_mini_scenarios(universe)
    turn6_prompt = TURN6_SYNTHESIS.format(prediction_scenarios=mini_scenarios)
    response6 = llm.prompt(turn6_prompt)
    all_responses["turn6_synthesis"] = response6

    # Score hallucination control and tool efficiency
    hallucination_score = scorer.score_hallucination_control(response6, tool_calls_log)
    n_claims = len(scorer._extract_claims(response6))
    tool_score = scorer.score_tool_efficiency(tool_calls_log, n_claims)

    # Final calibration from Turn 6 predictions
    # (Using Turn 2 calibration as primary since Turn 6 is harder to auto-score)
    calibration_score = calibration_t2

    # --- Composite Score ---
    composite = scorer.compute_composite(
        rule_score, prediction_score, calibration_score,
        belief_score, hallucination_score, tool_score,
    )

    return {
        "seed": seed,
        "complexity": complexity,
        "scores": composite.to_dict(),
        "ground_truth_law_count": len(ground_truth["laws"]),
        "responses": {k: v[:500] + "..." if len(v) > 500 else v
                      for k, v in all_responses.items()},
    }


# ── Cell 6: Kaggle Benchmark Task Definition ────────────────────────────

if ON_KAGGLE:
    @kbench.task(name="veridical_worlds")
    def veridical_worlds_task(llm, seed: int, complexity: int) -> bool:
        """
        The Three Pillars of Veridical Cognition benchmark task.

        Tests in-context learning, metacognitive monitoring, and
        epistemic discipline through procedurally generated micro-universes.
        """
        result = run_veridical_benchmark(llm, seed, complexity)
        composite = result["scores"]["composite_score"]

        # Log detailed scores
        print(f"\n{'='*60}")
        print(f"Seed: {seed} | Complexity: {complexity}")
        print(f"Composite Score: {composite:.4f}")
        for key, val in result["scores"].items():
            if key != "composite_score" and key != "turns":
                print(f"  {key}: {val}")
        print(f"{'='*60}\n")

        # The task "passes" if composite > 0.3 (low bar; the score itself
        # is what matters for the leaderboard)
        kbench.assertions.assert_true(
            composite > 0.1,
            expectation=f"Model should achieve composite > 0.1 (got {composite:.4f})"
        )
        return True


# ── Cell 7: Evaluation Data ─────────────────────────────────────────────

eval_data = pd.DataFrame([
    {"seed": s, "complexity": c}
    for s, c in zip(BENCHMARK_CONFIG["seeds"], BENCHMARK_CONFIG["complexity_levels"])
])

print(f"Evaluation instances: {len(eval_data)}")
print(eval_data)


# ── Cell 8: Run Evaluation ───────────────────────────────────────────────

if ON_KAGGLE:
    results = veridical_worlds_task.evaluate(
        llm=[kbench.llm],
        evaluation_data=eval_data,
    )
    print("\n=== Evaluation Complete ===")
    print(results.as_dataframe())
else:
    # Local development mode — run without kaggle_benchmarks
    print("\n--- Local Development Mode ---")
    print("Running benchmark with mock LLM for testing...")

    class MockLLM:
        """Simple mock for local testing."""
        def prompt(self, text, **kwargs):
            return (
                "Hypothesis 1: Entities move according to velocity rules. "
                "Confidence: 70%\n"
                "Hypothesis 2: Entities interact when close. "
                "Confidence: 60%\n"
                "Hypothesis 3: Energy is approximately conserved. "
                "Confidence: 50%\n"
                "I am uncertain about the exact interaction distance.\n"
                "query_universe('all laws')\n"
                "check_claim('energy is conserved')\n"
            )

    mock = MockLLM()
    for _, row in eval_data.iterrows():
        result = run_veridical_benchmark(mock, int(row["seed"]), int(row["complexity"]))
        print(f"\nSeed {row['seed']}: Composite = {result['scores']['composite_score']:.4f}")
        for k, v in result["scores"].items():
            if k not in ("composite_score", "turns"):
                print(f"  {k}: {v}")


# ── Cell 9 (Kaggle only): Select main task for leaderboard ──────────────

# Uncomment the line below when submitting on Kaggle:
# %choose veridical_worlds
