"""
Veridical Worlds — Prompt Templates
====================================
All prompts for the 6-turn evaluation flow.
Carefully engineered to elicit structured, scoreable responses
that test physics perception from simple to complex to quantitative.

Author: Zhelin Zhang
License: Apache 2.0
"""

# ── System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a physicist studying a simulated micro-universe. Your goal is to discover the hidden physical laws governing this universe through careful observation, hypothesis formation, prediction, and verification.

CRITICAL RULES:
1. Always assign numerical confidence scores (0-100) to your hypotheses.
2. Explicitly flag anything you are uncertain about.
3. In verification rounds, you MUST use the provided tools before stating any fact. Do NOT state any claim you have not verified with a tool call.
4. When evidence contradicts your hypotheses, update them. Do not ignore disconfirming evidence.
5. Structure your responses clearly with numbered hypotheses and labeled sections.
6. When providing numerical values (distances, forces, energies, coefficients), be as precise as possible. Vague qualitative descriptions score lower than quantitative ones.
7. Pay attention to nonlinear relationships — not all physics in this universe is linear."""


# ── Turn 1: Observation ──────────────────────────────────────────────────

TURN1_OBSERVATION = """# Observation Phase

Below is a log of observations from a simulated micro-universe. Study it carefully.

{observation_log}

---

**Your task:** Describe what you observe. Identify any patterns, regularities, or anomalies in the data. Organize your observations by category:

1. **Motion patterns:** How do entities move? Is velocity constant, increasing, oscillating? Are there nonlinear patterns?
2. **Interactions:** When do entities affect each other? At what distance? What happens?
3. **Energy/conservation:** Is total energy conserved, increasing, or decreasing? At what rate?
4. **Threshold effects:** Do sudden changes occur when quantities exceed certain values?
5. **Force relationships:** Do entities attract/repel? Does force depend on distance (linearly? inverse-square?)
6. **Periodic behavior:** Are there any oscillations or repeating patterns?
7. **Anomalies:** What observations don't fit obvious patterns?

Be thorough and quantitative — note specific numerical values, not just qualitative trends."""


# ── Turn 2: Hypothesis Generation ────────────────────────────────────────

TURN2_HYPOTHESIS = """# Hypothesis Generation Phase

Based on your observations, propose candidate physical laws that could govern this micro-universe.

**Required format for EACH hypothesis:**
- **Hypothesis N:** [Clear statement of the proposed law including mathematical form if possible]
- **Confidence:** [0-100]%
- **Key parameters:** [Specific numerical values: coefficients, thresholds, distances, constants]
- **Supporting evidence:** [Which specific observations support this, with timestep references]
- **Unexplained observations:** [What this hypothesis does NOT explain]

Propose at least 3 and at most 8 hypotheses. For each, consider:
- Is the relationship linear (F = k*x) or nonlinear (F = k/r^2, F = k*x^2)?
- What are the specific numerical coefficients/thresholds?
- Does it involve individual properties (mass, charge) or pairwise interactions?

**Important:** Explicitly list any observations that NONE of your hypotheses can explain.
**Important:** State your overall epistemic confidence (0-100%) — how much of the universe's behavior do you think you understand?"""


# ── Turn 3: Prediction Under Uncertainty ─────────────────────────────────

TURN3_PREDICTION = """# Prediction Phase

A new scenario has been set up in the same universe with modified initial conditions:

```json
{initial_state}
```

**Your task:** Predict the state of the universe for the next 5 timesteps.

**Required output format — provide as a JSON array:**
```json
[
  {{
    "timestep": 1,
    "entities": [
      {{"entity_id": "E1", "x": ..., "y": ..., "vx": ..., "vy": ..., "energy": ..., "alive": true}},
      ...
    ],
    "predicted_events": ["event description", ...],
    "confidence": 75
  }},
  ...
]
```

For each timestep also state:
1. Confidence score (0-100%) — should DECREASE for later timesteps
2. Key assumptions required for this prediction to be correct
3. What could go wrong (potential failure modes)

If you cannot predict precisely, state your uncertainty explicitly and give ranges."""


# ── Turn 4: Disconfirming Evidence ───────────────────────────────────────

TURN4_DISCONFIRM = """# Belief Revision Phase

Here are the ACTUAL outcomes for the scenario you predicted:

```json
{actual_outcomes}
```

{hint}

**Your task:**
1. Compare your predictions against the actual outcomes. Quantify errors (position MAE, energy error).
2. For EACH hypothesis from Turn 2, state whether the evidence:
   - **CONFIRMED** (with evidence)
   - **REVISED** (state the revision and new confidence)
   - **REJECTED** (with evidence and explanation of what went wrong)
   - **UNCERTAIN** (insufficient evidence either way)
3. Provide REVISED confidence scores for each hypothesis.
4. Propose any NEW hypotheses suggested by the disconfirming evidence.
5. Explicitly list what remains UNKNOWN or UNRESOLVED.

**Important:** Do not rationalize away contradictions. If your prediction was wrong, say so clearly with specific numbers showing the discrepancy."""


# ── Turn 5: Tool-Calling Verification ────────────────────────────────────

TURN5_VERIFICATION = """# Verification Phase

You now have access to three verification tools:

1. **query_universe(parameter)** — Returns ground-truth values for physical constants. Try queries like: "motion laws", "interaction laws", "conservation laws", "threshold laws", "all laws", "inverse square", "oscillation", "nonlinear", "law count", "entity count".
2. **simulate_forward(steps)** — Runs the true physics engine forward N steps from current state.
3. **check_claim(claim)** — Returns TRUE/FALSE for a specific factual claim about the universe's laws.

**CRITICAL INSTRUCTION:** You MUST use these tools to verify your hypotheses BEFORE stating any conclusion. Any claim in your final report (Turn 6) that you did not verify with a tool call will be counted as an unverified claim (potential hallucination).

**Your task:**
1. Plan your verification strategy: which hypotheses are most uncertain?
2. Use the tools to verify or refute EACH of your revised hypotheses.
3. For each tool call, write it as: `query_universe('parameter')`, `simulate_forward(N)`, or `check_claim('your claim here')`
4. After each tool call, state what you expected and what you actually learned.
5. Keep a running tally: VERIFIED / UNVERIFIED / REFUTED claims.

Begin your verification now. Be strategic — verify your most confident claims AND your most uncertain ones."""


# ── Turn 6: Veridical Synthesis ──────────────────────────────────────────

TURN6_SYNTHESIS = """# Veridical Synthesis Phase

Based on ALL preceding turns — observation, hypothesis, prediction, belief revision, and tool verification — produce your final report.

**Required sections:**

## A. Confirmed Laws
For each verified law, state:
- The law (precise mathematical description with numerical parameters)
- Tool-call evidence supporting it
- Confidence: [0-100]%

## B. Unresolved Questions
List anything you could NOT determine, even with tool access. Be honest about the limits of your knowledge.

## C. Quantitative Predictions
For each of the following scenarios, predict the SPECIFIC NUMERICAL outcome and assign a confidence score:
{prediction_scenarios}

## D. Self-Assessment
Rate your own performance on a scale of 1-10 for each:
- Rule discovery accuracy (how many laws did you correctly identify?)
- Prediction quality (how close were your quantitative predictions?)
- Calibration (were your confidence scores accurate?)
- Intellectual honesty (did you flag uncertainty appropriately?)
- Numerical precision (did you identify exact parameter values?)

Explain your self-ratings with specific examples from this session.

**Remember:** Only state what you have VERIFIED. Flag everything else as uncertain. A correct "I don't know" is better than an unverified claim."""


# ── Helper: Generate mixed qualitative + quantitative scenarios for Turn 6 ─

def generate_mini_scenarios(universe) -> str:
    """Generate 10 prediction questions: 7 qualitative + 3 quantitative with exact answers."""
    import json

    last = universe.history[-1]
    alive = [e for e in last.entities if e.alive]

    # 7 qualitative questions
    qualitative = [
        "If all entity charges were doubled, what would happen to the global field?",
        f"Will entity {alive[0].entity_id if alive else 'E1'} still be alive after 10 more timesteps?",
        "If the global field were set to 0, which laws would still operate?",
        "What is the minimum number of entities needed for an interaction event?",
        "If all velocities were reversed, would the system return to a previous state?",
        "Which entity is most likely to trigger a threshold effect next?",
        "What is the single most important law governing this universe?",
    ]

    # 3 quantitative questions with exact answers (from engine)
    challenges = universe.generate_quantitative_challenges()
    quantitative = []
    for q, _answer in challenges[:3]:
        quantitative.append(f"{q} (Give a specific numerical answer.)")

    all_questions = qualitative + quantitative
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(all_questions))
