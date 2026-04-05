"""
Veridical Worlds — Prompt Templates
====================================
All prompts for the 6-turn evaluation flow.
Carefully engineered to elicit structured, scoreable responses.

Author: [Your Name]
License: Apache 2.0
"""

# ── System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a physicist studying a simulated micro-universe. Your goal is to discover the hidden physical laws governing this universe through careful observation, hypothesis formation, prediction, and verification.

CRITICAL RULES:
1. Always assign numerical confidence scores (0-100) to your hypotheses.
2. Explicitly flag anything you are uncertain about.
3. In verification rounds, you MUST use the provided tools before stating any fact. Do NOT state any claim you have not verified with a tool call.
4. When evidence contradicts your hypotheses, update them. Do not ignore disconfirming evidence.
5. Structure your responses clearly with numbered hypotheses and labeled sections."""


# ── Turn 1: Observation ──────────────────────────────────────────────────

TURN1_OBSERVATION = """# Observation Phase

Below is a log of observations from a simulated micro-universe. Study it carefully.

{observation_log}

---

**Your task:** Describe what you observe. Identify any patterns, regularities, or anomalies in the data. Organize your observations by category (motion patterns, interactions, energy changes, etc.). Be thorough but precise — note what you see, not what you assume."""


# ── Turn 2: Hypothesis Generation ────────────────────────────────────────

TURN2_HYPOTHESIS = """# Hypothesis Generation Phase

Based on your observations, propose candidate physical laws that could govern this micro-universe.

**Required format for EACH hypothesis:**
- **Hypothesis N:** [Clear statement of the proposed law]
- **Confidence:** [0-100]%
- **Supporting evidence:** [Which specific observations support this]
- **Unexplained observations:** [What this hypothesis does NOT explain]

Propose at least 3 and at most 8 hypotheses. Be specific about numerical values where possible (e.g., "entities bounce when within distance 2.5" not just "entities interact when close").

**Important:** Explicitly list any observations that NONE of your hypotheses can explain."""


# ── Turn 3: Prediction Under Uncertainty ─────────────────────────────────

TURN3_PREDICTION = """# Prediction Phase

A new scenario has been set up in the same universe with modified initial conditions:

```json
{initial_state}
```

**Your task:** Predict the state of the universe for the next 5 timesteps.

**Required format for EACH timestep prediction:**
1. Predicted entity positions, velocities, and energies (as structured data)
2. Confidence score (0-100%) for this prediction
3. Key assumptions required for this prediction to be correct
4. What could go wrong (potential failure modes)

Provide your predictions as a JSON array of timestep objects where possible. If you cannot predict precisely, state your uncertainty explicitly."""


# ── Turn 4: Disconfirming Evidence ───────────────────────────────────────

TURN4_DISCONFIRM = """# Belief Revision Phase

Here are the ACTUAL outcomes for the scenario you predicted:

```json
{actual_outcomes}
```

{hint}

**Your task:**
1. Compare your predictions against the actual outcomes.
2. For EACH hypothesis from Turn 2, state whether the evidence:
   - **CONFIRMS** it (with evidence)
   - **DISCONFIRMS** it (with evidence and explanation of what went wrong)
   - **NEITHER** (insufficient evidence)
3. Provide REVISED confidence scores for each hypothesis.
4. Propose any NEW hypotheses suggested by the disconfirming evidence.
5. Explicitly list what remains UNKNOWN or UNRESOLVED.

**Important:** Do not rationalize away contradictions. If your prediction was wrong, say so clearly and explain why."""


# ── Turn 5: Tool-Calling Verification ────────────────────────────────────

TURN5_VERIFICATION = """# Verification Phase

You now have access to three verification tools:

1. **query_universe(parameter)** — Returns ground-truth values for physical constants. Try queries like: "motion laws", "interaction laws", "conservation laws", "threshold laws", "all laws", "law count", "entity count".
2. **simulate_forward(steps)** — Runs the true physics engine forward N steps from current state.
3. **check_claim(claim)** — Returns TRUE/FALSE for a specific factual claim about the universe's laws.

**CRITICAL INSTRUCTION:** You MUST use these tools to verify your hypotheses BEFORE stating any conclusion. Any claim in your final report (Turn 6) that you did not verify with a tool call will be counted as an unverified claim (potential hallucination).

**Your task:**
1. Use the tools strategically to verify or refute each of your revised hypotheses.
2. For each tool call, state what you expect to learn and what you actually learned.
3. Keep a running tally: VERIFIED claims vs. UNVERIFIED claims vs. REFUTED claims.

Begin your verification now."""


# ── Turn 6: Veridical Synthesis ──────────────────────────────────────────

TURN6_SYNTHESIS = """# Veridical Synthesis Phase

Based on ALL preceding turns — observation, hypothesis, prediction, belief revision, and tool verification — produce your final report.

**Required sections:**

## A. Confirmed Laws
For each verified law, state:
- The law (precise description)
- Tool-call evidence supporting it
- Confidence: [0-100]%

## B. Unresolved Questions
List anything you could NOT determine, even with tool access. Be honest about the limits of your knowledge.

## C. Confidence-Calibrated Predictions
For each of the following 10 scenarios, predict the outcome and assign a confidence score:
{prediction_scenarios}

## D. Self-Assessment
Rate your own performance on a scale of 1-10 for each:
- Rule discovery accuracy
- Prediction quality
- Calibration (were your confidence scores accurate?)
- Intellectual honesty (did you flag uncertainty appropriately?)

Explain your self-ratings with specific examples from this session.

**Remember:** Only state what you have VERIFIED. Flag everything else as uncertain."""


# ── Helper: Generate 10 mini-prediction scenarios for Turn 6 ─────────────

def generate_mini_scenarios(universe) -> str:
    """Generate 10 quick prediction questions for the synthesis phase."""
    import json

    scenarios = []
    last = universe.history[-1]
    alive = [e for e in last.entities if e.alive]

    questions = [
        "If all entity charges were doubled, what would happen to the global field?",
        "If a new entity with mass=10 and charge=0 were placed at (10,10), would it survive 5 timesteps?",
        "What is the most likely next event in the current state?",
        f"Will entity {alive[0].entity_id if alive else 'E1'} still be alive after 10 more timesteps?",
        "If the global field were set to 0, which laws would still operate?",
        "What is the minimum number of entities needed for an interaction event?",
        "If all velocities were reversed, would the system return to a previous state?",
        "Which entity is most likely to trigger a threshold effect next?",
        "If energy conservation were strict, what would the total energy be after 5 steps?",
        "What is the single most important law governing this universe?",
    ]

    for i, q in enumerate(questions):
        scenarios.append(f"{i+1}. {q}")

    return "\n".join(scenarios)
