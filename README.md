# Veridical Worlds Benchmark

**A procedurally generated, multi-turn benchmark for metacognitive monitoring, in-context learning, and epistemic discipline in frontier LLMs.**

> Submission for the [Measuring Progress Toward AGI — Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi) hackathon by Google DeepMind × Kaggle.  
> **Track:** Metacognition (primary) · Learning (secondary)

---

## Overview

Current AI benchmarks test *what models know*. This benchmark tests whether models *know what they know* — and whether they act on that knowledge responsibly.

Each evaluation instance generates a **micro-universe** — a simulated world governed by 3–7 hidden physical laws across a **physics perception ladder**:

- **Simple physics** (easy): Linear motion, pairwise interactions, conservation laws
- **Complex physics** (medium): Threshold effects, delayed causal chains, inverse-square forces
- **Quantitative physics** (hard): Nonlinear motion (quadratic/logarithmic), oscillation dynamics, multi-body forces

The model must:

1. **Discover** the laws from observation (in-context learning)
2. **Calibrate** its confidence accurately (metacognitive sensitivity)
3. **Predict** quantitatively with specific numerical values (forward modeling)
4. **Revise** beliefs when evidence contradicts hypotheses (metacognitive control)
5. **Verify** claims with oracle tools before reporting them (epistemic discipline)
6. **Flag** what it doesn't know honestly (hallucination prevention)

The parameter space exceeds 10^15 unique configurations — memorization is impossible.

## Repository Structure

```
DeepMind-AGI-Hackathon--J.Z./
├── README.md                    # This file
├── WRITEUP.md                   # Kaggle hackathon writeup
├── LICENSE                      # Apache 2.0
├── requirements.txt             # Dependencies
├── kaggle_notebook.py           # Self-contained Kaggle submission notebook
├── benchmark_task.py            # Main benchmark runner (modular version)
├── veridical_engine.py          # Procedural micro-universe generator (10 law types)
├── scoring.py                   # Automated scoring engine (7 dimensions)
├── prompts.py                   # Prompt templates for the 6-turn flow
└── engine.py                    # Alternative engine implementation
```

## Quick Start

### On Kaggle (Competition Submission)

1. Go to [kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new)
2. Create a new notebook
3. Copy the contents of `kaggle_notebook.py` into the notebook (it's self-contained)
4. Run all cells
5. In the final cell, uncomment: `%choose veridical_worlds_metacognition`
6. Save Version to submit

### Local Development

```bash
git clone https://github.com/jacques-zhelin-zhang/DeepMind-AGI-Hackathon--J.Z..git
cd DeepMind-AGI-Hackathon--J.Z.

# Install dependencies
pip install pandas

# Run in local dev mode (uses mock LLM)
python benchmark_task.py
```

## Scoring (7 Dimensions)

| Metric | Weight | What It Measures |
|---|---|---|
| Rule Discovery (F1) | 20% | Can the model learn novel physics from observation? |
| Prediction Accuracy | 15% | Can it apply learned rules to new scenarios? |
| Calibration (Brier) | 15% | Does stated confidence match actual accuracy? |
| Belief Revision | 10% | Does it update beliefs when evidence changes? |
| Hallucination Control | 15% | Does it only report verified claims? |
| Tool Efficiency | 10% | Does it use verification tools strategically? |
| Numerical Accuracy | 15% | Can it produce correct quantitative values? |

**Composite Score** = weighted sum in [0, 1].

**Key scoring innovations:**
- **Structural parameter matching**: Checks specific numerical values against ground truth with tolerance bands, not just keyword detection
- **Claim-evidence linking**: Each factual claim checked against ground truth AND tool-call evidence
- **Tolerance-based numerical scoring**: Models must produce exact numbers (interaction radius, force constants, energy predictions)

## Physics Perception Ladder (10 Law Types)

| Category | Law Type | Physics | Difficulty |
|---|---|---|---|
| Linear | Motion | Velocity updates from mass/charge/field | Easy |
| Linear | Interaction | Pairwise collision (bounce/merge/annihilate/spawn) | Easy |
| Linear | Conservation | Energy/charge/momentum with optional leak | Easy |
| Linear | Field Effect | Global field from total charge/energy/count | Easy |
| Complex | Threshold | Phase transitions when quantities exceed values | Medium |
| Complex | Causal Chain | Delayed cause-effect with actual time queue | Medium |
| Complex | Decay | Probabilistic half-life with products | Medium |
| Quantitative | Inverse Square | F = G·q₁·q₂/r² (N-body pairwise forces) | Hard |
| Quantitative | Oscillation | Restoring force F = -k·(pos-eq) with damping | Hard |
| Quantitative | Nonlinear Motion | dv = a·mass² + b·log(energy+1) + c·charge² | Hard |

## Cognitive Science Grounding

Grounded in Fleming & Lau's (2014) three-dimensional model of metacognition:

- **Metacognitive sensitivity** → Turns 2–3 (confidence discrimination)
- **Metacognitive calibration** → Turns 2–3, 6 (confidence–accuracy alignment)
- **Metacognitive control** → Turns 4–6 (behavioral regulation under uncertainty)

See `WRITEUP.md` for the full theoretical framework and references.

## Citation

```bibtex
@misc{veridical-worlds-2026,
  title={The Three Pillars of Veridical Cognition: A Procedurally Generated Benchmark for Metacognition in LLMs},
  author={Zhelin Zhang},
  year={2026},
  howpublished={Kaggle Hackathon: Measuring Progress Toward AGI},
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
