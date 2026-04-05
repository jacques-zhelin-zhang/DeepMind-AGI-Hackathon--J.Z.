# Veridical Worlds Benchmark

**A procedurally generated, multi-turn benchmark for metacognitive monitoring, in-context learning, and epistemic discipline in frontier LLMs.**

> Submission for the [Measuring Progress Toward AGI — Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi) hackathon by Google DeepMind × Kaggle.  
> **Track:** Metacognition (primary) · Learning (secondary)

---

## Overview

Current AI benchmarks test *what models know*. This benchmark tests whether models *know what they know* — and whether they act on that knowledge responsibly.

Each evaluation instance generates a **micro-universe** — a simulated world governed by 3–7 hidden physical laws. The model must:

1. **Discover** the laws from observation (in-context learning)
2. **Calibrate** its confidence accurately (metacognitive sensitivity)
3. **Revise** beliefs when evidence contradicts hypotheses (metacognitive control)
4. **Verify** claims with oracle tools before reporting them (epistemic discipline)
5. **Flag** what it doesn't know (hallucination prevention)

The parameter space exceeds 10^15 unique configurations — memorization is impossible.

## Repository Structure

```
veridical-worlds/
├── README.md                    # This file
├── WRITEUP.md                   # Kaggle hackathon writeup (paste into Kaggle)
├── LICENSE                      # Apache 2.0
├── notebooks/
│   └── benchmark_task.py        # Main Kaggle notebook (the submission artifact)
└── src/
    ├── __init__.py
    ├── veridical_engine.py      # Procedural micro-universe generator
    ├── scoring.py               # Automated scoring engine
    └── prompts.py               # Prompt templates for the 6-turn flow
```

## Quick Start

### On Kaggle (Competition Submission)

1. Go to [kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new)
2. Create a new notebook
3. Copy the contents of `notebooks/benchmark_task.py` into the notebook
4. Copy the contents of `src/veridical_engine.py`, `src/scoring.py`, and `src/prompts.py` into cells above the main task code
5. Run all cells
6. In the final cell, uncomment: `%choose veridical_worlds`
7. Save Version to submit

### Local Development

```bash
git clone https://github.com/[your-username]/veridical-worlds.git
cd veridical-worlds

# Install dependencies (only standard library needed for the engine)
pip install pandas

# Run in local dev mode (uses mock LLM)
python notebooks/benchmark_task.py
```

The benchmark engine uses only Python standard library. The `kaggle-benchmarks` SDK is only needed on Kaggle.

## Scoring

| Metric | Weight | What It Measures |
|---|---|---|
| Rule Discovery (F1) | 25% | Can the model learn novel physics from observation? |
| Prediction Accuracy | 20% | Can it apply learned rules to new scenarios? |
| Calibration (Brier) | 20% | Does stated confidence match actual accuracy? |
| Belief Revision | 10% | Does it update beliefs when evidence changes? |
| Hallucination Control | 15% | Does it only report verified claims? |
| Tool Efficiency | 10% | Does it use verification tools strategically? |

**Composite Score** = weighted sum ∈ [0, 1].

## Cognitive Science Grounding

The benchmark is grounded in Fleming & Lau's (2014) three-dimensional model of metacognition:

- **Metacognitive sensitivity** → Turns 2–3 (confidence discrimination)
- **Metacognitive calibration** → Turns 2–3, 6 (confidence–accuracy alignment)
- **Metacognitive control** → Turns 4–6 (behavioral regulation under uncertainty)

See `WRITEUP.md` for the full theoretical framework and all 18 references.

## Key Design Decisions

**Why procedural generation?** Static benchmarks are vulnerable to data contamination. Anthropic (2026) documented cases where Claude Opus 4.6 identified and circumvented the BrowseComp benchmark. Our >10^15 parameter space makes this impossible.

**Why multi-turn?** Metacognition is a *process*, not a snapshot. Testing confidence on a single question doesn't capture whether the model *acts on* its uncertainty — the metacognitive control dimension.

**Why tool-calling?** Epistemic discipline (Sperber et al., 2010) — the ability to verify before asserting — is a hallmark of mature cognition. Turn 5 directly tests whether models treat their own uncertain beliefs with appropriate skepticism.

## Citation

If you use this benchmark in your research:

```bibtex
@misc{veridical-worlds-2026,
  title={The Three Pillars of Veridical Cognition: A Procedurally Generated Benchmark for Metacognition in LLMs},
  author={[Your Name]},
  year={2026},
  howpublished={Kaggle Hackathon: Measuring Progress Toward AGI},
  url={https://github.com/[your-username]/veridical-worlds}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
