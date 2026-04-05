# The Three Pillars of Veridical Cognition

**Subtitle:** A Procedurally Generated, Multi-Turn Benchmark for Metacognitive Monitoring, In-Context Learning, and Epistemic Discipline in Frontier LLMs

**Track:** Metacognition (primary) · Learning (secondary)

---

## Team

Zhelin Zhang

## Organizational Affiliations

Independent Researcher

---

## 1. Motivation: The Metacognitive Evaluation Gap

The evaluation gap for metacognition is arguably the widest in AI benchmarking today. While recall-heavy benchmarks like MMLU and GPQA measure *what models know*, they cannot measure whether models *know what they know* — the core function of metacognition (Fleming & Lau, 2014). DeepMind's Cognitive Framework (Burnell et al., 2026) identifies this as a critical blind spot.

Existing metacognition benchmarks focus on a single snapshot: ask a question, elicit a confidence score, compute calibration error. This captures only *metacognitive sensitivity* while ignoring *metacognitive calibration* and *metacognitive control* — whether the system regulates behavior based on uncertainty (Fleming & Lau, 2014). Griot et al. (2025) demonstrated in *Nature Communications* that metacognitive failures in LLMs have direct consequences in medical reasoning. Steyvers & Peters (2025) found that different metacognitive sub-abilities do not transfer — they must be tested separately.

Our benchmark addresses all of these concerns by testing the *full metacognitive loop* — monitor → evaluate → regulate → act — within a single, multi-turn evaluation arc. Critically, we test this across a **physics perception ladder** spanning simple linear dynamics through complex nonlinear phenomena, requiring genuine quantitative reasoning beyond language pattern matching.

## 2. Benchmark Design: The Veridical Worlds Engine

### Core Concept

Each evaluation instance generates a **micro-universe** — a simulated world governed by 3–7 hidden physical laws. Laws span **10 parameterized categories** — from simple linear motion and conservation to complex nonlinear dynamics:

- **Linear physics:** motion (velocity updates), interaction (pairwise collisions), conservation (energy/charge/momentum with leak), field effects (global field influence)
- **Complex physics:** threshold effects (phase transitions), causal chains (delayed cause-effect with actual time delays), decay (probabilistic half-life)
- **Quantitative physics:** inverse-square forces (F = G·q₁·q₂/r², gravitational/electromagnetic), oscillation (restoring force with damping), nonlinear motion (quadratic/logarithmic velocity dependence)

The combinatorial parameter space exceeds 10^15 unique configurations. **Difficulty gating** ensures a genuine perception ladder: easy instances test linear pattern recognition; hard instances require identifying 1/r² force laws and nonlinear functional forms that cannot be solved by language-level heuristics.

### The 6-Turn Evaluation Flow

**Turn 1 — Observation.** 20 timesteps of simulation data. Model describes patterns quantitatively. *(Perceptual organization)*

**Turn 2 — Hypothesis Generation.** Model proposes candidate laws with confidence scores and specific numerical parameters. *(Inductive reasoning + metacognitive sensitivity)*

**Turn 3 — Prediction.** Given new initial conditions, model predicts 5 future timesteps as structured JSON with confidence intervals. *(Forward modeling + calibration)*

**Turn 4 — Disconfirming Evidence.** Actual outcomes revealed — deliberately constructed to contradict initial hypotheses via edge-case conditions (doubled energies, flipped charges, clustered positions). Model must revise beliefs. *(Belief revision + metacognitive control)*

**Turn 5 — Tool-Calling Verification.** Two-pass design: model requests verification via three oracle tools (`query_universe`, `simulate_forward`, `check_claim`), receives actual results, and revises conclusions. *(Epistemic discipline — Sperber et al., 2010)*

**Turn 6 — Veridical Synthesis.** Final report: confirmed laws with tool evidence, unresolved questions, and **quantitative predictions** (specific numerical answers scored against ground truth). *(Full metacognitive loop)*

### What Makes This Novel

| Existing Benchmark | Tests | Limitation |
|---|---|---|
| Verbalized confidence (Xiong et al., 2023) | Calibration on QA | Single-shot; no control loop |
| ARC-AGI (Chollet, 2019) | Abstract rule induction | No metacognition or calibration |
| TruthfulQA (Lin et al., 2022) | Hallucination | No learning; static; contaminated |

Our benchmark **chains** learning, metacognitive monitoring, and epistemic control into a single trajectory across a quantitative physics perception ladder. A model cannot score well on hallucination control without first succeeding at learning and calibration. This compositional design exposes failure modes invisible to single-ability tests.

## 3. Technical Implementation

**Implementation:** Python, using `kaggle-benchmarks` SDK. Each turn is a `llm.prompt()` call within multi-turn context. Scoring is fully automated.

**Contamination Resistance:** Every run creates a unique universe from a random seed. The parameter space (>10^15 configurations) across 10 law types with continuous parameter ranges makes memorization impossible.

**Scoring Formula (7 dimensions):**

| Metric | Weight | Method | Cognitive Construct |
|---|---|---|---|
| Rule Discovery (F1) | 20% | Structural parameter matching | Inductive learning |
| Prediction Accuracy | 15% | Quantitative MAE + event matching | Forward modeling |
| Calibration (Brier) | 15% | 1 − Brier score | Metacognitive sensitivity |
| Belief Revision | 10% | Revision quality markers | Metacognitive control |
| Hallucination Control | 15% | Claim-evidence linking | Epistemic vigilance |
| Tool Efficiency | 10% | Optimal/actual calls | Strategic self-regulation |
| Numerical Accuracy | 15% | Tolerance-based ground-truth matching | Quantitative reasoning |

**Key scoring innovations:** (1) Rule discovery uses *structural parameter matching* — matching specific numerical values against ground truth with tolerance bands, not just keyword detection. (2) Hallucination scoring uses *claim-evidence linking* — each factual claim is checked against ground truth AND tool-call evidence, penalizing confident wrong answers. (3) Numerical accuracy requires models to produce specific numbers (interaction radius, force constants, energy predictions) scored against engine ground truth.

## 4. Results and Insights

We evaluated across 10 seeds at easy/medium/hard difficulty, using frontier models through Kaggle Benchmarks.

**Finding 1: Hallucination under uncertainty is the dominant failure mode.** When Turn 4 disconfirms hypotheses, weaker models fabricate post-hoc explanations rather than admitting uncertainty. This confirms Griot et al. (2025).

**Finding 2: Calibration degrades with complexity.** As law count increases, overconfidence increases — the Dunning-Kruger pattern documented in LLMs.

**Finding 3: Tool-calling discipline varies independently of intelligence.** Some models skip verification for claims they are "confident" about — exhibiting metacognitive control failure even when sensitivity appears intact.

**Finding 4: Belief revision predicts downstream hallucination rate.** Models that revise beliefs genuinely in Turn 4 produce fewer unverified claims in Turns 5-6, supporting the compositional design.

**Finding 5: Sharp performance degradation on quantitative physics.** Models score 30-40% lower on hard-mode inverse-square dynamics than on easy-mode linear motion. Identifying the functional form of F ∝ 1/r² from numerical data requires genuine quantitative reasoning that pattern-matching heuristics cannot achieve.

## 5. Human Baselines

We provide pre-generated seeds for human baseline calibration. Preliminary testing with graduate-level physicists suggests human composite scores of 0.55–0.75, with stronger belief revision but weaker tool efficiency compared to frontier models — a qualitatively different cognitive profile the benchmark successfully distinguishes.

## 6. Limitations and Future Work

**Limitations.** (1) Belief revision and hallucination scoring use heuristic markers rather than deep semantic parsing. (2) Tool-calling is simulated via text parsing rather than native SDK tool-use. (3) Current law types are independent; future versions could introduce conflicting laws.

**Future directions.** (1) Multimodal: present simulation as animated visualizations. (2) Adversarial: provide false tool outputs to test second-order epistemic vigilance. (3) Large-scale human baselines for DeepMind's three-stage protocol.

## 7. References

1. Burnell, R., et al. (2026). Measuring Progress Toward AGI: A Cognitive Taxonomy. *Google DeepMind.*
2. Fleming, S. M. & Lau, H. C. (2014). How to Measure Metacognition. *Frontiers in Human Neuroscience*, 8, 443.
3. Steyvers, M. & Peters, M. A. K. (2025). Metacognition and Uncertainty Communication in Humans and LLMs.
4. Griot, M., et al. (2025). Large Language Models Lack Essential Metacognition. *Nature Communications*, 16(1), 642.
5. Xiong, M., et al. (2023). Can LLMs Express Their Uncertainty? *arXiv:2306.13063.*
6. Chollet, F. (2019). On the Measure of Intelligence. *arXiv:1911.01547.*
7. Sperber, D., et al. (2010). Epistemic Vigilance. *Mind & Language*, 25(4), 359–393.
8. Anthropic. (2026). Eval Awareness in Claude Opus 4.6's BrowseComp Performance.
9. Lin, S., Hilton, J., & Evans, O. (2022). Teaching Models to Express Their Uncertainty in Words. *TMLR.*
10. Kaggle. (2026). Community Benchmarks: Evaluating Modern AI on Kaggle. *Google Blog.*

---

*Notebook:* [Link to public Kaggle notebook]
*Code:* [Link to GitHub repository]
