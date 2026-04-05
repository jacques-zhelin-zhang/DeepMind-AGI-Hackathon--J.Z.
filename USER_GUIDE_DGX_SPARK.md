# Running Veridical Worlds on NVIDIA DGX Spark

A step-by-step guide for testing and running the Veridical Worlds benchmark on an [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) personal AI supercomputer.

---

## DGX Spark at a Glance

| Spec | Value |
|------|-------|
| CPU | ARM Blackwell GB10 — 10x Cortex-X925 (4 GHz) + 10x Cortex-A725 (2.8 GHz) |
| GPU | Integrated Blackwell GPU, up to 1 PFLOP FP4 sparse |
| Memory | 128 GB unified coherent (shared CPU/GPU) |
| OS | DGX OS (Ubuntu 24.04 LTS, ARM64, Linux 6.11 kernel) |
| Form Factor | Desktop (150mm x 150mm x 50mm) |

The DGX Spark's 128 GB unified memory and Blackwell GPU make it capable of running local LLM inference (up to 70B parameters), which is ideal for testing this benchmark against real models without needing cloud access.

---

## Prerequisites

Before starting, ensure your DGX Spark has:

- **DGX OS** installed and updated (ships pre-installed)
- **Network access** (Ethernet or Wi-Fi configured)
- A user account with sudo privileges

Verify your system:

```bash
# Check OS and architecture
uname -a
# Expected: Linux ... aarch64 GNU/Linux

# Check CPU cores (should show 20)
python3 -c "import os; print(f'CPU cores: {os.cpu_count()}')"

# Check available memory
free -h
# Expected: ~128 GB total

# Check GPU
nvidia-smi
```

---

## Step 1: Set Up Python Environment

The DGX Spark runs ARM64 Linux. Use a virtual environment to keep dependencies isolated.

```bash
# Install Python 3.11+ if not already present
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# Clone the repository
git clone https://github.com/jacques-zhelin-zhang/DeepMind-AGI-Hackathon--J.Z..git
cd DeepMind-AGI-Hackathon--J.Z.

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas
```

> **Note:** The physics engine and scoring use only the Python standard library. `pandas` is needed for evaluation data handling. No GPU libraries (PyTorch, TensorFlow) are required for the benchmark itself.

---

## Step 2: Run the Mock Test (No LLM Required)

This validates that the engine, scoring, and full 6-turn pipeline work correctly on ARM64.

```bash
python benchmark_task.py
```

**Expected output:**

```
kaggle_benchmarks not available. Running in local dev mode.
Imported engine from src/ package.
Evaluation instances: 5
   seed  complexity
0    42           3
1   137           4
2   256           5
3  1337           5
4  2026           6

--- Local Development Mode ---
Running benchmark with mock LLM for testing...

Seed 42: Composite = 0.3992
  rule_discovery_f1: 0.3333
  prediction_accuracy: 0.15
  calibration_brier: 0.7
  ...
```

This runs the full pipeline with a hardcoded mock LLM. It tests:
- All 10 physics law types (motion, interaction, conservation, threshold, causal_chain, decay, field_effect, inverse_square, oscillation, nonlinear_motion)
- Difficulty gating (complexity 3 = linear, 5 = nonlinear, 6 = all types)
- Structural parameter matching in scoring
- Numerical accuracy scoring
- Claim-evidence hallucination scoring

**If this works, your environment is correctly set up.**

---

## Step 3: Run with a Local LLM (Recommended for DGX Spark)

The DGX Spark's 128 GB unified memory can run large language models locally. This lets you test the benchmark against real models without Kaggle credentials.

### Option A: Using Ollama (Easiest)

[Ollama](https://ollama.com/) is the simplest way to run LLMs on DGX Spark. NVIDIA provides an [official playbook](https://github.com/NVIDIA/dgx-spark-playbooks) for this.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose based on your needs)
ollama pull llama3.1:70b      # Best quality (fits in 128GB)
ollama pull llama3.1:8b       # Faster, good for iteration
ollama pull qwen2.5:32b       # Strong reasoning model

# Verify it's running
ollama list
```

### Option B: Using vLLM (Higher Throughput)

For batch evaluation across multiple seeds:

```bash
pip install vllm

# Start vLLM server with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --dtype auto \
    --max-model-len 8192 \
    --port 8000
```

### Create a Local LLM Adapter

Create `run_local.py` to bridge the benchmark with your local LLM:

```python
#!/usr/bin/env python3
"""
Run Veridical Worlds benchmark against a local LLM on DGX Spark.
Supports Ollama and any OpenAI-compatible API endpoint.
"""

import json
import requests
import pandas as pd
from benchmark_task import run_veridical_benchmark

# ── Configuration ──────────────────────────────────────────────
# For Ollama:
API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:70b"
USE_OLLAMA = True

# For vLLM / OpenAI-compatible:
# API_URL = "http://localhost:8000/v1/chat/completions"
# MODEL = "meta-llama/Llama-3.1-70B-Instruct"
# USE_OLLAMA = False


class LocalLLM:
    """Adapter that wraps a local LLM API to match the kbench.llm interface."""

    def __init__(self):
        self.history = []

    def prompt(self, text, system=None, **kwargs):
        if system:
            self.history = [{"role": "system", "content": system}]

        self.history.append({"role": "user", "content": text})

        if USE_OLLAMA:
            response = requests.post(API_URL, json={
                "model": MODEL,
                "messages": self.history,
                "stream": False,
                "options": {"num_ctx": 8192, "temperature": 0.3}
            }, timeout=300)
            result = response.json()
            reply = result["message"]["content"]
        else:
            response = requests.post(API_URL, json={
                "model": MODEL,
                "messages": self.history,
                "max_tokens": 4096,
                "temperature": 0.3,
            }, timeout=300)
            result = response.json()
            reply = result["choices"][0]["message"]["content"]

        self.history.append({"role": "assistant", "content": reply})
        return reply


# ── Run Evaluation ─────────────────────────────────────────────
if __name__ == "__main__":
    llm = LocalLLM()

    # Test seeds across the difficulty ladder
    eval_configs = [
        (42, 3),     # easy: linear physics
        (137, 4),    # easy-medium
        (256, 5),    # medium: adds inverse-square or oscillation
        (1337, 5),   # medium
        (2026, 6),   # hard: nonlinear motion
    ]

    results = []
    for seed, complexity in eval_configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: seed={seed}, complexity={complexity}")
        print(f"{'='*60}")

        # Reset conversation history for each evaluation instance
        llm.history = []

        result = run_veridical_benchmark(llm, seed, complexity)
        composite = result["scores"]["composite_score"]
        results.append({"seed": seed, "complexity": complexity, "composite": composite})

        print(f"\nComposite Score: {composite:.4f}")
        for k, v in result["scores"].items():
            if k not in ("composite_score", "turns"):
                print(f"  {k}: {v}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nMean composite: {df['composite'].mean():.4f}")
    print(f"Min composite:  {df['composite'].min():.4f}")
    print(f"Max composite:  {df['composite'].max():.4f}")
```

### Run It

```bash
# Make sure Ollama is running with your chosen model
ollama serve &
ollama pull llama3.1:70b

# Run the benchmark
python run_local.py
```

**Expected runtime on DGX Spark:**
- 70B model: ~5-10 min per seed (6 turns of inference)
- 8B model: ~1-2 min per seed
- Full 5-seed evaluation: 25-50 min (70B) or 5-10 min (8B)

---

## Step 4: Run on Kaggle Platform (Competition Submission)

For the actual competition submission, use the self-contained `kaggle_notebook.py`:

1. Go to [kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new)
2. Create a new notebook
3. Copy the entire contents of `kaggle_notebook.py` (938 lines, fully self-contained)
4. Run all cells
5. In the final cell, uncomment: `%choose veridical_worlds_metacognition`
6. Click **Save Version** to submit

The Kaggle platform provides access to frontier models (Gemini, Claude, etc.) via the `kbench.llm` object with a daily quota of $50.

---

## Step 5: Parallel Evaluation (Leverage All 20 Cores)

The DGX Spark has 20 CPU cores. While LLM inference is the bottleneck (not CPU), you can parallelize the physics engine testing:

```python
#!/usr/bin/env python3
"""Parallel engine stress test across all DGX Spark cores."""

from multiprocessing import Pool, cpu_count
from veridical_engine import MicroUniverse
import time

def test_universe(args):
    seed, complexity = args
    u = MicroUniverse(seed=seed, complexity=complexity)
    u.simulate(50)  # Extended simulation
    gt = u.get_ground_truth()
    challenges = u.generate_quantitative_challenges()
    return {
        "seed": seed,
        "complexity": complexity,
        "laws": len(gt["laws"]),
        "law_types": [l["law_type"] for l in gt["laws"]],
        "final_alive": len(u.history[-1].entities),
        "challenges": len(challenges),
    }

if __name__ == "__main__":
    configs = [(seed, c) for seed in range(100) for c in [3, 5, 7]]
    print(f"Testing {len(configs)} universes across {cpu_count()} cores...")

    start = time.time()
    with Pool(cpu_count()) as pool:
        results = pool.map(test_universe, configs)
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.1f}s ({len(configs)/elapsed:.1f} universes/sec)")

    # Check law type distribution
    from collections import Counter
    all_types = [t for r in results for t in r["law_types"]]
    print("\nLaw type distribution:")
    for law_type, count in Counter(all_types).most_common():
        print(f"  {law_type}: {count}")
```

---

## Troubleshooting

### ARM64 Package Issues

If `pip install` fails for any package on ARM64:

```bash
# Use conda-forge which has better ARM64 support
conda install -c conda-forge pandas

# Or build from source
pip install --no-binary :all: pandas
```

### Ollama Not Detecting GPU

```bash
# Check NVIDIA drivers
nvidia-smi

# Ensure Ollama sees the GPU
ollama run llama3.1:8b --verbose 2>&1 | grep -i gpu

# If GPU not detected, reinstall Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### Out of Memory with 70B Models

The 128 GB unified memory should handle 70B models, but if you hit OOM:

```bash
# Use a smaller model
ollama pull llama3.1:8b

# Or use quantized versions
ollama pull llama3.1:70b-instruct-q4_0  # ~40GB

# Or reduce context window in run_local.py
"options": {"num_ctx": 4096}  # instead of 8192
```

### Benchmark Takes Too Long

```bash
# Reduce evaluation to 2 seeds for quick testing
# Edit run_local.py:
eval_configs = [
    (42, 3),     # easy
    (256, 5),    # medium
]
```

---

## What the Benchmark Tests

| Turn | Cognitive Ability | What Happens |
|------|-------------------|--------------|
| 1 | Perception | Model observes 20 timesteps of physics simulation |
| 2 | Inductive Learning | Model proposes hidden laws with confidence scores |
| 3 | Forward Modeling | Model predicts 5 future timesteps quantitatively |
| 4 | Belief Revision | Model revises hypotheses after disconfirming evidence |
| 5 | Epistemic Discipline | Model uses tools to verify claims (two-pass) |
| 6 | Veridical Synthesis | Final report with verified laws and honest uncertainty |

**Physics Perception Ladder:**

| Difficulty | Law Types | What It Tests |
|------------|-----------|---------------|
| Easy (complexity 3-4) | Linear motion, interactions, conservation | Basic pattern recognition |
| Medium (complexity 5-6) | + Inverse-square, oscillation, causal delays | Nonlinear dynamics identification |
| Hard (complexity 7) | + Nonlinear motion (quadratic/logarithmic) | Quantitative reasoning beyond language |

---

## File Reference

| File | Purpose | Size |
|------|---------|------|
| `benchmark_task.py` | Main runner (modular, imports from source files) | 391 lines |
| `kaggle_notebook.py` | Self-contained Kaggle submission | 938 lines |
| `veridical_engine.py` | Physics engine (10 law types) | 900+ lines |
| `scoring.py` | 7-dimension scoring engine | 450+ lines |
| `prompts.py` | 6-turn prompt templates | 180+ lines |
| `run_local.py` | Local LLM adapter (create per Step 3) | ~100 lines |
| `WRITEUP.md` | Competition writeup | 1,271 words |

---

## Quick Reference Commands

```bash
# Activate environment
cd DeepMind-AGI-Hackathon--J.Z. && source .venv/bin/activate

# Mock test (no LLM needed)
python benchmark_task.py

# Test with local Ollama model
ollama serve &
python run_local.py

# Parallel engine stress test
python parallel_test.py

# Check a specific universe
python -c "
from veridical_engine import MicroUniverse
u = MicroUniverse(seed=42, complexity=7)
u.simulate(20)
gt = u.get_ground_truth()
for law in gt['laws']:
    print(f\"{law['law_id']}: {law['law_type']} — {law['description']}\")
"
```
