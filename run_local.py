#!/usr/bin/env python3
"""
Run Veridical Worlds benchmark against a local LLM on DGX Spark.
Supports Ollama and any OpenAI-compatible API endpoint (vLLM, TRT-LLM, etc.).

Usage:
    # With Ollama (default):
    ollama serve &
    ollama pull llama3.1:70b
    python run_local.py

    # With vLLM:
    python run_local.py --api-url http://localhost:8000/v1/chat/completions \
                        --model meta-llama/Llama-3.1-70B-Instruct \
                        --openai-compat

    # Quick test (2 seeds only):
    python run_local.py --quick
"""

import argparse
import json
import time
import requests
import pandas as pd

from benchmark_task import run_veridical_benchmark


class LocalLLM:
    """Adapter that wraps a local LLM API to match the kbench.llm interface."""

    def __init__(self, api_url, model, use_ollama=True, ctx_size=8192, temperature=0.3):
        self.api_url = api_url
        self.model = model
        self.use_ollama = use_ollama
        self.ctx_size = ctx_size
        self.temperature = temperature
        self.history = []

    def prompt(self, text, system=None, **kwargs):
        if system:
            self.history = [{"role": "system", "content": system}]

        self.history.append({"role": "user", "content": text})

        if self.use_ollama:
            response = requests.post(self.api_url, json={
                "model": self.model,
                "messages": self.history,
                "stream": False,
                "options": {"num_ctx": self.ctx_size, "temperature": self.temperature}
            }, timeout=600)
            response.raise_for_status()
            result = response.json()
            reply = result["message"]["content"]
        else:
            response = requests.post(self.api_url, json={
                "model": self.model,
                "messages": self.history,
                "max_tokens": 4096,
                "temperature": self.temperature,
            }, timeout=600)
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"]

        self.history.append({"role": "assistant", "content": reply})
        return reply


def main():
    parser = argparse.ArgumentParser(description="Run Veridical Worlds on a local LLM")
    parser.add_argument("--api-url", default="http://localhost:11434/api/chat",
                        help="LLM API endpoint (default: Ollama)")
    parser.add_argument("--model", default="llama3.1:70b",
                        help="Model name (default: llama3.1:70b)")
    parser.add_argument("--openai-compat", action="store_true",
                        help="Use OpenAI-compatible API format (for vLLM, TRT-LLM)")
    parser.add_argument("--ctx-size", type=int, default=8192,
                        help="Context window size (default: 8192)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with only 2 seeds")
    args = parser.parse_args()

    use_ollama = not args.openai_compat
    llm = LocalLLM(args.api_url, args.model, use_ollama, args.ctx_size, args.temperature)

    # Test connection
    print(f"Testing connection to {args.api_url} with model {args.model}...")
    try:
        llm.prompt("Say 'ready' if you can hear me.", system="Respond with a single word.")
        print("Connection OK.\n")
    except Exception as e:
        print(f"ERROR: Could not connect to LLM: {e}")
        print("Make sure your model server is running (e.g., `ollama serve &`)")
        return

    # Evaluation configs: seed, complexity pairs
    if args.quick:
        eval_configs = [(42, 3), (256, 5)]
    else:
        eval_configs = [
            (42, 3),      # easy: linear physics
            (137, 4),     # easy-medium
            (256, 5),     # medium: adds inverse-square or oscillation
            (1337, 5),    # medium
            (2026, 6),    # hard: nonlinear motion
        ]

    results = []
    total_start = time.time()

    for seed, complexity in eval_configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: seed={seed}, complexity={complexity}")
        print(f"{'='*60}")

        llm.history = []  # Reset conversation for each instance
        start = time.time()

        try:
            result = run_veridical_benchmark(llm, seed, complexity)
            elapsed = time.time() - start
            composite = result["scores"]["composite_score"]

            results.append({
                "seed": seed,
                "complexity": complexity,
                "composite": composite,
                "rule_discovery": result["scores"]["rule_discovery_f1"],
                "prediction": result["scores"]["prediction_accuracy"],
                "calibration": result["scores"]["calibration_brier"],
                "belief_update": result["scores"]["belief_update_quality"],
                "hallucination": result["scores"]["hallucination_control"],
                "tool_efficiency": result["scores"]["tool_efficiency"],
                "numerical": result["scores"]["numerical_accuracy"],
                "time_sec": round(elapsed, 1),
            })

            print(f"\nComposite Score: {composite:.4f} ({elapsed:.1f}s)")
            for k, v in result["scores"].items():
                if k not in ("composite_score", "turns"):
                    print(f"  {k}: {v}")

        except Exception as e:
            elapsed = time.time() - start
            print(f"\nERROR on seed {seed}: {e} ({elapsed:.1f}s)")
            results.append({
                "seed": seed, "complexity": complexity, "composite": 0.0,
                "rule_discovery": 0, "prediction": 0, "calibration": 0,
                "belief_update": 0, "hallucination": 0, "tool_efficiency": 0,
                "numerical": 0, "time_sec": round(elapsed, 1),
            })

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY — {args.model}")
    print(f"{'='*60}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nMean composite:  {df['composite'].mean():.4f}")
    print(f"Min composite:   {df['composite'].min():.4f}")
    print(f"Max composite:   {df['composite'].max():.4f}")
    print(f"Total time:      {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save results
    output_file = f"results_{args.model.replace('/', '_')}_{int(time.time())}.json"
    df.to_json(output_file, orient="records", indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
