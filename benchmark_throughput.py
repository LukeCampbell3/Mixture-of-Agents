#!/usr/bin/env python3
"""
Local model throughput benchmark.
Measures time-to-first-token, tokens/sec, and total latency
using the streaming OllamaClient that is now wired into the system.
"""

import sys
import time
import requests

OLLAMA_URL  = "http://localhost:11434"
TEST_PROMPT = "Explain what a Python list comprehension is in two sentences."
MAX_TOKENS  = 120

# ── helpers ──────────────────────────────────────────────────────────────────

def available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
        sys.exit(1)


def benchmark_model(model: str) -> dict:
    """Run one benchmark pass using the streaming OllamaClient."""
    from app.models.local_llm_client import OllamaClient
    client = OllamaClient(model=model, base_url=OLLAMA_URL)

    t_wall_start = time.perf_counter()
    response = client.generate(TEST_PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
    t_wall_end = time.perf_counter()

    m = client.last_metrics()
    return {
        "model":       model,
        "ttft_s":      m.get("ttft_s"),
        "total_s":     m.get("total_s", round(t_wall_end - t_wall_start, 3)),
        "tokens":      m.get("tokens", 0),
        "tok_per_sec": m.get("tok_per_sec", 0),
        "response":    response[:100].replace("\n", " "),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    models = available_models()
    if not models:
        print("No models found in Ollama.")
        sys.exit(1)

    print(f"\nBenchmarking {len(models)} model(s)")
    print(f"Prompt : \"{TEST_PROMPT}\"")
    print(f"Tokens : up to {MAX_TOKENS}\n")
    print(f"{'Model':<22} {'TTFT':>7} {'Total':>8} {'Tokens':>7} {'Tok/s':>8}")
    print("─" * 58)

    results = []
    for model in models:
        sys.stdout.write(f"  {model:<20} running...")
        sys.stdout.flush()
        try:
            r = benchmark_model(model)
            results.append(r)
            ttft_str = f"{r['ttft_s']}s" if r["ttft_s"] is not None else "  n/a"
            print(
                f"\r  {r['model']:<20} "
                f"{ttft_str:>7} "
                f"{r['total_s']:>7}s "
                f"{r['tokens']:>7} "
                f"{r['tok_per_sec']:>7} t/s"
            )
            print(f"    └─ \"{r['response']}...\"")
        except Exception as e:
            print(f"\r  {model:<20}  ERROR: {e}")

    # ── summary ──────────────────────────────────────────────────────────────
    if results:
        best = max(results, key=lambda x: x["tok_per_sec"])
        print(f"\n{'─'*58}")
        print(f"Fastest : {best['model']}  —  {best['tok_per_sec']} tok/s")
        print(f"TTFT    : {best['ttft_s']}s   Total: {best['total_s']}s")

        if best["tok_per_sec"] < 5:
            print("\n⚠  Throughput is very low (<5 tok/s).")
            print("   Recommendations:")
            print("   • Use qwen2.5:1.5b — it is ~4x faster than 7b on CPU")
            print("   • Set OLLAMA_NUM_THREADS to your physical core count:")
            print("     $env:OLLAMA_NUM_THREADS=8; ollama serve")
            print("   • Keep context short — long prompts slow generation")
        elif best["tok_per_sec"] < 15:
            print("\n✅ Acceptable throughput for interactive use.")
            print("   Tip: set OLLAMA_NUM_THREADS for a further boost.")
        else:
            print("\n✅ Good throughput — model is well-suited for this hardware.")

        print()


if __name__ == "__main__":
    main()
