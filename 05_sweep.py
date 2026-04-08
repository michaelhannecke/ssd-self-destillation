#!/usr/bin/env python3
"""
Step 5 — Temperature Sweep: The Core SSD Evidence
==================================================
Reproduces Figure 2 from the paper at small scale:
  - Sweep T_eval on the base model → flat curve
  - Plot SSD result → horizontal line above the curve

This is the most important result for the article because it shows
that SSD produces changes *in the model itself* that no decoding
configuration can replicate (Section 3.3).

Usage:
    python 05_sweep.py
    python 05_sweep.py --n-problems 30 --n-samples 3   # quick version
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import SSDConfig, format_prompt


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def quick_eval(model, tokenizer, problems, config, temp, n_samples, sampler_fn):
    """Run a lightweight eval at a single temperature."""
    results = {}
    for problem in problems:
        correct = 0
        for _ in range(n_samples):
            prompt = format_prompt(
                f"Complete the following Python function:\n\n{problem['prompt']}"
            )
            completion = sampler_fn(model, tokenizer, prompt, config, temp)
            is_correct = check_correctness_quick(problem, completion)
            correct += int(is_correct)
        results[problem["task_id"]] = {"n": n_samples, "c": correct}

    pass1 = np.mean([
        pass_at_k(v["n"], v["c"], 1)
        for v in results.values()
    ])
    return float(pass1)


def check_correctness_quick(problem, completion, timeout=8):
    """Minimal correctness check — same as in 04_eval.py."""
    import subprocess
    import sys
    import tempfile

    # Extract code
    code = completion.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            lines = code.split("\n")
            if lines and lines[0].strip() in ("python", "py", ""):
                lines = lines[1:]
            code = "\n".join(lines)

    full_code = f"""
{problem['prompt']}
{code}

{problem['test']}

check({problem['entry_point']})
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=timeout, text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            Path(f.name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="SSD Step 5: Temperature Sweep")
    parser.add_argument("--base-dir", type=str, default="./ssd_run")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n-problems", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--temps", type=str, default=None,
                        help="Comma-separated temps, e.g. '0.5,0.7,0.9,1.1,1.3'")
    args = parser.parse_args()

    config = SSDConfig(base_dir=args.base_dir)
    if args.model:
        config.model_id = args.model
    n_samples = args.n_samples or config.sweep_n_samples
    temps = [float(t) for t in args.temps.split(",")] if args.temps else config.sweep_temps

    # ── Setup ─────────────────────────────────────────────────
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm import generate as mlx_generate
    from mlx_lm.generate import make_sampler

    def sampler_fn(model, tokenizer, prompt, cfg, temp):
        return mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=cfg.eval_max_tokens,
            sampler=make_sampler(temp=temp, top_p=cfg.eval_top_p),
            verbose=False,
        )

    # ── Load problems ─────────────────────────────────────────
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = [dict(row) for row in ds]
    if args.n_problems:
        problems = problems[:args.n_problems]
    print(f"Loaded {len(problems)} HumanEval problems")
    print(f"Temperatures: {temps}")
    print(f"Samples per problem per temp: {n_samples}")
    total_gens = len(problems) * n_samples * len(temps) * 2  # base + ssd
    print(f"Total generations needed: ~{total_gens}")

    config.eval_results_path.mkdir(parents=True, exist_ok=True)
    sweep_results = {"base": {}, "ssd": {}}

    # ── Base model sweep ──────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  BASE MODEL TEMPERATURE SWEEP")
    print(f"{'═'*60}")
    model, tokenizer = load(config.model_id)

    for temp in temps:
        t0 = time.time()
        p1 = quick_eval(model, tokenizer, problems, config, temp, n_samples, sampler_fn)
        elapsed = time.time() - t0
        sweep_results["base"][str(temp)] = p1
        print(f"  T={temp:.1f}  pass@1={p1*100:.1f}%  ({elapsed:.0f}s)")

    del model
    if hasattr(mx, "metal"):
        mx.metal.clear_cache()

    # ── SSD model sweep ───────────────────────────────────────
    adapter_path = config.adapter_path
    has_adapter = (adapter_path / "adapters.safetensors").exists()

    if has_adapter:
        print(f"\n{'═'*60}")
        print(f"  SSD MODEL TEMPERATURE SWEEP")
        print(f"{'═'*60}")
        model, tokenizer = load(config.model_id, adapter_path=str(adapter_path))

        for temp in temps:
            t0 = time.time()
            p1 = quick_eval(model, tokenizer, problems, config, temp, n_samples, sampler_fn)
            elapsed = time.time() - t0
            sweep_results["ssd"][str(temp)] = p1
            print(f"  T={temp:.1f}  pass@1={p1*100:.1f}%  ({elapsed:.0f}s)")
    else:
        print(f"\n⚠  No SSD adapter found at {adapter_path}")
        print("   Sweep will only show base model results.")

    # ── Save raw data ─────────────────────────────────────────
    sweep_file = config.eval_results_path / "sweep_results.json"
    with open(sweep_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep data saved to {sweep_file}")

    # ── Print summary table ───────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  TEMPERATURE SWEEP SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'T_eval':>6}  {'Base pass@1':>12}  {'SSD pass@1':>12}  {'Delta':>8}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}")

    best_base = 0.0
    for temp in temps:
        base_val = sweep_results["base"].get(str(temp), 0)
        ssd_val = sweep_results["ssd"].get(str(temp))
        best_base = max(best_base, base_val)

        base_str = f"{base_val*100:.1f}%"
        ssd_str = f"{ssd_val*100:.1f}%" if ssd_val is not None else "  —"
        delta_str = f"{(ssd_val - base_val)*100:+.1f}pp" if ssd_val is not None else "  —"
        print(f"  {temp:6.1f}  {base_str:>12}  {ssd_str:>12}  {delta_str:>8}")

    if has_adapter:
        best_ssd = max(sweep_results["ssd"].values())
        margin = best_ssd - best_base
        print(f"\n  Best base:  {best_base*100:.1f}%")
        print(f"  Best SSD:   {best_ssd*100:.1f}%")
        print(f"  SSD margin: {margin*100:+.1f}pp over best-tuned base")
        print(f"\n  → This margin is the key result (cf. Figure 2 in paper):")
        print(f"    SSD changes the model itself, not just the decoding.")

    # ── Generate plot script ──────────────────────────────────
    plot_script = config.eval_results_path / "plot_sweep.py"
    with open(plot_script, "w") as f:
        f.write(PLOT_SCRIPT)
    print(f"\n  Plot: python {plot_script}")
    print(f"{'═'*60}")


PLOT_SCRIPT = '''#!/usr/bin/env python3
"""Generate the temperature sweep plot (Figure 2 analogue)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path(__file__).parent
with open(results_dir / "sweep_results.json") as f:
    data = json.load(f)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Base model sweep
temps = sorted([float(t) for t in data["base"].keys()])
base_vals = [data["base"][str(t)] * 100 for t in temps]
ax.plot(temps, base_vals, "o-", color="#E8713A", linewidth=2,
        markersize=8, label="Base model (sweep)", zorder=3)

# SSD model sweep (if available)
if data.get("ssd"):
    ssd_temps = sorted([float(t) for t in data["ssd"].keys()])
    ssd_vals = [data["ssd"][str(t)] * 100 for t in ssd_temps]
    ax.plot(ssd_temps, ssd_vals, "s-", color="#4A90D9", linewidth=2,
            markersize=8, label="SSD model (sweep)", zorder=3)

    # Shade the margin
    best_base = max(base_vals)
    best_ssd = max(ssd_vals)
    margin = best_ssd - best_base
    ax.axhline(best_base, color="#E8713A", linestyle="--", alpha=0.5)
    ax.axhline(best_ssd, color="#4A90D9", linestyle="--", alpha=0.5)

    # Annotate margin
    mid_t = (min(temps) + max(temps)) / 2
    ax.annotate(
        f"+{margin:.1f}pp",
        xy=(max(temps) + 0.05, (best_base + best_ssd) / 2),
        fontsize=14, fontweight="bold", color="#4A90D9",
        ha="left", va="center",
    )

ax.set_xlabel("Evaluation Temperature $T_{eval}$", fontsize=13)
ax.set_ylabel("pass@1 (%)", fontsize=13)
ax.set_title("SSD vs. Best Base Decoding (HumanEval)", fontsize=14)
ax.legend(fontsize=12, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(min(temps) - 0.05, max(temps) + 0.2)

plt.tight_layout()
plt.savefig(results_dir / "sweep_plot.png", dpi=150, bbox_inches="tight")
plt.savefig(results_dir / "sweep_plot.svg", bbox_inches="tight")
print(f"Plot saved to {results_dir / 'sweep_plot.png'}")
plt.show()
'''


if __name__ == "__main__":
    main()
