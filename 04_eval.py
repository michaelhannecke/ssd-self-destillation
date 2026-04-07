#!/usr/bin/env python3
"""
Step 4 — Evaluate: Base Model vs. SSD on HumanEval
===================================================
Runs pass@1 and pass@5 on HumanEval for both the base model
and the SSD-finetuned model, using the paper's recommended
decoding settings (Table 3 / Table 4).

This is the "money shot" for the tutorial: does SSD actually
improve code generation on Apple Silicon with LoRA?

Usage:
    python 04_eval.py
    python 04_eval.py --skip-base        # only eval SSD model
    python 04_eval.py --n-samples 5      # faster, less precise
    python 04_eval.py --n-problems 20    # subset for quick test

Safety note:
    This script executes model-generated code in a subprocess.
    It uses timeouts and basic sandboxing, but for production use
    consider Docker or a proper sandbox.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import SSDConfig, format_prompt

# Reuse the custom sampler from step 1
from importlib import import_module


def load_humaneval() -> list[dict]:
    """Load HumanEval problems."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for row in ds:
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],          # Function signature + docstring
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return problems


def generate_completion(model, tokenizer, problem: dict, config, temp: float, sampler_fn) -> str:
    """Generate a single code completion for a HumanEval problem."""
    # HumanEval expects the model to complete the function body
    # given the signature + docstring as prompt
    chat_prompt = format_prompt(
        f"Complete the following Python function:\n\n{problem['prompt']}"
    )
    return sampler_fn(model, tokenizer, chat_prompt, config, temp)


def extract_code(completion: str, entry_point: str) -> str:
    """
    Extract executable code from model output.
    Handles markdown code blocks and raw code.
    """
    # Try to extract from markdown code block
    if "```python" in completion:
        blocks = completion.split("```python")
        if len(blocks) > 1:
            code = blocks[1].split("```")[0]
            return code.strip()
    if "```" in completion:
        blocks = completion.split("```")
        if len(blocks) > 1:
            code = blocks[1].split("```")[0]
            # Remove language identifier if present
            lines = code.split("\n")
            if lines and lines[0].strip() in ("python", "py", ""):
                lines = lines[1:]
            return "\n".join(lines).strip()

    # Fallback: return as-is
    return completion.strip()


def check_correctness(problem: dict, completion: str, timeout: int = 10) -> bool:
    """
    Execute the generated code against HumanEval test cases.
    Returns True if all tests pass.
    """
    code = extract_code(completion, problem["entry_point"])

    # Build the full test program:
    # 1. The function prompt (signature + docstring)
    # 2. The generated completion (function body)
    # 3. The test harness
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
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            Path(f.name).unlink(missing_ok=True)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.
    n: total samples, c: number correct, k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_model(
    model, tokenizer, problems: list[dict], config, temp: float,
    sampler_fn, n_samples: int, label: str
) -> dict:
    """Run full evaluation and return metrics."""
    print(f"\n{'═'*60}")
    print(f"  Evaluating: {label}")
    print(f"  T_eval={temp}, top_p={config.eval_top_p}, top_k={config.eval_top_k}")
    print(f"  {n_samples} samples × {len(problems)} problems")
    print(f"{'═'*60}\n")

    results = defaultdict(list)
    t0 = time.time()

    for i, problem in enumerate(problems):
        correct_count = 0

        for s in range(n_samples):
            completion = generate_completion(
                model, tokenizer, problem, config, temp, sampler_fn
            )
            is_correct = check_correctness(problem, completion)
            correct_count += int(is_correct)
            results[problem["task_id"]].append(is_correct)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            running_pass1 = np.mean([
                pass_at_k(len(v), sum(v), 1) for v in results.values()
            ])
            print(
                f"  [{i+1:3d}/{len(problems)}]  "
                f"running pass@1={running_pass1:.3f}  "
                f"({elapsed:.0f}s elapsed)"
            )

    # ── Compute metrics ───────────────────────────────────────
    pass1_scores = []
    pass5_scores = []

    for task_id, outcomes in results.items():
        n = len(outcomes)
        c = sum(outcomes)
        pass1_scores.append(pass_at_k(n, c, 1))
        if n >= 5:
            pass5_scores.append(pass_at_k(n, c, 5))

    metrics = {
        "label": label,
        "temp": temp,
        "n_problems": len(problems),
        "n_samples": n_samples,
        "pass@1": float(np.mean(pass1_scores)),
        "pass@5": float(np.mean(pass5_scores)) if pass5_scores else None,
        "per_problem": {
            tid: {"n": len(outcomes), "correct": sum(outcomes)}
            for tid, outcomes in results.items()
        },
    }

    elapsed = time.time() - t0
    print(f"\n  Results for {label}:")
    print(f"    pass@1 = {metrics['pass@1']:.4f} ({metrics['pass@1']*100:.1f}%)")
    if metrics["pass@5"] is not None:
        print(f"    pass@5 = {metrics['pass@5']:.4f} ({metrics['pass@5']*100:.1f}%)")
    print(f"    Time: {elapsed:.0f}s")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SSD Step 4: HumanEval Evaluation")
    parser.add_argument("--base-dir", type=str, default="./ssd_run")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--skip-base", action="store_true", help="Skip base model eval")
    parser.add_argument("--skip-ssd", action="store_true", help="Skip SSD model eval")
    parser.add_argument("--n-samples", type=int, default=None, help="Override sample count")
    parser.add_argument("--n-problems", type=int, default=None, help="Limit problems")
    args = parser.parse_args()

    config = SSDConfig(base_dir=args.base_dir)
    if args.model:
        config.model_id = args.model
    if args.n_samples:
        config.n_eval_samples = args.n_samples

    config.eval_results_path.mkdir(parents=True, exist_ok=True)

    # ── Import sampler ────────────────────────────────────────
    # Reuse the generate_response from step 1 but make temp configurable
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.utils import generate_step
    from importlib.machinery import SourceFileLoader

    gen_module = SourceFileLoader("gen", str(Path(__file__).parent / "01_generate.py")).load_module()

    def sampler_fn(model, tokenizer, prompt, cfg, temp):
        """Wrapper that overrides temperature."""
        original_temp = cfg.t_train
        cfg.t_train = temp  # Temporarily override
        cfg.gen_top_k = cfg.eval_top_k
        cfg.gen_top_p = cfg.eval_top_p
        cfg.gen_max_tokens = cfg.eval_max_tokens
        result = gen_module.generate_response(model, tokenizer, prompt, cfg)
        cfg.t_train = original_temp
        return result

    # ── Load problems ─────────────────────────────────────────
    problems = load_humaneval()
    if args.n_problems:
        problems = problems[:args.n_problems]
    print(f"Loaded {len(problems)} HumanEval problems")

    all_metrics = []

    # ── Evaluate base model ───────────────────────────────────
    if not args.skip_base:
        print(f"\nLoading base model: {config.model_id}")
        model, tokenizer = load(config.model_id)

        base_metrics = evaluate_model(
            model, tokenizer, problems, config,
            temp=config.t_eval_base,
            sampler_fn=sampler_fn,
            n_samples=config.n_eval_samples,
            label="Base Model"
        )
        all_metrics.append(base_metrics)

        # Free memory before loading adapter
        del model
        mx.metal.clear_cache() if hasattr(mx, "metal") else None

    # ── Evaluate SSD model ────────────────────────────────────
    if not args.skip_ssd:
        adapter_path = config.adapter_path
        if not (adapter_path / "adapters.safetensors").exists():
            print(f"\n⚠  No adapter found at {adapter_path}")
            print("   Run 03_train.sh first, or use --skip-ssd")
        else:
            print(f"\nLoading SSD model: {config.model_id} + {adapter_path}")
            model, tokenizer = load(config.model_id, adapter_path=str(adapter_path))

            ssd_metrics = evaluate_model(
                model, tokenizer, problems, config,
                temp=config.t_eval_ssd,
                sampler_fn=sampler_fn,
                n_samples=config.n_eval_samples,
                label="SSD Model (LoRA)"
            )
            all_metrics.append(ssd_metrics)

    # ── Summary ───────────────────────────────────────────────
    if len(all_metrics) >= 2:
        base = all_metrics[0]
        ssd = all_metrics[1]
        delta = ssd["pass@1"] - base["pass@1"]

        print(f"\n{'═'*60}")
        print(f"  COMPARISON")
        print(f"{'═'*60}")
        print(f"  Base model pass@1:  {base['pass@1']*100:.1f}%  (T={base['temp']})")
        print(f"  SSD model pass@1:   {ssd['pass@1']*100:.1f}%  (T={ssd['temp']})")
        print(f"  Delta:              {delta*100:+.1f}pp")
        if base["pass@5"] and ssd["pass@5"]:
            delta5 = ssd["pass@5"] - base["pass@5"]
            print(f"\n  Base model pass@5:  {base['pass@5']*100:.1f}%")
            print(f"  SSD model pass@5:   {ssd['pass@5']*100:.1f}%")
            print(f"  Delta:              {delta5*100:+.1f}pp")
        print(f"{'═'*60}")

    # ── Save results ──────────────────────────────────────────
    results_file = config.eval_results_path / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
