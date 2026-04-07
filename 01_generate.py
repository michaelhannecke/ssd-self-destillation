#!/usr/bin/env python3
"""
Step 1 — Self-Sampling: Generate SSD Training Data
===================================================
Sample one solution per prompt from the base model using the paper's
training-time decoding configuration (T_train=1.6, top_p=0.8).

The key SSD insight: we do NOT filter for correctness.
Every raw output becomes training data — gibberish included.

Usage:
    python 01_generate.py
    python 01_generate.py --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit
    python 01_generate.py --n-prompts 100   # quick test run
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from config import SSDConfig, format_prompt


# -----------------------------------------------------------------
# Sampling approach
# -----------------------------------------------------------------
# The paper uses: temp -> top-k -> top-p -> sample (Appendix A).
# mlx_lm.generate() supports temp and top_p natively.
# top_k is not exposed in all mlx-lm versions, so we rely on
# top_p=0.8 as the primary truncation mechanism.  This still
# captures the core SSD effect: temperature-shifted, truncated
# sampling that reshapes the distribution.
# -----------------------------------------------------------------


def generate_response(
    model, tokenizer, prompt: str,
    temp: float, top_p: float, max_tokens: int,
) -> str:
    """
    Generate a single response using mlx_lm.generate() — the stable
    public API.  Wraps the call so the rest of the pipeline doesn't
    depend on internal mlx-lm details.
    """
    sampler = make_sampler(temp=temp, top_p=top_p)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    return response


def load_prompts(config: SSDConfig, n_prompts: int = None) -> list[dict]:
    """
    Load coding problem prompts.
    Uses MBPP for training (larger pool) and HumanEval for eval.
    """
    from datasets import load_dataset

    if config.train_dataset == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
        prompts = []
        for row in ds:
            prompts.append({
                "task_id": f"mbpp/{row['task_id']}",
                "prompt": row["text"],
                "test_list": row.get("test_list", []),
            })
    elif config.train_dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        prompts = []
        for row in ds:
            prompts.append({
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test": row.get("test", ""),
            })
    else:
        raise ValueError(f"Unknown dataset: {config.train_dataset}")

    if n_prompts:
        prompts = prompts[:n_prompts]

    print(f"Loaded {len(prompts)} prompts from {config.train_dataset}")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="SSD Step 1: Self-Sampling")
    parser.add_argument("--model", type=str, default=None, help="Override model ID")
    parser.add_argument("--n-prompts", type=int, default=None, help="Limit prompts (for testing)")
    parser.add_argument("--base-dir", type=str, default="./ssd_run")
    args = parser.parse_args()

    config = SSDConfig(base_dir=args.base_dir)
    if args.model:
        config.model_id = args.model

    config.base_dir.mkdir(parents=True, exist_ok=True)

    # -- Load model ------------------------------------------------
    print(f"Loading model: {config.model_id}")
    model, tokenizer = load(config.model_id)
    print("Model loaded.\n")

    # -- Load prompts ----------------------------------------------
    prompts = load_prompts(config, args.n_prompts)

    # -- Generate --------------------------------------------------
    print(f"Generating {config.n_samples_per_prompt} sample(s) per prompt")
    print(f"  T_train={config.t_train}, top_p={config.gen_top_p}")
    print(f"  max_tokens={config.gen_max_tokens}")
    print(f"  Output: {config.raw_samples_path}")
    print(f"  Note: top_k={config.gen_top_k} from the paper is approximated via top_p.")
    print()

    results = []
    t0 = time.time()

    with open(config.raw_samples_path, "w") as f:
        for i, problem in enumerate(prompts):
            prompt_text = format_prompt(problem["prompt"])

            for sample_idx in range(config.n_samples_per_prompt):
                response = generate_response(
                    model, tokenizer, prompt_text,
                    temp=config.t_train,
                    top_p=config.gen_top_p,
                    max_tokens=config.gen_max_tokens,
                )

                record = {
                    "task_id": problem["task_id"],
                    "prompt": problem["prompt"],
                    "completion": response,
                    "t_train": config.t_train,
                    "top_p": config.gen_top_p,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()  # write incrementally in case of interruption
                results.append(record)

            # Progress
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate if rate > 0 else 0

            if (i + 1) % 10 == 0 or i == 0:
                has_code = "def " in response or "class " in response or "return " in response
                print(
                    f"  [{i+1:4d}/{len(prompts)}]  "
                    f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining  "
                    f"{'ok code' if has_code else '-- no code'}"
                )

    elapsed = time.time() - t0

    # -- Summary statistics ----------------------------------------
    n_with_code = sum(
        1 for r in results
        if "def " in r["completion"] or "class " in r["completion"]
    )
    avg_len = sum(len(r["completion"]) for r in results) / max(len(results), 1)

    print(f"\n{'='*60}")
    print(f"Generation complete")
    print(f"  Samples: {len(results)}")
    print(f"  With extractable code: {n_with_code}/{len(results)} ({100*n_with_code/max(len(results),1):.1f}%)")
    print(f"  Avg completion length: {avg_len:.0f} chars")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s per sample)")
    print(f"  Saved to: {config.raw_samples_path}")
    print(f"\n  Note: SSD uses these raw outputs WITHOUT correctness filtering!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
