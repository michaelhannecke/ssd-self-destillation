#!/usr/bin/env python3
"""
Step 1 — Self-Sampling: Generate SSD Training Data
===================================================
Sample one solution per prompt from the base model using the paper's
training-time decoding configuration (T_train=1.6, top_k=20, top_p=0.8).

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

import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import generate_step

from config import SSDConfig, format_prompt


# ─────────────────────────────────────────────────────────────────
# Custom sampler with top-k support
# mlx-lm's built-in generate() doesn't expose top-k directly,
# so we replicate the vLLM pipeline: temp → top-k → top-p → sample
# This matches Appendix A of the paper exactly.
# ─────────────────────────────────────────────────────────────────

def sample_token(logits: mx.array, temp: float, top_k: int, top_p: float) -> mx.array:
    """
    Sample one token following the paper's decoding pipeline:
      1. Temperature scaling
      2. Top-k filtering
      3. Top-p (nucleus) filtering
      4. Categorical sample

    This mirrors vLLM's Sampler (Figure 8 in the paper).
    """
    # Step 1: Temperature scaling
    if temp <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temp

    # Step 2: Top-k filtering
    if top_k > 0 and top_k < logits.shape[-1]:
        # Get the k-th largest value as threshold
        top_k_vals = mx.topk(logits, k=top_k, axis=-1)
        threshold = top_k_vals[:, -1:]  # smallest of the top-k
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    # Step 3: Top-p (nucleus) filtering
    if top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)

        # Reverse to descending order
        sorted_indices = sorted_indices[:, ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)

        # Create mask: keep tokens until cumulative mass >= top_p
        # Shift cumulative by 1 so the token that crosses threshold is kept
        mask = mx.concatenate(
            [mx.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], axis=-1
        )
        mask = mask >= top_p

        # Zero out masked probabilities, scatter back
        sorted_probs = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)
        probs = mx.zeros_like(probs)
        probs = probs.at[
            mx.arange(probs.shape[0])[:, None],
            sorted_indices
        ].add(sorted_probs)

        # Re-normalize
        probs = probs / (mx.sum(probs, axis=-1, keepdims=True) + 1e-10)

        # Step 4: Sample from the filtered distribution
        token = mx.random.categorical(mx.log(probs + 1e-10), axis=-1)
        return token

    # Fallback: just temperature + top-k, no top-p
    token = mx.random.categorical(logits, axis=-1)
    return token


def generate_response(
    model, tokenizer, prompt: str, config: SSDConfig
) -> str:
    """
    Generate a single response using the SSD training-time decoding config.
    Uses streaming token generation for memory efficiency on Apple Silicon.
    """
    tokens = tokenizer.encode(prompt, return_tensors="mlx")
    if hasattr(tokens, "tolist"):
        input_ids = tokens.tolist()
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
    else:
        input_ids = list(tokens)

    generated = []
    eos_token_id = tokenizer.eos_token_id

    # Handle models with multiple EOS tokens (common in Qwen3)
    eos_ids = set()
    if eos_token_id is not None:
        eos_ids.add(eos_token_id)
    # Qwen uses <|im_end|> as stop token in chat mode
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.add(im_end_id)

    # Use mlx-lm's generate_step for streaming generation
    prompt_tokens = mx.array([input_ids])

    for i, (token_logits, _) in enumerate(
        generate_step(prompt_tokens, model)
    ):
        if i >= config.gen_max_tokens:
            break

        # Apply our custom sampling (temp → top-k → top-p)
        logits = token_logits.reshape(1, -1)
        next_token = sample_token(
            logits, config.t_train, config.gen_top_k, config.gen_top_p
        )
        token_id = next_token.item()

        if token_id in eos_ids:
            break

        generated.append(token_id)

    return tokenizer.decode(generated)


def load_prompts(config: SSDConfig, n_prompts: int = None) -> list[dict]:
    """
    Load coding problem prompts.
    Uses MBPP for training (larger pool) and HumanEval for eval (clean split).
    """
    from datasets import load_dataset

    if config.train_dataset == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
        prompts = []
        for row in ds:
            prompts.append({
                "task_id": f"mbpp/{row['task_id']}",
                "prompt": row["text"],  # Natural language description
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

    # ── Load model ────────────────────────────────────────────
    print(f"Loading model: {config.model_id}")
    model, tokenizer = load(config.model_id)
    print("Model loaded.\n")

    # ── Load prompts ──────────────────────────────────────────
    prompts = load_prompts(config, args.n_prompts)

    # ── Generate ──────────────────────────────────────────────
    print(f"Generating {config.n_samples_per_prompt} sample(s) per prompt")
    print(f"  T_train={config.t_train}, top_k={config.gen_top_k}, top_p={config.gen_top_p}")
    print(f"  max_tokens={config.gen_max_tokens}")
    print(f"  Output: {config.raw_samples_path}\n")

    results = []
    t0 = time.time()

    with open(config.raw_samples_path, "w") as f:
        for i, problem in enumerate(prompts):
            prompt_text = format_prompt(problem["prompt"])

            for sample_idx in range(config.n_samples_per_prompt):
                response = generate_response(model, tokenizer, prompt_text, config)

                record = {
                    "task_id": problem["task_id"],
                    "prompt": problem["prompt"],
                    "completion": response,
                    "t_train": config.t_train,
                    "top_k": config.gen_top_k,
                    "top_p": config.gen_top_p,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)

            # Progress
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate if rate > 0 else 0

            if (i + 1) % 10 == 0 or i == 0:
                # Quick quality check: does the output contain code?
                has_code = "def " in response or "class " in response or "return " in response
                print(
                    f"  [{i+1:4d}/{len(prompts)}]  "
                    f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining  "
                    f"{'✓ code' if has_code else '✗ no code'}"
                )

    elapsed = time.time() - t0

    # ── Summary statistics ────────────────────────────────────
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
    print(f"\n  ⚠ Remember: SSD uses these raw outputs WITHOUT correctness filtering!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
