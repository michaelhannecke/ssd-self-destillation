#!/usr/bin/env python3
"""
Step 2 — Prepare Training Data for LoRA Fine-Tuning
====================================================
Converts raw SSD samples into the JSONL format expected by mlx-lm's
LoRA trainer. Applies only the minimal degeneracy filter from the paper:
  - Remove empty responses
  - Remove single-line stubs

NO correctness filtering. This is the whole point of SSD.

Usage:
    python 02_prepare_data.py
    python 02_prepare_data.py --min-length 50  # chars, adjust filter threshold
"""

import argparse
import json
import random
from pathlib import Path

from config import SSDConfig, format_training_example


def minimal_degeneracy_filter(completion: str, min_length: int = 20) -> bool:
    """
    Paper Section 3.1: "We apply only a minimal degeneracy filter to remove
    clearly unusable outputs, such as empty responses and single-line stubs."

    Returns True if the sample should be KEPT.
    """
    # Empty or near-empty
    stripped = completion.strip()
    if len(stripped) < min_length:
        return False

    # Single-line stub
    lines = [l for l in stripped.split("\n") if l.strip()]
    if len(lines) <= 1:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="SSD Step 2: Prepare LoRA Data")
    parser.add_argument("--base-dir", type=str, default="./ssd_run")
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SSDConfig(base_dir=args.base_dir)
    random.seed(args.seed)

    # ── Load raw samples ──────────────────────────────────────
    print(f"Loading raw samples from {config.raw_samples_path}")
    samples = []
    with open(config.raw_samples_path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  Total raw samples: {len(samples)}")

    # ── Apply minimal filter ──────────────────────────────────
    kept = []
    removed = 0
    for s in samples:
        if minimal_degeneracy_filter(s["completion"], args.min_length):
            kept.append(s)
        else:
            removed += 1

    print(f"  After degeneracy filter: {len(kept)} kept, {removed} removed")
    print(f"  Retention rate: {100*len(kept)/max(len(samples),1):.1f}%")

    # ── Quick data quality analysis (for the article) ─────────
    has_code = sum(
        1 for s in kept
        if any(kw in s["completion"] for kw in ["def ", "class ", "return ", "import "])
    )
    has_gibberish = sum(
        1 for s in kept
        if any(ord(c) > 0x4E00 for c in s["completion"][-200:])  # CJK chars at end
    )
    print(f"\n  Quality snapshot (no filtering applied, just stats):")
    print(f"    Contains Python keywords: {has_code}/{len(kept)} ({100*has_code/max(len(kept),1):.1f}%)")
    print(f"    Possible gibberish at end: {has_gibberish}/{len(kept)} ({100*has_gibberish/max(len(kept),1):.1f}%)")
    print(f"    → SSD trains on ALL of these. Bad data, good results (Section 4.4).\n")

    # ── Format for mlx-lm LoRA ────────────────────────────────
    formatted = []
    for s in kept:
        text = format_training_example(s["prompt"], s["completion"])
        formatted.append({"text": text})

    # ── Train/valid split ─────────────────────────────────────
    random.shuffle(formatted)
    n_val = max(1, int(len(formatted) * config.val_split))
    val_data = formatted[:n_val]
    train_data = formatted[n_val:]

    # ── Write output ──────────────────────────────────────────
    config.train_data_dir.mkdir(parents=True, exist_ok=True)

    train_path = config.train_data_dir / "train.jsonl"
    valid_path = config.train_data_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── Length statistics for training config ──────────────────
    lengths = [len(item["text"]) for item in train_data]
    avg_len = sum(lengths) / max(len(lengths), 1)
    max_len = max(lengths) if lengths else 0
    p95_len = sorted(lengths)[int(0.95 * len(lengths))] if lengths else 0

    print(f"  Output directory: {config.train_data_dir}")
    print(f"  Train: {len(train_data)} samples → {train_path}")
    print(f"  Valid: {len(val_data)} samples → {valid_path}")
    print(f"\n  Token length estimates (chars, rough):")
    print(f"    Average: {avg_len:.0f}")
    print(f"    P95:     {p95_len}")
    print(f"    Max:     {max_len}")

    # ── Print sample for sanity check ─────────────────────────
    if train_data:
        print(f"\n{'─'*60}")
        print("Sample training example (first 500 chars):")
        print(f"{'─'*60}")
        print(train_data[0]["text"][:500])
        print("...")


if __name__ == "__main__":
    main()
