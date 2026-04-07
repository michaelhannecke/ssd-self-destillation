# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction of "Embarrassingly Simple Self-Distillation Improves Code Generation" (Zhang et al., Apple, April 2026) adapted for Apple Silicon + MLX + LoRA. The pipeline self-samples from a code model at high temperature, fine-tunes on raw (unfiltered) outputs via LoRA, then evaluates whether the model distribution has improved.

Paper: arXiv:2604.01193 | Reference impl: github.com/apple/ml-ssd

## Pipeline

The project is a sequential 5-step pipeline. Each step is a numbered script:

```
00_setup.sh          → Environment setup (venv + deps)
01_generate.py       → Sample from base model (T=1.6, top_p=0.8) → ssd_run/raw_samples.jsonl
02_prepare_data.py   → Minimal degeneracy filter, format for mlx-lm → ssd_run/train_data/{train,valid}.jsonl
03_train.sh          → LoRA fine-tuning via `python -m mlx_lm lora` → ssd_run/adapters/
04_eval.py           → HumanEval pass@1/pass@5, base vs SSD → ssd_run/eval_results/eval_results.json
05_sweep.py          → Temperature sweep (Figure 2 repro) → ssd_run/eval_results/sweep_results.json + plot
```

All scripts share configuration from `config.py` (`SSDConfig` dataclass + prompt formatting).

## Commands

```bash
# Setup
source .venv/bin/activate
./00_setup.sh                    # or: pip install mlx mlx-lm datasets numpy matplotlib

# Smoke test (~15 min)
python 01_generate.py --n-prompts 20
python 02_prepare_data.py
ITERS=50 ./03_train.sh
python 04_eval.py --n-problems 10 --n-samples 3

# Full run
python 01_generate.py            # ~900 MBPP prompts, ~75 min on M2 Ultra
python 02_prepare_data.py
./03_train.sh                    # 500 iters, ~30-60 min
python 04_eval.py                # 164 HumanEval problems, ~90 min
python 05_sweep.py               # 11 temps x base+SSD, ~2-4 h
```

Training params are overridable via env vars: `ITERS`, `MODEL`, `DATA_DIR`, `ADAPTER_DIR`, `LR`, `LORA_RANK`, `LORA_LAYERS`, `BATCH_SIZE`, `GRAD_ACCUM`.

## Architecture Notes

- **config.py** is the single source of truth for all hyperparameters (model ID, temperatures, LoRA config, sweep temps). Also contains `format_prompt()` and `format_training_example()` which implement the Qwen3 `<|im_start|>` chat template.
- **No correctness filtering**: The core SSD insight is that raw model outputs (including gibberish) are used as training data without filtering. Only a minimal degeneracy filter (empty/single-line) is applied in `02_prepare_data.py`.
- **Eval executes generated code**: `04_eval.py` and `05_sweep.py` run model-generated Python in subprocesses with timeouts. This is inherent to HumanEval evaluation.
- **03_train.sh** wraps `python -m mlx_lm lora` — it generates a `lora_config.yaml` (for LoRA rank/scale/dropout) and does not contain custom training logic.
- **All outputs go to `ssd_run/`** (configurable via `--base-dir`).

## Key Differences from Paper

- LoRA (rank 16, 16 layers) instead of full SFT — limits distributional reshaping
- 4-bit quantized model instead of fp16
- MBPP (~900 prompts) instead of rSTARcoder (~10K)
- HumanEval eval instead of LiveCodeBench v6
- Expected gain: +2-4pp pass@1 (vs +7.5pp in paper)

## Dependencies

MLX ecosystem: `mlx`, `mlx-lm`, `datasets`, `numpy`, `matplotlib`. Python 3.11+, Apple Silicon required.
