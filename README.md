# SSD Tutorial: Self-Distillation for Code Generation on Apple Silicon

Hands-on reproduction of **"Embarrassingly Simple Self-Distillation Improves
Code Generation"** (Zhang et al., Apple, April 2026) — adapted for
**Apple Silicon + MLX + LoRA**.

Paper: [arXiv:2604.01193](https://arxiv.org/abs/2604.01193)  
Code: [github.com/apple/ml-ssd](https://github.com/apple/ml-ssd)

---

## What This Does

SSD is a 3-step recipe that improves a code model using only its own raw outputs:

```
┌─────────────┐      ┌───────────────┐      ┌──────────────┐
│  1. SAMPLE   │ ──▶  │  2. FINE-TUNE  │ ──▶  │  3. DECODE   │
│  T=1.6       │      │  SFT on raw    │      │  T=1.1       │
│  top_k=20    │      │  outputs       │      │  top_k=20    │
│  top_p=0.8   │      │  (no filter!)  │      │  top_p=0.8   │
└─────────────┘      └───────────────┘      └──────────────┘

  ✗ No RL          ✗ No verifier      ✗ No teacher
  ✗ No execution   ✗ No correctness filtering
```

## Honest Differences vs. Paper

| Aspect | Paper | This Tutorial |
|--------|-------|---------------|
| Hardware | 8×B200 GPUs | Mac Studio M-series |
| Inference | vLLM (CUDA) | mlx-lm (Metal) |
| Training | Full SFT, Megatron-LM | LoRA, mlx-lm |
| Model | Qwen3-4B-Instruct (fp16) | Qwen3-4B-Instruct (4-bit) |
| Train prompts | ~10K (rSTARcoder) | ~900 (MBPP) |
| Eval | LiveCodeBench v6 | HumanEval |
| Expected gain | +7.5pp pass@1 | +2–4pp pass@1 (estimated) |

The LoRA compromise is real: full SFT can reshape all model weights including
the "support compression" effect (Section 4.3 in the paper). LoRA only touches
a subset, which limits how aggressively distractor tails can be suppressed.

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4, 16GB+ unified memory, 64GB recommended)
- Python 3.11+

## Setup

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate

# Core dependencies
pip install mlx mlx-lm datasets numpy

# For evaluation plots
pip install matplotlib

# For HumanEval execution (already in stdlib, but verify)
python -c "import subprocess, tempfile; print('OK')"
```

## Pipeline

Run each step sequentially:

```bash
# Step 1: Sample solutions from the base model (~1-3h on M2 Ultra)
python 01_generate.py
python 01_generate.py --n-prompts 50  # quick test first

# Step 2: Format data for LoRA training (~seconds)
python 02_prepare_data.py

# Step 3: LoRA fine-tuning (~30-60min)
chmod +x 03_train.sh
./03_train.sh

# Step 4: Evaluate base vs. SSD on HumanEval (~1-2h)
python 04_eval.py
python 04_eval.py --n-problems 30 --n-samples 3  # quick test

# Step 5: Temperature sweep — the key result (~2-4h full, 30min quick)
python 05_sweep.py --n-problems 30 --n-samples 3
python 05_sweep.py  # full run
```

### Quick Smoke Test (~15 min)

```bash
python 01_generate.py --n-prompts 20
python 02_prepare_data.py
ITERS=50 ./03_train.sh
python 04_eval.py --n-problems 10 --n-samples 3
```

---

## What to Look For

### The temperature sweep is the money shot

The paper's Figure 2 shows that sweeping T_eval on the base model yields a
*flat curve* (±2pp), while SSD sits above the curve's peak. This proves that
SSD changes the model distribution itself — not just the decoding.

With LoRA, the margin will be smaller, but the pattern should be visible:
the SSD line should sit above or at the upper bound of the base sweep.

### Diversity preservation (pass@5 > pass@1 gains)

If pass@5 improves more than pass@1, that's evidence that SSD preserves
exploration at fork positions while tightening lock positions (Section 4.1).

### Bad data still works

Try generating with a very high temperature:
```bash
python 01_generate.py --n-prompts 100
# Then edit config.py: t_train = 2.0, gen_top_k = 0, gen_top_p = 1.0
# This replicates Section 4.4 — "Bad Data, Good Results"
```

---

## File Overview

```
ssd-tutorial/
├── config.py           # All hyperparameters (Tables 3, 4 from paper)
├── 01_generate.py      # Step 1: Self-sampling with T_train + truncation
├── 02_prepare_data.py  # Step 2: Format for LoRA (minimal filter only)
├── 03_train.sh         # Step 3: LoRA fine-tuning
├── 04_eval.py          # Step 4: HumanEval pass@1/pass@5
├── 05_sweep.py         # Step 5: Temperature sweep (Figure 2 repro)
└── README.md           # This file
```

---

## Model Alternatives

If `mlx-community/Qwen3-4B-Instruct-4bit` is not available, try:

```python
# In config.py, change model_id to:
"mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"  # good coding model
"mlx-community/Qwen2.5-3B-Instruct-4bit"         # general instruct
```

Adjust `t_train` and `t_eval` if using a different model — the paper's
values are specific to Qwen3-4B-Instruct.

---

## License

Tutorial code: MIT  
Paper: © Apple, see [arxiv.org/abs/2604.01193](https://arxiv.org/abs/2604.01193)
