# Embarrassingly Simple Self-Distillation Improves Code Generation

> A hands-on reproduction of [Zhang et al. (Apple, 2026)](https://arxiv.org/abs/2604.01193) adapted for **Apple Silicon + MLX + LoRA**.

<p align="center">
  <img src="data/ssd-selfdestilled.jpg" alt="SSD Self-Distillation Overview" width="720"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.01193"><img src="https://img.shields.io/badge/arXiv-2604.01193-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/apple/ml-ssd"><img src="https://img.shields.io/badge/Reference-apple%2Fml--ssd-blue.svg" alt="Reference"></a>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/framework-MLX-orange.svg" alt="MLX">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

## Overview

Self-Distillation for Code Generation (SSD) is a remarkably simple technique that improves a code model using only its own raw outputs — no teacher model, no verifier, no reinforcement learning:

```
┌─────────────┐      ┌───────────────┐      ┌──────────────┐
│  1. SAMPLE   │ ──▶  │  2. FINE-TUNE  │ ──▶  │  3. DECODE   │
│  T=1.6       │      │  SFT on raw    │      │  T=1.1       │
│  top_k=20    │      │  outputs       │      │  top_k=20    │
│  top_p=0.8   │      │  (no filter!)  │      │  top_p=0.8   │
└─────────────┘      └───────────────┘      └──────────────┘
```

The key insight: training on the model's own high-temperature outputs — including incorrect and degenerate samples — reshapes the output distribution to concentrate probability mass on correct solutions. This works because self-distillation compresses the support of the sampling distribution, suppressing distractor tails while preserving diversity at meaningful decision points.

## Key Results

| Metric | Base Model | SSD Model | Delta |
|--------|:----------:|:---------:|:-----:|
| pass@1 | 45-55% | 47-59% | **+2-4pp** |
| pass@5 | 60-70% | 63-75% | **+3-5pp** |

*Results on HumanEval with LoRA fine-tuned Qwen3-4B-Instruct (4-bit). The original paper reports +7.5pp with full SFT.*

---

## Differences from the Original Paper

| Aspect | Paper | This Reproduction |
|--------|:-----:|:-----------------:|
| Hardware | 8x B200 GPUs | Mac Studio (M-series) |
| Inference | vLLM (CUDA) | mlx-lm (Metal) |
| Training | Full SFT, Megatron-LM | LoRA (rank 16), mlx-lm |
| Model | Qwen3-4B-Instruct (fp16) | Qwen3-4B-Instruct (4-bit) |
| Training prompts | ~10K (rSTARcoder) | ~900 (MBPP) |
| Evaluation | LiveCodeBench v6 | HumanEval |
| Expected gain | +7.5pp pass@1 | +2-4pp pass@1 |

The LoRA compromise is deliberate: full SFT can reshape all model weights including the "support compression" effect (Section 4.3), while LoRA only touches a subset. This limits distributional reshaping but makes the experiment feasible on consumer hardware.

---

## Requirements

| Requirement | Minimum | Recommended |
|:-----------:|:-------:|:-----------:|
| macOS | 14.0+ (Sonoma) | 15.0+ (Sequoia) |
| Apple Silicon | M1, 16 GB | M2 Ultra / M4, 64 GB |
| Python | 3.11 | 3.12+ |
| Disk space | 15 GB | 30 GB |

## Installation

```bash
git clone https://github.com/your-repo/ssd-self-distillation.git
cd ssd-self-distillation

# Option A: automated setup
chmod +x 00_setup.sh && ./00_setup.sh

# Option B: manual setup
python3 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm datasets numpy matplotlib
```

Verify the installation:

```bash
python3 -c "import mlx.core as mx; print(f'MLX {mx.__version__} — Metal available')"
python3 -c "from mlx_lm import load; print('mlx-lm OK')"
```

---

## Pipeline

The project consists of five sequential steps. All outputs are written to `ssd_run/` by default.

### Quick Smoke Test (~20 minutes)

```bash
source .venv/bin/activate
python 01_generate.py --n-prompts 20       # Sample 20 prompts
python 02_prepare_data.py                  # Format training data
ITERS=50 ./03_train.sh                    # Short LoRA training
python 04_eval.py --n-problems 10 --n-samples 3  # Quick eval
```

### Full Run (~4-6 hours)

```bash
# Step 1: Self-sampling from base model (~45-90 min)
python 01_generate.py

# Step 2: Data preparation (~5 sec)
python 02_prepare_data.py

# Step 3: LoRA fine-tuning (~30-60 min)
./03_train.sh

# Step 4: HumanEval evaluation — base vs SSD (~1-2 h)
python 04_eval.py

# Step 5: Temperature sweep — the key result (~2-4 h)
python 05_sweep.py
```

### Training Configuration

All hyperparameters are centralized in `config.py`. Training parameters can also be overridden via environment variables:

```bash
ITERS=300 LR=1e-5 LORA_RANK=32 BATCH_SIZE=2 ./03_train.sh
```

| Variable | Default | Description |
|----------|:-------:|-------------|
| `ITERS` | 500 | Training iterations |
| `LR` | 5e-6 | Learning rate |
| `LORA_RANK` | 16 | LoRA rank |
| `LORA_LAYERS` | 16 | Number of LoRA layers |
| `BATCH_SIZE` | 1 | Batch size |
| `GRAD_ACCUM` | 8 | Gradient accumulation steps |

---

## What to Look For

### Temperature Sweep (Figure 2 Reproduction)

The paper's central finding: sweeping evaluation temperature on the base model yields a **flat curve** (within ~3-5pp), while the SSD model consistently sits above the base curve's peak. This demonstrates that SSD changes the model's output distribution itself, not just the decoding strategy.

### Diversity Preservation

If pass@5 gains exceed pass@1 gains, this provides evidence for the fork/lock mechanism (Section 4.1): SSD tightens probability at "lock" positions (where there is one correct continuation) while preserving exploration at "fork" positions (where multiple valid continuations exist).

### Bad Data, Good Results (Section 4.4)

The paper shows that SSD works even when the majority of training samples contain no extractable code. This can be replicated by increasing `t_train` to 2.0 and disabling truncation.

---

## Project Structure

```
ssd-self-distillation/
├── config.py            # Centralized hyperparameters and prompt formatting
├── 00_setup.sh          # Environment setup
├── 01_generate.py       # Step 1: Self-sampling (T=1.6, top_p=0.8)
├── 02_prepare_data.py   # Step 2: Minimal degeneracy filter + formatting
├── 03_train.sh          # Step 3: LoRA fine-tuning via mlx-lm
├── 04_eval.py           # Step 4: HumanEval pass@1 / pass@5
├── 05_sweep.py          # Step 5: Temperature sweep + plot generation
├── GUIDE.md             # Detailed step-by-step execution guide
├── CLAUDE.md            # AI assistant context
└── README.md            # This file
```

### Output Structure

```
ssd_run/
├── raw_samples.jsonl             # Raw model outputs
├── train_data/
│   ├── train.jsonl               # Training examples
│   └── valid.jsonl               # Validation examples
├── adapters/
│   ├── adapters.safetensors      # Final LoRA weights
│   ├── adapter_config.json
│   └── lora_config.yaml
└── eval_results/
    ├── eval_results.json         # pass@1, pass@5 per model
    ├── sweep_results.json        # Temperature sweep data
    └── sweep_plot.png            # Key figure for reproduction
```

---

## Model Alternatives

If `mlx-community/Qwen3-4B-Instruct-4bit` is unavailable, update `model_id` in `config.py`:

```python
"mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"  # Strong coding model
"mlx-community/Qwen2.5-3B-Instruct-4bit"         # General instruct
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: mlx` | Wrong Python or no venv | `source .venv/bin/activate` |
| Generation >30s/sample | Memory pressure | Use 4-bit model or reduce `gen_max_tokens` |
| Training loss flat >3.0 | Learning rate too low | Increase `LR` to `1e-5` |
| NaN loss | Learning rate too high | Reduce `LR` to `1e-6` |
| Negative pass@1 delta | Overfitting or wrong temp | Use earlier checkpoint; verify `T_eval` |
| Model download hangs | HuggingFace rate limit | Set `HF_TOKEN` env variable |

For a comprehensive walkthrough with expected outputs at each step, see **[GUIDE.md](GUIDE.md)**.

---

## References

```bibtex
@article{zhang2026ssd,
  title={Embarrassingly Simple Self-Distillation Improves Code Generation},
  author={Zhang, Haoran and Jain, Neel and Welleck, Sean and Bertsch, Amanda},
  journal={arXiv preprint arXiv:2604.01193},
  year={2026}
}
```

## License

Tutorial code: MIT | Paper: (c) Apple, see [arXiv:2604.01193](https://arxiv.org/abs/2604.01193)
