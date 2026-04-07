# SSD Tutorial — Step-by-Step Execution Guide

> Run Apple's "Embarrassingly Simple Self-Distillation" on your Mac Studio.  
> Total time: ~4–6 hours for the full pipeline, ~20 min for a smoke test.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| macOS | 14.0+ (Sonoma) | 15.0+ (Sequoia) |
| Apple Silicon | M1, 16 GB | M2 Ultra/M4, 64 GB |
| Python | 3.11 | 3.12 |
| Free disk | 15 GB | 30 GB |

The model alone takes ~2.5 GB (4-bit) or ~8 GB (fp16). Generation output,
training data, and adapters add another 1–5 GB depending on prompt count.

---

## Phase 0: Environment Setup

### 0.1 — Clone or copy the tutorial directory

```bash
# If you downloaded the zip:
cd ~/Projects  # or wherever you work
unzip ssd-tutorial.zip
cd ssd-tutorial

# Or if copied from Claude:
cd /path/to/ssd-tutorial
```

### 0.2 — Create and activate virtual environment

```bash
chmod +x 00_setup.sh
./00_setup.sh
```

If the script hangs on Metal detection, that's fine — it's an info check only.

**Manual alternative** (if the script fails):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm datasets numpy matplotlib
```

### 0.3 — Verify the install

```bash
python3 -c "import mlx.core as mx; print(f'MLX {mx.__version__}, Metal available')"
python3 -c "from mlx_lm import load; print('mlx-lm OK')"
python3 -c "from datasets import load_dataset; print('datasets OK')"
```

All three should print without errors. If `mlx` fails, you're likely not on
Apple Silicon or need to update macOS.

### 0.4 — Check model availability

```bash
python3 -c "
from mlx_lm import load
model, tok = load('mlx-community/Qwen3-4B-Instruct-2507-4bit')
print(f'Model loaded. Vocab size: {tok.vocab_size}')
"
```

**If this fails with a 404 or model-not-found error**, the MLX-converted
Qwen3-4B isn't available yet. Switch to a fallback:

```bash
# Edit config.py line ~28, change model_id to:
model_id: str = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
```

Then re-run the check. The first download takes 2–5 min depending on
connection speed.

> **Checkpoint:** You should see "Model loaded" with a vocab size in the
> 100K–150K range. If yes, proceed.

---

## Phase 1: Smoke Test (~20 minutes)

Run the full pipeline on a tiny subset first. This catches config errors,
missing dependencies, and API mismatches before you commit hours.

### 1.1 — Generate 20 samples

```bash
source .venv/bin/activate  # if not already active
python 01_generate.py --n-prompts 20
```

**Expected output:**

```
Loading model: mlx-community/Qwen3-4B-Instruct-4bit
Model loaded.

Loaded 20 prompts from mbpp
Generating 1 sample(s) per prompt
  T_train=1.6, top_k=20, top_p=0.8
  max_tokens=4096
  Output: ./ssd_run/raw_samples.jsonl

  [   1/20]  8s elapsed, ~152s remaining  ✓ code
  [  10/20]  72s elapsed, ~72s remaining  ✓ code
  [  20/20]  140s elapsed, ~0s remaining  ✗ no code

============================================================
Generation complete
  Samples: 20
  With extractable code: 14/20 (70.0%)
  Avg completion length: 1823 chars
  ...
  ⚠ Remember: SSD uses these raw outputs WITHOUT correctness filtering!
============================================================
```

**What to check:**
- Speed should be ~3–8s per sample on M2 Ultra. If >30s, the model may be
  swapping to disk — check Activity Monitor for memory pressure.
- "With extractable code" around 50–80% is normal at T=1.6. The paper's
  Section 4.4 shows that even 38% code rate still produces gains.
- Some `✗ no code` is expected and desired — this IS the training data.

**If you see errors:**
- `OutOfMemoryError`: Reduce `gen_max_tokens` in `config.py` to 2048.
- `KeyError: '<|im_end|>'`: Your model doesn't use Qwen-style tokens.
  Switch to a Qwen model or adjust `format_prompt()` in `config.py`.

### 1.2 — Prepare training data

```bash
python 02_prepare_data.py
```

**Expected output:**

```
Loading raw samples from ./ssd_run/raw_samples.jsonl
  Total raw samples: 20
  After degeneracy filter: 18 kept, 2 removed
  Retention rate: 90.0%

  Quality snapshot (no filtering applied, just stats):
    Contains Python keywords: 14/18 (77.8%)
    Possible gibberish at end: 1/18 (5.6%)
    → SSD trains on ALL of these. Bad data, good results (Section 4.4).

  Output directory: ./ssd_run/train_data
  Train: 17 samples → ./ssd_run/train_data/train.jsonl
  Valid: 1 samples → ./ssd_run/train_data/valid.jsonl
```

**What to check:**
- `train.jsonl` and `valid.jsonl` exist and are non-empty.
- Quick sanity check on the content:

```bash
head -c 500 ./ssd_run/train_data/train.jsonl
```

You should see JSON lines with `<|im_start|>` chat markup and Python code.

### 1.3 — Run a short LoRA training

```bash
ITERS=50 ./03_train.sh
```

**Expected output:**

```
═══════════════════════════════════════════════════════════
 SSD Step 3: LoRA Fine-Tuning
═══════════════════════════════════════════════════════════

 Model:       mlx-community/Qwen3-4B-Instruct-4bit
 Iterations:  50
 ...

Iter 10: Train loss 1.823, Learning Rate 5.00e-06, ...
Iter 20: Train loss 1.654, ...
Iter 30: Train loss 1.512, ...
Iter 40: Train loss 1.398, ...
Iter 50: Train loss 1.301, Val loss 1.456, ...

 Training complete.
 Adapter saved to: ./ssd_run/adapters
```

**What to check:**
- Training loss should decrease. If it stays flat or increases, the data
  format may be wrong — check `train.jsonl` encoding.
- Adapter files should exist:

```bash
ls -la ./ssd_run/adapters/
# Should show: adapters.safetensors, adapter_config.json, lora_config.yaml
```

**Common issues:**
- `No such file train.jsonl`: Check path. `mlx-lm` expects `train.jsonl`
  in the data directory, not `train.json`.
- `CUDA not available`: Ignore — MLX uses Metal, not CUDA.
- Very slow (<1 iter/min): Normal for larger sequence lengths. Reduce
  `--max-seq-length` to 2048 in `03_train.sh`.

### 1.4 — Quick evaluation

```bash
python 04_eval.py --n-problems 10 --n-samples 3
```

**Expected output:**

```
Loaded 10 HumanEval problems

═══════════════════════════════════════════════════════════
  Evaluating: Base Model
  T_eval=0.7, top_p=0.8, top_k=20
  3 samples × 10 problems
═══════════════════════════════════════════════════════════

  [  1/10]  running pass@1=0.333  (15s elapsed)
  [ 10/10]  running pass@1=0.467  (148s elapsed)

  Results for Base Model:
    pass@1 = 0.4667 (46.7%)

═══════════════════════════════════════════════════════════
  Evaluating: SSD Model (LoRA)
  ...

  Results for SSD Model (LoRA):
    pass@1 = 0.5000 (50.0%)

═══════════════════════════════════════════════════════════
  COMPARISON
═══════════════════════════════════════════════════════════
  Base model pass@1:  46.7%  (T=0.7)
  SSD model pass@1:   50.0%  (T=1.1)
  Delta:              +3.3pp
═══════════════════════════════════════════════════════════
```

**Note:** On 10 problems with 3 samples, results will be noisy. A positive
delta is encouraging but not statistically reliable yet. That's fine for
the smoke test — you just want to confirm the pipeline runs end-to-end.

> **Checkpoint:** If all 4 steps completed without crashes, the smoke test
> passes. Proceed to the full run.

---

## Phase 2: Full Run (~4–6 hours)

### 2.1 — Generate full training data (~45–90 min)

```bash
python 01_generate.py
```

This generates ~900 samples from all MBPP training prompts. On a Mac Studio
M2 Ultra with 64 GB, expect ~5s per sample → ~75 minutes total.

**Monitor resource usage during generation:**

```bash
# In a second terminal:
watch -n 5 'echo "=== Memory ===" && vm_stat | head -5 && echo "=== GPU ===" && sudo powermetrics --samplers gpu_power -i 1000 -n 1 2>/dev/null | grep "GPU"'

# Or simply:
top -l 1 | grep PhysMem
```

If "Memory pressure" in Activity Monitor goes to yellow/red, the model is
too large. Switch to a 4-bit quant or a smaller model.

### 2.2 — Prepare data (~5 seconds)

```bash
python 02_prepare_data.py
```

Verify the output before training:

```bash
wc -l ./ssd_run/train_data/train.jsonl   # Should be ~850–900 lines
wc -l ./ssd_run/train_data/valid.jsonl   # Should be ~45–50 lines
```

### 2.3 — LoRA training (~30–60 min)

```bash
./03_train.sh
```

**What healthy training looks like:**

```
Iter   50: Train loss 2.134, Val loss 2.267
Iter  100: Train loss 1.845, Val loss 2.098   ← decreasing
Iter  200: Train loss 1.523, Val loss 1.812   ← still decreasing
Iter  300: Train loss 1.312, Val loss 1.734   ← gap widening slightly
Iter  400: Train loss 1.198, Val loss 1.701   ← val plateauing
Iter  500: Train loss 1.087, Val loss 1.695   ← done
```

**Red flags:**
- Val loss increases consistently after iter 100 → overfitting. Reduce
  `ITERS` to where val loss was lowest, or increase `LORA_RANK`.
- Train loss stuck > 3.0 → learning rate too low or data format issue.
- NaN loss → learning rate too high. Reduce `LR` to `1e-6`.

If you see overfitting, you can use an earlier checkpoint:

```bash
ls ./ssd_run/adapters/
# checkpoint-100/ checkpoint-200/ ... adapters.safetensors (latest)

# To eval with an earlier checkpoint:
python 04_eval.py  # edit config.py adapter_path to point to checkpoint-200/
```

### 2.4 — Full evaluation (~1–2 hours)

```bash
python 04_eval.py
```

This runs 10 samples each on all 164 HumanEval problems, for both
base and SSD model. Expect ~90 min total (two full passes).

**Expected results range (LoRA on 4-bit Qwen3-4B):**

| Metric | Base | SSD | Delta |
|--------|------|-----|-------|
| pass@1 | 45–55% | 47–59% | +2–4pp |
| pass@5 | 60–70% | 63–75% | +3–5pp |

If your delta is negative, check:
1. Did training converge? (Val loss should be < train loss × 1.5)
2. Is the eval temperature correct? (SSD should use T=1.1, not T=0.7)
3. Try a different checkpoint (iter 200, 300, etc.)

### 2.5 — Temperature sweep (~2–4 hours)

This is the most important step for the article.

```bash
# Recommended: start with a reduced sweep to save time
python 05_sweep.py --n-problems 50 --n-samples 5 --temps "0.5,0.7,0.9,1.1,1.3,1.5"

# Full sweep (if the reduced version looks promising)
python 05_sweep.py --n-problems 80 --n-samples 5
```

**Expected output pattern:**

```
═══════════════════════════════════════════════════════════
  TEMPERATURE SWEEP SUMMARY
═══════════════════════════════════════════════════════════
  T_eval   Base pass@1   SSD pass@1     Delta
  ──────   ────────────   ────────────   ────────
     0.5        48.2%        50.4%     +2.2pp
     0.7        50.1%        52.0%     +1.9pp   ← base optimum here
     0.9        49.3%        53.8%     +4.5pp
     1.1        47.6%        54.1%     +6.5pp   ← SSD optimum here
     1.3        45.1%        51.2%     +6.1pp
     1.5        41.8%        47.3%     +5.5pp

  Best base:  50.1%
  Best SSD:   54.1%
  SSD margin: +4.0pp over best-tuned base
```

**The pattern you're looking for:**
1. Base sweep is flat — narrow range of ~3–5pp across all temps.
2. SSD curve peaks at a higher temperature than the base curve.
3. SSD's best point sits above the base curve's best point.

Even if the absolute margin is small (+2pp), the *shape* tells the story:
SSD changed the model distribution, not just the decoding.

### 2.6 — Generate the sweep plot

```bash
python ./ssd_run/eval_results/plot_sweep.py
```

Output: `./ssd_run/eval_results/sweep_plot.png` — this is the key figure
for the Medium article (analogous to Figure 2 in the paper).

---

## Phase 3: Optional Extensions

### 3A — "Bad Data, Good Results" Replication (Section 4.4)

The paper shows that SSD works even when 62% of training samples contain
no extractable code. To replicate:

```bash
# Generate with extreme temperature, no truncation
cp config.py config_backup.py

# Edit config.py:
#   t_train: float = 2.0
#   gen_top_k: int = 0      # disabled
#   gen_top_p: float = 1.0  # disabled

python 01_generate.py --n-prompts 200 --base-dir ./ssd_run_gibberish
python 02_prepare_data.py --base-dir ./ssd_run_gibberish
ITERS=300 DATA_DIR=./ssd_run_gibberish/train_data \
  ADAPTER_DIR=./ssd_run_gibberish/adapters ./03_train.sh
python 04_eval.py --base-dir ./ssd_run_gibberish --skip-base --n-problems 50

# Restore config
cp config_backup.py config.py
```

If the gibberish-trained model still beats the base, that's a strong
confirmation of the paper's central claim: the gain comes from
distributional reshaping, not from training on correct code.

### 3B — Different Model Comparison

Run the full pipeline on a second model to show generalization:

```bash
python 01_generate.py --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --base-dir ./ssd_run_coder3b
# ... repeat steps 2-5 with --base-dir ./ssd_run_coder3b
```

### 3C — pass@5 vs pass@1 Analysis

The paper emphasizes that pass@5 gains often exceed pass@1 gains,
indicating diversity preservation. After running `04_eval.py` with
`--n-samples 10`, compare the deltas:

```bash
python3 -c "
import json
with open('./ssd_run/eval_results/eval_results.json') as f:
    data = json.load(f)
for m in data:
    print(f\"{m['label']}: pass@1={m['pass@1']*100:.1f}%, pass@5={m.get('pass@5', 0)*100:.1f}%\")
"
```

If `delta_pass5 > delta_pass1`, that's evidence for the fork/lock mechanism.

---

## Troubleshooting Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: mlx` | Wrong Python / no venv | `source .venv/bin/activate` |
| Model download hangs | HuggingFace rate limit | Set `HF_TOKEN` env variable |
| Generation >30s per sample | Memory pressure, swapping | Use 4-bit model or reduce `gen_max_tokens` |
| `adapters.safetensors` missing | Training didn't complete | Check `training.log` for errors |
| Negative pass@1 delta | Overfitting or wrong temp | Use earlier checkpoint; verify T_eval |
| `subprocess.TimeoutExpired` in eval | Generated code has infinite loop | Normal — those samples count as failures |
| Plot script errors | Missing matplotlib | `pip install matplotlib` |
| `top_k` not working as expected | MLX array indexing issue | Test with `--n-prompts 5` first, check generated code quality |

---

## File Outputs After Full Run

```
ssd_run/
├── raw_samples.jsonl          # ~900 raw model outputs (Step 1)
├── train_data/
│   ├── train.jsonl            # ~855 training examples (Step 2)
│   └── valid.jsonl            # ~45 validation examples
├── adapters/
│   ├── adapters.safetensors   # Final LoRA weights (Step 3)
│   ├── adapter_config.json
│   ├── lora_config.yaml       # LoRA rank/scale/dropout config
│   └── training.log
└── eval_results/
    ├── eval_results.json      # pass@1, pass@5 per model (Step 4)
    ├── sweep_results.json     # Temperature sweep data (Step 5)
    ├── sweep_plot.png         # The key figure for the article
    └── plot_sweep.py          # Re-generate the plot anytime
```

---

## Time Budget Summary

| Step | Smoke test (20 prompts) | Full run (900 prompts) |
|------|------------------------|----------------------|
| 01 Generate | ~3 min | 45–90 min |
| 02 Prepare | < 5 sec | < 5 sec |
| 03 Train | ~3 min (50 iters) | 30–60 min (500 iters) |
| 04 Eval | ~5 min (10 problems) | 60–120 min (164 problems) |
| 05 Sweep | ~10 min (10 probs × 6 temps) | 2–4 h (80 probs × 11 temps) |
| **Total** | **~20 min** | **~4–6 hours** |

All times for Mac Studio M2 Ultra, 64 GB. Scale roughly linearly with
token generation speed on other Apple Silicon configs.
