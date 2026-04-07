#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Step 3 — LoRA Fine-Tuning on Self-Generated Data
# ═══════════════════════════════════════════════════════════════
#
# This trains a LoRA adapter using mlx-lm on the SSD dataset
# generated in Steps 1-2.
#
# Key difference vs. paper:
#   Paper uses full SFT with Megatron-LM on 8×B200 GPUs.
#   We use LoRA on Apple Silicon — a meaningful compromise that
#   limits how much the distribution can be reshaped, but makes
#   the whole pipeline runnable on a single Mac Studio.
#
# Usage:
#   chmod +x 03_train.sh && ./03_train.sh
#   ./03_train.sh --model mlx-community/Qwen3-4B-Instruct-2507-4bit

set -euo pipefail

# ── Defaults (override via env or args) ───────────────────────
MODEL="${MODEL:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"
DATA_DIR="${DATA_DIR:-./ssd_run/train_data}"
ADAPTER_DIR="${ADAPTER_DIR:-./ssd_run/adapters}"
ITERS="${ITERS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LORA_LAYERS="${LORA_LAYERS:-16}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-5e-6}"
WARMUP="${WARMUP:-50}"
SAVE_EVERY="${SAVE_EVERY:-100}"
VAL_EVERY="${VAL_EVERY:-50}"

# Allow first positional arg to override model
if [[ "${1:-}" == "--model" ]] && [[ -n "${2:-}" ]]; then
    MODEL="$2"
fi

echo "═══════════════════════════════════════════════════════════"
echo " SSD Step 3: LoRA Fine-Tuning"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo " Model:       $MODEL"
echo " Data:        $DATA_DIR"
echo " Adapters:    $ADAPTER_DIR"
echo " Iterations:  $ITERS"
echo " Batch size:  $BATCH_SIZE (× $GRAD_ACCUM accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective)"
echo " LoRA:        $LORA_LAYERS layers, rank $LORA_RANK"
echo " LR:          $LR"
echo ""
echo " ⚠  Paper uses full SFT on 8×B200. LoRA is a compromise."
echo "    Expect smaller but directionally correct gains."
echo ""

mkdir -p "$ADAPTER_DIR"

# ── Run training ──────────────────────────────────────────────
python -m mlx_lm.lora \
    --model "$MODEL" \
    --data "$DATA_DIR" \
    --train \
    --adapter-path "$ADAPTER_DIR" \
    --iters "$ITERS" \
    --batch-size "$BATCH_SIZE" \
    --grad-checkpoint \
    --num-layers "$LORA_LAYERS" \
    --lora-rank "$LORA_RANK" \
    --learning-rate "$LR" \
    --steps-per-eval "$VAL_EVERY" \
    --save-every "$SAVE_EVERY" \
    --max-seq-length 4096 \
    2>&1 | tee "$ADAPTER_DIR/training.log"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Training complete."
echo " Adapter saved to: $ADAPTER_DIR"
echo " Training log:     $ADAPTER_DIR/training.log"
echo ""
echo " Next: python 04_eval.py"
echo "═══════════════════════════════════════════════════════════"
