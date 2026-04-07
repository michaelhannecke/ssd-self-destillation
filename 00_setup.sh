#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Step 0 — Environment Setup
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

echo "Setting up SSD tutorial environment..."

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "⚠  Warning: This tutorial is optimized for Apple Silicon (arm64)."
    echo "   Detected: $(uname -m)"
    echo "   MLX requires Apple Silicon. Proceeding anyway..."
fi

# Create venv if not active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Activated: .venv"
else
    echo "Using existing venv: $VIRTUAL_ENV"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    mlx \
    mlx-lm \
    datasets \
    numpy \
    matplotlib

# Verify
echo ""
echo "Verifying installation..."
python3 -c "
import mlx.core as mx
import mlx_lm
import datasets
import numpy
print(f'  mlx:      {mx.__version__}')
print(f'  mlx-lm:   OK')
print(f'  datasets:  {datasets.__version__}')
print(f'  numpy:    {numpy.__version__}')
print(f'  Metal:    {\"available\" if hasattr(mx, \"metal\") else \"not detected\"} ')
print()
print('All dependencies installed successfully.')
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Setup complete. Next:"
echo ""
echo "   source .venv/bin/activate  # if not already active"
echo "   python 01_generate.py --n-prompts 20  # smoke test"
echo "═══════════════════════════════════════════════════════════"
