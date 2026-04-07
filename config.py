"""
SSD Tutorial Configuration
==========================
Hyperparameters aligned with:
  "Embarrassingly Simple Self-Distillation Improves Code Generation"
  (Zhang et al., Apple, April 2026)

Adapted for Apple Silicon / MLX with LoRA instead of full SFT.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SSDConfig:
    # ── Paths ──────────────────────────────────────────────────
    base_dir: Path = Path("./ssd_run")
    raw_samples_path: Path = field(default=None)
    train_data_dir: Path = field(default=None)
    adapter_path: Path = field(default=None)
    eval_results_path: Path = field(default=None)

    # ── Model ──────────────────────────────────────────────────
    # MLX-converted Qwen3-4B-Instruct.  Change to your local path
    # or any mlx-community variant, e.g.:
    #   "mlx-community/Qwen2.5-Coder-3B-Instruct-8bit"
    model_id: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    # ── Generation (Step 1) — Table 3 from paper ──────────────
    t_train: float = 1.6       # Training-time sampling temperature
    gen_top_p: float = 0.8     # Training-time nucleus threshold
    gen_top_k: int = 20        # Training-time top-k (needs custom sampler)
    gen_max_tokens: int = 4096
    n_samples_per_prompt: int = 1  # Paper uses N=1

    # ── Training (Step 2) ─────────────────────────────────────
    lora_layers: int = 16
    lora_rank: int = 16
    learning_rate: float = 5e-6
    batch_size: int = 1        # Limited by Apple Silicon memory
    grad_accum_steps: int = 4  # Effective batch = 4
    n_iters: int = 500
    warmup_iters: int = 50
    val_split: float = 0.05

    # ── Evaluation (Step 3) — Table 3 / Table 4 ──────────────
    t_eval_ssd: float = 1.1    # Eval temp for SSD model (Table 3)
    t_eval_base: float = 0.7   # Eval temp for base model (Table 4)
    eval_top_p: float = 0.8
    eval_top_k: int = 20
    eval_max_tokens: int = 2048
    n_eval_samples: int = 10   # For pass@k estimation

    # ── Sweep (Step 4) ────────────────────────────────────────
    sweep_temps: list = field(default_factory=lambda: [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
    ])
    sweep_n_samples: int = 5   # Samples per problem per temp

    # ── Prompt source ─────────────────────────────────────────
    # "mbpp" for training prompts, "humaneval" for eval
    train_dataset: str = "mbpp"
    eval_dataset: str = "humaneval"

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.raw_samples_path = self.raw_samples_path or self.base_dir / "raw_samples.jsonl"
        self.train_data_dir = self.train_data_dir or self.base_dir / "train_data"
        self.adapter_path = self.adapter_path or self.base_dir / "adapters"
        self.eval_results_path = self.eval_results_path or self.base_dir / "eval_results"


# ── Chat template for Qwen3-Instruct ──────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful programming assistant. "
    "Solve the given problem in Python. "
    "Provide your solution inside a ```python code block."
)


def format_prompt(problem_text: str) -> str:
    """Format a problem into the Qwen3 chat template."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_training_example(problem_text: str, completion: str) -> str:
    """Full chat turn for SFT training."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )
