"""Env-driven runtime config. All settings override via BERNOULLI_* env vars.

The backbone is intentionally not pinned at the type level — see
vision_and_roadmap.md's model-flexibility note. Any transformers-compatible VLM
can be dropped in via BERNOULLI_MODEL_ID; the pinned revision travels with it so
eval numbers reproduce.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Dtype = Literal["bfloat16", "float16", "float32"]
ScorerBackend = Literal["hf", "vllm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BERNOULLI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- backbone ---------------------------------------------------------
    # Dev default: Qwen2.5-VL-7B-Instruct (~15 GB, fits a 24 GB GPU).
    # The 32B-AWQ production step-up on L40S 48 GB is parked on hardware
    # availability; the serving code path is backbone-agnostic.
    #
    # Swapping backbones: set BERNOULLI_MODEL_ID + BERNOULLI_MODEL_REVISION
    # and restart. The scorer autodetects text-only vs vision-text and
    # dispatches to AutoModelForCausalLM or AutoModelForImageTextToText.
    # Two gotchas to verify on any new backbone:
    #   1. Letters A-Z + Yes/No must be single tokens on the tokenizer.
    #      Scorer startup asserts this and fails loud if not.
    #   2. Reasoning-mode models (Qwen3, GLM 4.5, DeepSeek R1) emit <think>
    #      before the answer, which breaks the next-token read. Disable
    #      thinking via chat template / system prompt, or pick a classical
    #      chat-tuned model (Qwen2.5, Llama 3, Mistral, Gemma 2, Phi 3, ...).
    # See HOW-IT-WORKS.md §7 for the full family-compatibility taxonomy.
    model_id: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="HF hub id of the backbone LM. Swap via env var; see HOW-IT-WORKS.md §7.",
    )
    model_revision: str | None = Field(
        default="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        description="Pinned HF revision SHA. None = latest at load time (not recommended).",
    )
    dtype: Dtype = "bfloat16"
    device: str = Field(
        default="cuda", description="torch device string, e.g. 'cuda', 'cuda:0', 'cpu'"
    )
    max_model_len: int = Field(default=32768, ge=1024, le=262144)

    # ---- scorer selection -------------------------------------------------
    scorer: ScorerBackend = Field(
        default="hf",
        description="'hf' = transformers (dev/tests), 'vllm' = production (M4+).",
    )

    # ---- paths ------------------------------------------------------------
    hf_home: str | None = Field(
        default=None,
        description="HF cache dir. Falls back to HF_HOME env, then ~/.cache/huggingface.",
    )

    # ---- debias defaults --------------------------------------------------
    default_debias: Literal["none", "reverse", "cyclic"] = "reverse"
    default_calibrated: bool = True


def load_settings() -> Settings:
    """Load settings from env + .env. Importable but not auto-run at import time."""
    return Settings()
