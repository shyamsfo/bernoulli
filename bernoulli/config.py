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
    model_id: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="HF hub id of the backbone VLM. See milestones.md for the dev vs production pick.",
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
