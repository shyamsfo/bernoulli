"""Scorer interface + HFScorer (dev/test backend).

The Scorer protocol is the swappable backbone seam — any VLM or decoder-only LM
that transformers or vLLM can load slots in via config.model_id. VLLMScorer
lands in M4 for production.

Backbone dispatch is config-driven, not hard-coded: `_load_backbone()` reads
`AutoConfig.from_pretrained(model_id)` and routes vision-language models (those
with a `vision_config` sub-config, e.g. Qwen2.5-VL, Qwen3-VL, Llama-3.2-Vision)
to `AutoModelForImageTextToText`, and text-only models to `AutoModelForCausalLM`.
Both expose `.forward() -> ... .logits` of shape (batch, seq, vocab_size), which
is the only interface Bernoulli actually needs.

To swap backbones: set `BERNOULLI_MODEL_ID` + `BERNOULLI_MODEL_REVISION` env vars.
No code change needed as long as (a) the model's tokenizer encodes letters A-Z
and Yes/No as single tokens (checked by `assert_single_token_labels` at scorer
construction), and (b) the model is registered under one of the two Auto classes
above (nearly every modern HF-hosted LM is).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@runtime_checkable
class Scorer(Protocol):
    """Return raw logits over a restricted set of label tokens at the answer position.

    One forward pass per `score` call. Debiasing (reversed/cyclic option order)
    and calibration (temperature scaling) sit above this interface. Scorer is
    intentionally dumb.

    `score_batch` is the batched variant — call this when scoring multiple
    prompts for the same decision (e.g., all reverse/cyclic permutations). On
    VLLMScorer this is one engine call with continuous batching; HFScorer
    falls back to a sequential loop (padding + attention masks are deferred).
    """

    model_id: str
    revision: str | None

    def score(
        self,
        prompt: str,
        allowed_token_ids: list[int],
        *,
        images: list[str] | None = None,
    ) -> NDArray[np.float32]:
        """One forward pass. Return logits (float32) in the order of allowed_token_ids."""
        ...

    def score_batch(
        self,
        prompts: list[str],
        allowed_token_ids_list: list[list[int]],
        *,
        images_list: list[list[str] | None] | None = None,
    ) -> list[NDArray[np.float32]]:
        """Batched score. Same semantics as `score`, N prompts in, N arrays out."""
        ...


def _is_vl_config(config: object) -> bool:
    """Does this backbone's config describe a vision-language model?

    The signal we use: presence of a non-None `vision_config` sub-config.
    This holds for every VLM in the HF ecosystem we're aware of (Qwen2.5-VL,
    Qwen3-VL, Llama-3.2-Vision, Gemma-3-Vision, Pixtral, InternVL, LLaVA, ...)
    without needing a hard-coded list of model_type strings.
    """
    vc = getattr(config, "vision_config", None)
    return vc is not None


def _load_backbone(
    model_id: str,
    revision: str | None,
    torch_dtype: object,
    device: str,
) -> PreTrainedModel:
    """Load a backbone via config-driven Auto* dispatch.

    - VL (has `vision_config`): `AutoModelForImageTextToText`
    - Text-only: `AutoModelForCausalLM`

    Both expose `forward() -> ... .logits` of shape (batch, seq, vocab_size),
    which is the only interface Bernoulli actually calls into.
    """
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model_cls = AutoModelForImageTextToText if _is_vl_config(config) else AutoModelForCausalLM
    return cast(
        "PreTrainedModel",
        model_cls.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device,
        ),
    )


class HFScorer:
    """transformers-backed scorer for the dev / test path.

    Loads the full model at construction and asserts single-token labels on the
    tokenizer. All subsequent `score` calls run under `torch.no_grad()` and
    return numpy arrays decoupled from the GPU.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        from bernoulli.labels import assert_single_token_labels

        self.model_id = model_id
        self.revision = revision
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        assert_single_token_labels(self.tokenizer)

        torch_dtype = getattr(torch, dtype)
        self._model = _load_backbone(model_id, revision, torch_dtype, device)
        self._model.eval()  # type: ignore[no-untyped-call]
        self._torch = torch

    def score(
        self,
        prompt: str,
        allowed_token_ids: list[int],
        *,
        images: list[str] | None = None,
    ) -> NDArray[np.float32]:
        if images:
            raise NotImplementedError(
                "image scoring lands in M4 (needs AutoProcessor + torchvision)"
            )

        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        last_logits = outputs.logits[0, -1, :]
        selected = last_logits[allowed_token_ids].float().cpu().numpy()
        return np.asarray(selected, dtype=np.float32)

    def score_batch(
        self,
        prompts: list[str],
        allowed_token_ids_list: list[list[int]],
        *,
        images_list: list[list[str] | None] | None = None,
    ) -> list[NDArray[np.float32]]:
        """Sequential loop over `score`. HF-level padding + attention-mask
        batching is deferred — the real batching win is on the vLLM path."""
        if len(prompts) != len(allowed_token_ids_list):
            raise ValueError(
                f"prompts len {len(prompts)} != allowed_token_ids len {len(allowed_token_ids_list)}"
            )
        imgs = images_list if images_list is not None else [None] * len(prompts)
        return [
            self.score(p, a, images=i)
            for p, a, i in zip(prompts, allowed_token_ids_list, imgs, strict=True)
        ]

    def generate(self, prompt: str, *, max_new_tokens: int = 10) -> str:
        """Greedy text generation. Baseline-only path — not part of the Scorer protocol.

        Lives on HFScorer so the generative-baseline comparison in evals/baselines.py
        can share the model + tokenizer that were loaded at construction. vLLM has
        its own generate path (lands in M4).
        """
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Cast locally — `.generate()` lives on `GenerationMixin`, which every
        # instruct-tuned backbone inherits, but it isn't on PreTrainedModel's
        # own stubs so mypy can't see it from the typed-attribute path.
        model = cast(Any, self._model)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        return cast(str, self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def load_scorer(settings: object | None = None) -> Scorer:
    """Factory: return an HFScorer or VLLMScorer based on config.scorer.

    `settings` defaults to `bernoulli.config.load_settings()`; pass an
    override for tests. Imports the vLLM backend lazily so the hf-only path
    doesn't pay for the vllm module (and its CUDA assertions) at import time.
    """
    from bernoulli.config import load_settings

    s = settings if settings is not None else load_settings()
    if getattr(s, "scorer", "hf") == "vllm":
        from bernoulli.vllm_scorer import VLLMScorer

        return VLLMScorer(
            s.model_id,  # type: ignore[attr-defined]
            revision=s.model_revision,  # type: ignore[attr-defined]
            dtype=s.dtype,  # type: ignore[attr-defined]
            max_model_len=s.max_model_len,  # type: ignore[attr-defined]
        )
    return HFScorer(
        s.model_id,  # type: ignore[attr-defined]
        revision=s.model_revision,  # type: ignore[attr-defined]
        dtype=s.dtype,  # type: ignore[attr-defined]
        device=s.device,  # type: ignore[attr-defined]
    )
