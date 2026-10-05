"""Scorer interface + HFScorer (dev/test backend).

The Scorer protocol is the swappable backbone seam — any VLM that transformers
or vLLM can load slots in via config.model_id. VLLMScorer lands in M4 for
production. For M2 we only need HFScorer, text-only (images in M4).

Hard-coded model class: Qwen2.5-VL is loaded via Qwen2_5_VLForConditionalGeneration.
When M4 adds a second production backbone, generalize via a small registry
(auto-dispatch on config.name_or_path). Not worth building the registry for one
model today.
"""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Scorer(Protocol):
    """Return raw logits over a restricted set of label tokens at the answer position.

    One forward pass per `score` call. Debiasing (reversed/cyclic option order)
    and calibration (temperature scaling) sit above this interface. Scorer is
    intentionally dumb.
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
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        from bernoulli.labels import assert_single_token_labels

        self.model_id = model_id
        self.revision = revision
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        assert_single_token_labels(self.tokenizer)

        torch_dtype = getattr(torch, dtype)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,  # type: ignore[arg-type]
            torch_dtype=torch_dtype,
            device_map=device,
        )
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

    def generate(self, prompt: str, *, max_new_tokens: int = 10) -> str:
        """Greedy text generation. Baseline-only path — not part of the Scorer protocol.

        Lives on HFScorer so the generative-baseline comparison in evals/baselines.py
        can share the model + tokenizer that were loaded at construction. vLLM has
        its own generate path (lands in M4).
        """
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[misc]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        return cast(str, self.tokenizer.decode(new_tokens, skip_special_tokens=True))
