"""VLLMScorer — production path for text-only prompts.

Same Scorer protocol as HFScorer: one `score(prompt, allowed_token_ids)`
call per decision, returns raw logprobs (not logits) in the order of
`allowed_token_ids`. Softmax over the restricted set gives the same answer
either way — constants cancel.

How we read next-token probabilities out of vLLM:
    SamplingParams(max_tokens=1, logprobs=K, allowed_token_ids=label_ids)
vLLM then generates exactly one token (constrained to label_ids) and
reports the top-K unrestricted logprobs at that position. We pick out the
allowed labels from the returned dict; any missing label gets a very
negative logprob (treated as effectively zero probability after softmax).

K is sized to comfortably cover the full label set — 128 is well above the
26-letter max chunk + Yes/No, and well under vLLM's cap. If a label is
missing from top-K, that's a signal the model is strongly committing to
some other token, which means that label would've been near-zero probability
anyway.

Prefix caching + allowed_token_ids are the two features the vision doc §4
pins; we enable both and verify at construction time that vLLM honored them.

Images: M6, same as HFScorer today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bernoulli.labels import assert_single_token_labels

if TYPE_CHECKING:  # Avoid importing vllm at type-check time (heavy, GPU-only)
    pass

_LOGPROBS_TOPK = 128
_MISSING_LOGPROB = -1e9  # effectively -inf under softmax


class VLLMScorer:
    """vLLM-backed scorer for the production path.

    Loads the engine at construction, asserts single-token labels on the
    tokenizer, and keeps prefix caching enabled so that multiple questions
    sharing a state prefix pay for the prefix once.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_id = model_id
        self.revision = revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        assert_single_token_labels(self.tokenizer)

        self._llm = LLM(
            model=model_id,
            revision=revision,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            # Default is 20; we need up to _LOGPROBS_TOPK at request time to
            # cover 26-letter chunks + overhead.
            max_logprobs=_LOGPROBS_TOPK,
            # Qwen2.5-VL is a VL model, but text-only prompts don't trigger
            # the vision tower. Image path lands in M6.
            limit_mm_per_prompt={"image": 0},
            trust_remote_code=False,
        )

    def score(
        self,
        prompt: str,
        allowed_token_ids: list[int],
        *,
        images: list[str] | None = None,
    ) -> NDArray[np.float32]:
        if images:
            raise NotImplementedError("image scoring lands in M6")

        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=_LOGPROBS_TOPK,
            allowed_token_ids=allowed_token_ids,
        )
        outputs = self._llm.generate([prompt], params, use_tqdm=False)
        logprobs_list = outputs[0].outputs[0].logprobs
        if logprobs_list is None:
            raise RuntimeError("vLLM returned no logprobs; check SamplingParams.logprobs")
        logprobs_dict = logprobs_list[0]
        # logprobs_dict: {token_id: Logprob(logprob=..., rank=..., decoded_token=...)}
        selected = np.array(
            [
                getattr(logprobs_dict.get(tid), "logprob", _MISSING_LOGPROB)
                if tid in logprobs_dict
                else _MISSING_LOGPROB
                for tid in allowed_token_ids
            ],
            dtype=np.float32,
        )
        return selected
