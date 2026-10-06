# WildGuardTest

Response-harm binary classification over (prompt, response) pairs from AllenAI's WildGuardMix dataset. Target comparison: Llama Guard 3, ShieldGemma, WildGuard-7B — all of which also read token probabilities, so a like-for-like comparison is honest.

- **HF path**: [`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix) · config `wildguardtest`
- **Eval split**: `test` (1,725 examples).
- **Train split**: config `wildguardtrain` split `train` (~86k examples) — consumed by `BGEm3LR.fit` and other trainable baselines.

## State encoding

Each example's `state_text` combines the user prompt and the model response:

```
Prompt: <user prompt>
Response: <model response>
```

## Loader

```python
from benchmarks.guardrails.wildguard_test import load, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question=BinaryQuestion("response_harm"), gold='Yes'|'No')
```

Gold values are `"Yes"` (harmful) / `"No"` (unharmful). Rows without a response-harm annotation (prompt-only datapoints) are silently skipped.

### Multi-head framing (parked)

WildGuard annotates three heads per example: `prompt_harm_label`, `response_harm_label`, `response_refusal_label`. The natural Bernoulli framing is a single forward pass that produces probabilities for all three heads simultaneously — the whole "multiple questions over one state" story. Supporting this needs a schema change (`BenchmarkExample.question` → `questions: list[Question]`) and a corresponding metric runner that scores each head independently. Parked until the use-case push actually needs it; this first-pass benchmark reports response-harm only. The other two labels are preserved in `meta` so a future multi-head runner can read them without touching the loader.

## Baselines applicable here

| Baseline            | Applies? | Note                                                                     |
|---------------------|----------|--------------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | Binary — Yes/No single-token logit read.                                  |
| `Generative`        | ✅       | Same backbone, text completion parsed.                                    |
| `DeBERTaZeroshot`   | ❌       | Zero-shot pipeline needs natural-language labels (`"harmful"` / `"safe"`), not Yes/No. Will stay empty on this column until a per-benchmark adapter lands. |
| `BGEm3LR`           | ✅       | `.fit(load_train())` on the (prompt+response) state_text. LR learns a 2-class head.    |

Domain-specific comparison targets (not yet wired as `Baseline` subclasses; land per-benchmark when the first sweep runs):

- **Llama Guard 3** (`meta-llama/Llama-Guard-3-8B`) — single-pass harmfulness read-off.
- **ShieldGemma** (`google/shieldgemma-2b` or `-9b`) — Gemma-based safety classifier.
- **WildGuard-7B** (`allenai/wildguard`) — the paper's native model.

## Reword stems (stability test)

Three paraphrased stems in `loader.py :: REWORD_STEMS`.
