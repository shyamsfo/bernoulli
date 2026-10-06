# ToxicChat

Binary toxicity classification over real user queries collected from the LMSYS/Vicuna demo and hand-labeled. Known for distribution-shift from Jigsaw/PerspectiveAPI training data — classifiers that score 90%+ on Jigsaw often drop into the 60s here.

- **HF path**: [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat) · config `toxicchat1123`
- **Eval split**: `test` (5,082 examples).
- **Train split**: `train` (5,083 examples) — consumed by `BGEm3LR.fit` and other trainable baselines.

## Loader

```python
from benchmarks.guardrails.toxicchat import load, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text=user_input, question=BinaryQuestion("toxic"), gold='Yes'|'No')
```

The `state_text` is the raw user input (strip-only). The `jailbreaking` label travels in `meta` for future multi-head runs.

## Baselines applicable here

| Baseline            | Applies? | Note                                                                 |
|---------------------|----------|----------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | Binary — Yes/No single-token logit read.                              |
| `Generative`        | ✅       | Same backbone, text completion parsed.                                |
| `DeBERTaZeroshot`   | ❌       | Zero-shot pipeline needs natural-language labels (`"toxic"` / `"non-toxic"`), not Yes/No. Deferred. |
| `BGEm3LR`           | ✅       | `.fit(load_train())` on user_input. LR learns a 2-class head.         |

Domain-specific comparison targets (not yet wired as `Baseline` subclasses):

- **Llama Guard 3** / **ShieldGemma** / **OpenAI Moderation** — the paper's own leaderboard.

## Reword stems (stability test)

Three paraphrased stems in `loader.py :: REWORD_STEMS`.

## Notes

- The dataset ships `openai_moderation` scores per example — a natural comparison baseline, not wired here. See `meta` on each example.
- Not in the pre-M8 `evals/reports/`.
