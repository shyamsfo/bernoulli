# CLINC150-OOS

Out-of-scope detection over short user utterances. 150 in-scope intents across 10 domains (banking, credit cards, travel, small talk, etc.) plus an explicit **out-of-scope** class — the canonical benchmark for "none of the above" probability behavior.

- **HF path**: [`clinc_oos`](https://huggingface.co/datasets/clinc_oos) · config `plus`
- **Eval split**: `test` (5,500 examples: 4,500 in-scope + 1,000 OOS).
- **Train split**: `train` (15,250 examples: 15,000 in-scope + 250 OOS) — consumed by `BGEm3LR.fit`.

## Framing

Binary — "Is this message in-scope?" with gold `Yes` for in-scope rows, `No` for OOS. This matches the landing-page use-case card for support triage: *the probability flags "none of these"*. The 150-way intent classification is parked; the natural shape is a multi-head run (in-scope + which-intent-if-in-scope) which needs a schema change.

The OOS label id is resolved at load time via `features['intent'].names.index("oos")`. If the dataset ever renames that label, loading fails loudly.

## Loader

```python
from benchmarks.triage.clinc150_oos import load, load_train, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question=BinaryQuestion("in_scope"), gold='Yes'|'No')
```

Each example carries `meta.is_oos: bool`, `meta.intent_id: int`, and `meta.intent_name: str` for subset analysis.

## Baselines applicable here

| Baseline            | Applies? | Note                                                                 |
|---------------------|----------|----------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | Binary — Yes/No single-token logit read.                              |
| `Generative`        | ✅       | Same backbone, text completion parsed.                                |
| `DeBERTaZeroshot`   | ❌       | Zero-shot pipeline needs natural-language labels; deferred.           |
| `BGEm3LR`           | ✅       | `.fit(load_train())` on user text. LR learns a 2-class head.          |

## Required additional metric: OOS AUROC

Standard accuracy is misleading because the test set is 82% in-scope. The methodology section commits to also reporting:

- **OOS detection AUROC** — treating `P(gold == "No")` as the score (higher = more likely OOS) and `is_oos` as the ground-truth flag. This is the right calibration metric for "none of these" probability behavior.

Not computed by the generic runner today; follow-up when the first CLINC150-OOS sweep is scheduled.

## Reword stems (stability test)

Three paraphrased stems in `loader.py :: REWORD_STEMS`.
