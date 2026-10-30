# Yelp 1-5 stars

Rating classification over Yelp review paragraphs. First benchmark in the suite that uses `RatingQuestion` — exercises the integer-string response-key convention (`"1"` .. `"5"`) and the expected-value aggregation machinery.

- **HF path**: [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full)
- **Eval split**: `test` (50,000 examples).
- **Train split**: `train` (650,000 examples) — consumed by `BGEm3LR.fit`. Expect the first full run to use `--train-limit` (fitting the LR on half a million embeddings is overkill for the signal).

## Framing

5-star rating as a `RatingQuestion(scale=(1, 5))`. The gold is `"1"` .. `"5"` as a string, matching the response-distribution key convention in `bernoulli/labels.py`. Bernoulli reports both the full probability distribution over the five rating values and an expected-value scalar (`Σ p_i × i`); the methodology below privileges the discrete-accuracy headline for comparability with the generative baseline and the fine-tuned ceiling.

## Loader

```python
from benchmarks.ratings.yelp_stars import load, load_train, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question=RatingQuestion(scale=(1, 5)), gold='1'..'5')
```

## Baselines applicable here

| Baseline            | Applies? | Note                                                                 |
|---------------------|----------|----------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | Rating — logits read at the answer position restricted to A-E labels (letters A..E map to stars 1..5 internally; see `bernoulli/labels.py`). |
| `Generative`        | ✅       | Same backbone, text completion parsed for an integer.                 |
| `DeBERTaZeroshot`   | ❌       | Zero-shot pipeline over integer labels degrades to coin flips; the baseline's own `predict` raises `NotImplementedError` on `RatingQuestion`. |
| `BGEm3LR`           | ✅       | `.fit(load_train(limit=N))` on review text. LR treats the five ratings as nominal classes — ordinal semantics are lost but the output is still a usable distribution. |

Fine-tuned ceiling candidate: `textattack/bert-base-uncased-yelp-polarity` is only 2-class; a 5-class public fine-tune may need to be trained. Deferred.

## Required additional metrics: MAE + off-by-one accuracy

Standard accuracy under-represents what matters for ratings — predicting 4 stars when the gold is 5 is better than predicting 1. The methodology section commits to also reporting:

- **MAE** = mean |expected_stars - gold_int|. Lower is better.
- **Off-by-one accuracy** = fraction of examples where `|argmax(dist) - gold_int| ≤ 1`.

Not computed by the generic runner today; follow-up when the first Yelp sweep is scheduled.

## Reword stems (stability test)

Three paraphrased stems in `loader.py :: REWORD_STEMS`. Reorder stability is `None` for ratings (the integer order is semantic) — the stability runner skips reorder for `RatingQuestion` by design.
