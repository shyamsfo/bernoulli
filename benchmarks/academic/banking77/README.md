# Banking77

77-way intent classification over customer-support utterances. Framed as a 77-option `ChoiceQuestion`; exceeds the 26-letter alphabet so `bernoulli/chunked.py` handles scoring via chunked (3-pass) aggregation.

- **HF path**: [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) (parquet reupload of the original PolyAI dataset; the PolyAI dataset-script format no longer loads under `datasets>=3`).
- **Eval split**: `test` (3,076 examples).
- **Train split**: `train` (10,003 examples) — consumed by `BGEm3LR.fit` and other trainable baselines.

## Loader

```python
from benchmarks.academic.banking77 import load, load_train, question, REWORD_STEMS

q = question()  # ChoiceQuestion with 77 options, built by walking the test split
for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question, gold='refund_not_showing_up' | ..., source='banking77')
```

### Shape deviation from other academic benchmarks

Unlike `sst2` and `ag_news`, Banking77 does not expose a module-level `QUESTION` constant. The 77-way option list is derived from the dataset's `label_text` column rather than hardcoded (the labels are long and brittle to type by hand). `question()` is a function that builds the `ChoiceQuestion` on demand. The runner, DeBERTa adapter, and stability runner call it when they need the question outside of a `load()` loop.

## Baselines applicable here

| Baseline            | Applies? | Note                                                                        |
|---------------------|----------|-----------------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | 77 options → `chunked.py` splits into 3 chunks of ≤26 letters, softmaxed globally. |
| `Generative`        | ✅       | Same backbone, text completion parsed. Expect poor parse-match on 77-way.    |
| `DeBERTaZeroshot`   | ✅       | Choice over 77 option strings. Slow (one forward pass per label).            |
| `BGEm3LR`           | ✅       | `.fit(load_train())` once, then predict per example. Likely the strongest encoder baseline on this task. |

Fine-tuned ceiling candidate: `bert-base-uncased-banking77` or the MTEB leaderboard winner. Wire-in deferred.

## Reword stems (stability test)

Hand-authored paraphrases in `loader.py :: REWORD_STEMS`. The stability runner rebuilds the `ChoiceQuestion` with each stem (reusing `question().options`) and compares top-1 answers against the reference run.

## Historical pre-migration run

The pre-M8 report lives at [`evals/reports/banking77.md`](../../../evals/reports/banking77.md). Headline accuracy at the time: **0.5822** (zero-shot Qwen2.5-VL-7B, chunked scoring, no debias within chunks). ECE (15-bin) 0.1272, NLL 1.6178. The post-migration results in `results/` here will supersede these and add the DeBERTa / BGE-m3+LR columns + stability + coverage.
