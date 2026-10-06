# SST-2

Binary sentiment classification over one-sentence excerpts from the Stanford Sentiment Treebank. Framed as a 2-option `ChoiceQuestion` (`"negative"`, `"positive"`) so Bernoulli answers with a label letter.

- **HF path**: [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2)
- **Eval split**: `validation` (872 examples — GLUE's test labels are not public).
- **Train split**: `train` (67,349 examples) — consumed by `BGEm3LR.fit` and any other trainable baseline. Not used by Bernoulli (zero-shot).

## Loader

```python
from benchmarks.academic.sst2 import load, load_train, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question, gold='negative'|'positive', source='sst2', meta={example_id})
```

`load()` yields the eval stream; `load_train()` yields the training stream. Both honor the optional `limit` kwarg for smoke runs.

## Baselines applicable here

| Baseline            | Applies? | Note                                                                 |
|---------------------|----------|----------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | The whole point. Two options, letter-labeled, logit-read.             |
| `Generative`        | ✅       | Same backbone, text completion parsed. Isolates logit vs. generation. |
| `DeBERTaZeroshot`   | ✅       | Choice over the two option strings. Default hypothesis template.      |
| `BGEm3LR`           | ✅       | `.fit(load_train())` once, then predict per example.                  |

No per-benchmark fine-tuned ceiling here yet — strong public SST-2 fine-tunes exist (`distilbert-base-uncased-finetuned-sst-2-english` is the obvious pick), but wiring it in requires a per-benchmark adapter class. Deferred until the first full-suite run is on paper.

## Reword stems (stability test)

Hand-authored paraphrases in `loader.py :: REWORD_STEMS`. Three variants, same intent, same answer space. The stability runner (M8 task 11) rebuilds the `ChoiceQuestion` with each stem and compares top-1 answers against the reference run.

## Historical pre-migration run

The pre-M8 report lives at [`evals/reports/sst2.md`](../../../evals/reports/sst2.md). Numbers at the time (zero-shot Qwen2.5-VL-7B, reverse debias):

| config                      | accuracy | ECE (15-bin) | NLL    |
|-----------------------------|----------|--------------|--------|
| raw (debias=reverse)        | 0.9174   | 0.0280       | 0.2458 |
| calibrated (T=1.35)         | 0.9174   | 0.0253       | 0.2338 |
| generative baseline         | 0.9174   | 0.0826       | 2.2815 |

The post-migration numbers in `results/` here will supersede these and add the DeBERTa / BGE-m3+LR columns + stability + coverage.
