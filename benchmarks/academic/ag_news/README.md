# AG News

4-way topic classification over short news snippets. Framed as a 4-option `ChoiceQuestion` (`"World"`, `"Sports"`, `"Business"`, `"Sci/Tech"`).

- **HF path**: [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news)
- **Eval split**: `test` (7,600 examples).
- **Train split**: `train` (120,000 examples) — consumed by `BGEm3LR.fit` and other trainable baselines.

## Loader

```python
from benchmarks.academic.ag_news import load, load_train, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question, gold='World'|'Sports'|'Business'|'Sci/Tech', source='ag_news')
```

## Baselines applicable here

| Baseline            | Applies? | Note                                                                   |
|---------------------|----------|------------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | Four options, letter-labeled (A-D), logit-read. One forward pass.       |
| `Generative`        | ✅       | Same backbone, text completion parsed. Isolates logit vs. generation.   |
| `DeBERTaZeroshot`   | ✅       | Choice over the four option strings. Default hypothesis template.       |
| `BGEm3LR`           | ✅       | `.fit(load_train())` once, then predict per example.                    |

Fine-tuned ceiling candidate: `textattack/bert-base-uncased-ag-news` or similar. Wire-in deferred until the first full-suite run is on paper.

## Reword stems (stability test)

Hand-authored paraphrases in `loader.py :: REWORD_STEMS`. The stability runner (M8 task 11) rebuilds the `ChoiceQuestion` with each stem and compares top-1 answers against the reference run.

## Historical pre-migration run

The pre-M8 report lives at [`evals/reports/ag_news.md`](../../../evals/reports/ag_news.md). Headline accuracy at the time: **0.8479** (zero-shot Qwen2.5-VL-7B, reverse debias). ECE (15-bin) 0.0995, NLL 0.6065. The post-migration results in `results/` here will supersede these and add the DeBERTa / BGE-m3+LR columns + stability + coverage.
