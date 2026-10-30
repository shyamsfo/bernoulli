# PAWS

Paraphrase Adversaries from Word Scrambling. Binary paraphrase-detection over sentence pairs that have high lexical overlap but may or may not share meaning — a hard adversarial set designed to break bag-of-words shortcuts.

- **HF path**: [`google-research-datasets/paws`](https://huggingface.co/datasets/google-research-datasets/paws) · config `labeled_final` (clean hand-labeled split).
- **Eval split**: `test` (8,000 examples).
- **Train split**: `train` (49,401 examples) — consumed by `BGEm3LR.fit` and other trainable baselines.

## State encoding

PAWS is a sentence-pair task but our `State` type carries a single text block. The loader stitches both sentences into one `state_text`:

```
Sentence 1: <sentence1>
Sentence 2: <sentence2>
```

## Loader

```python
from benchmarks.academic.paws import load, load_train, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question=BinaryQuestion, gold='Yes'|'No', source='paws')
```

First benchmark in the academic suite with a `BinaryQuestion`. Gold values are `"Yes"` / `"No"` to match the binary response-distribution keys in `bernoulli/labels.py`.

## Baselines applicable here

| Baseline            | Applies? | Note                                                                 |
|---------------------|----------|----------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | Binary — Yes/No single-token logit read.                              |
| `Generative`        | ✅       | Same backbone, text completion parsed.                                |
| `DeBERTaZeroshot`   | ❌       | DeBERTa-zeroshot's "candidate labels" aren't semantic for Yes/No; needs hand-authored natural-language labels (`"paraphrase"` / `"not a paraphrase"`). Deferred — the baseline will be skipped on this column until that per-benchmark adapter lands. |
| `BGEm3LR`           | ✅       | `.fit(load_train())` on the sentence-pair state_text. LR learns a 2-class head.  |

Fine-tuned ceiling candidate: `google-research/paws`-based BERT fine-tunes exist in the literature; HF has `textattack/bert-base-uncased-PAWS`.

## Reword stems (stability test)

Three paraphrased stems in `loader.py :: REWORD_STEMS`. The stability runner will swap in each stem and compare top-1 answers against the reference run.

## Notes

- Not in the pre-M8 `evals/reports/`. First comparability number lands when the first M8 benchmark run completes.
- PAWS is adversarial: both the generative baseline and lexical-overlap heuristics historically struggle here. The gap between calibrated logit-read and generative-parse is likely to be sharper than on SST-2.
