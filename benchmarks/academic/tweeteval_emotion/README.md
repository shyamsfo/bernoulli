# TweetEval-emotion

4-way emotion classification over tweets. Part of the TweetEval benchmark suite from Cardiff NLP. Framed as a 4-option `ChoiceQuestion` (`"anger"`, `"joy"`, `"optimism"`, `"sadness"`).

- **HF path**: [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) · config `emotion`
- **Eval split**: `test` (1,421 examples).
- **Train split**: `train` (3,257 examples) — consumed by `BGEm3LR.fit` and other trainable baselines.

## Loader

```python
from benchmarks.academic.tweeteval_emotion import load, load_train, QUESTION, REWORD_STEMS

for ex in load(limit=100):
    ...  # BenchmarkExample(state_text, question, gold='anger'|'joy'|'optimism'|'sadness', source='tweeteval_emotion')
```

## Baselines applicable here

| Baseline            | Applies? | Note                                                                 |
|---------------------|----------|----------------------------------------------------------------------|
| `BernoulliHTTP`     | ✅       | Four options, letter-labeled (A-D), logit-read.                       |
| `Generative`        | ✅       | Same backbone, text completion parsed.                                |
| `DeBERTaZeroshot`   | ✅       | Choice over the four emotion strings.                                 |
| `BGEm3LR`           | ✅       | `.fit(load_train())` once; the strongest encoder baseline likely.     |

Fine-tuned ceiling candidate: `cardiffnlp/twitter-roberta-base-emotion` — a strong public RoBERTa fine-tune on this exact task. Wire-in deferred until the first full-suite run is on paper.

## Reword stems (stability test)

Three paraphrased stems in `loader.py :: REWORD_STEMS`. Same intent, same answer space.

## Notes

- Not present in the pre-M8 `evals/reports/`. First comparability number lands when the first full M8 benchmark run is scheduled.
- Tweets are short (median ~15 tokens); the state_text is used as-is after a strip, no additional preprocessing.
