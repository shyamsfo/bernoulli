# Benchmarks

External comparison of Bernoulli against competing baselines and the Jev category. Everything here exists to produce **publishable, reproducible numbers** — how we rank against the model classes that actually compete, and whether our calibration + stability claims hold up under outside scrutiny.

For *internal* correctness checks (unit tests, GPU-marked scorer tests, latency profiling for ops) see `tests/` and `evals/`. This folder is strictly outward-facing.

Plan + exit criteria: [`product/milestones.md`](../product/milestones.md), milestones **M8** (foundation + academic suite), **M9** (use-case coverage), **M10** (JevBench adapter + leaderboard submission).

## Layout

```
benchmarks/
├── README.md                       # this file — summary leaderboard + methodology
├── common/                         # shared harness (M8 task 2–4)
│   ├── dataset.py                  # BenchmarkExample loader interface
│   ├── metrics.py                  # accuracy / macro-F1 / ECE / Brier / NLL / coverage / stability
│   └── baselines.py                # Bernoulli-via-HTTP + generative + DeBERTa + BGE-m3 + ceiling
├── academic/                       # M8 — comparability suite from the Jev paper
│   ├── sst2/
│   ├── ag_news/
│   ├── banking77/
│   ├── tweeteval_emotion/
│   ├── paws/
│   └── arxiv_post_cutoff/
├── guardrails/                     # M9
│   ├── wildguard_test/
│   ├── toxicchat/
│   └── xstest/
├── triage/                         # M9
│   └── clinc150_oos/
├── ratings/                        # M9
│   └── yelp_stars/
└── jevbench/                       # M10 — upstream adapter + leaderboard submission
```

Each benchmark subfolder owns its own `README.md`, loader, and `results/latest.md`. Historical runs live beside `latest.md` as `results/YYYY-MM-DD.md`.

## How to run

A single benchmark (lands in M8 task 4 alongside `benchmarks/common/`):

```bash
just benchmark academic/sst2                        # single benchmark
just benchmark academic/sst2 --baselines generative,deberta
just benchmark-all                                  # everything (slow)
```

Direct invocation without the recipe:

```bash
uv run python -m benchmarks.run academic/sst2 --out results/$(date +%F).md
```

The runner POSTs to a Bernoulli server (default `http://127.0.0.1:8000`). Start one with `just serve` or `just docker-run` first.

## Summary leaderboard

Updated by each benchmark run. Empty rows will fill in as M8 → M9 → M10 progress.

Headline metric per benchmark is **accuracy** unless noted. ECE is 10-bin (JevBench convention). Stability is `(reorder, reword)`.

### Academic (M8)

| Benchmark             | Bernoulli | Generative (same model) | DeBERTa-zeroshot | BGE-m3 + LR | Fine-tuned ceiling | ECE (10-bin) | Stability |
|-----------------------|-----------|--------------------------|------------------|-------------|--------------------|--------------|-----------|
| [SST-2](academic/sst2/) | TBD       | TBD                      | TBD              | TBD         | TBD                | TBD          | TBD       |
| [AG News](academic/ag_news/) | TBD       | TBD                      | TBD              | TBD         | TBD                | TBD          | TBD       |
| [Banking77](academic/banking77/) | TBD       | TBD                      | TBD              | TBD         | TBD                | TBD          | TBD       |
| [TweetEval-emotion](academic/tweeteval_emotion/) | TBD       | TBD                      | TBD              | TBD         | TBD                | TBD          | TBD       |
| [PAWS](academic/paws/) | TBD       | TBD                      | n/a (binary)     | TBD         | TBD                | TBD          | TBD       |
| [arXiv post-cutoff](academic/arxiv_post_cutoff/) | TBD       | TBD                      | TBD              | TBD         | TBD                | TBD          | TBD       |

### Use cases (M9)

| Benchmark           | Headline metric      | Bernoulli | External baseline(s)              | ECE (10-bin) | Stability |
|---------------------|----------------------|-----------|-----------------------------------|--------------|-----------|
| [WildGuardTest](guardrails/wildguard_test/) | accuracy / F1 | TBD | Llama Guard 3 / ShieldGemma / WG | TBD | TBD |
| ToxicChat           | accuracy / F1        | TBD       | Llama Guard 3 / ShieldGemma       | TBD          | TBD       |
| XSTest              | accuracy + refusal % | TBD       | Llama Guard 3 / ShieldGemma       | TBD          | TBD       |
| CLINC150 (with OOS) | accuracy / OOS AUROC | TBD       | DeBERTa-zeroshot                  | TBD          | TBD       |
| Yelp 1-5 stars      | MAE / off-by-one acc | TBD       | DeBERTa-zeroshot                  | TBD          | TBD       |

### Jev category (M10)

| Benchmark | Bernoulli | Jev | Delta |
|-----------|-----------|-----|-------|
| JevBench  | TBD       | TBD | TBD   |

### Latency + cost (reported once per backbone, not per benchmark)

Measured over HTTP against the running server; see each benchmark's `results/latest.md` for per-task numbers if they diverge.

| Backbone                             | p50 (1q)  | p50 (5q)  | p50 (20q) | $ / 1k decisions |
|--------------------------------------|-----------|-----------|-----------|------------------|
| Qwen2.5-VL-7B (g5.xlarge A10G, vLLM) | 76 ms     | 368 ms    | 1466 ms   | TBD              |
| Qwen2.5-VL-32B-AWQ (M4e, parked)     | TBD       | TBD       | TBD       | TBD              |

(Latency numbers above lifted from [`evals/reports/loadtest.md`](../evals/reports/loadtest.md).)

## Methodology

### Metrics

- **Accuracy** — top-1 label match against gold.
- **Macro-F1** — unweighted mean of per-class F1. Matters on imbalanced sets (Banking77, CLINC150, Yelp).
- **ECE** — Expected Calibration Error. Reported in **both 10 bins (JevBench convention, headline)** and 15 bins (our pre-M8 convention, kept for continuity with historical [`evals/reports/`](../evals/reports/) numbers).
- **Brier** — mean squared error of the top-class probability vs the 0/1 gold indicator.
- **NLL** — negative log-likelihood of the gold label under the predicted distribution.
- **Coverage curves** — accuracy at 95% / 90% / 80% / 50% autodecision rates. Rank examples by top-class confidence; keep the top X%; report accuracy on what's kept. This is the curve behind the landing-page slider.
- **Latency** — p50 / p95 wall-clock per `/v1/decide` request at 1, 5, and 20 questions per request. HTTP, not direct scorer.
- **Cost per 1k decisions** — on-demand instance $/hr × (p50 × 1000) / 3600. Not a serving-at-steady-state number — it's a reproducible per-call ceiling.
- **Stability** — fraction of examples whose top-class answer is unchanged under:
  - **Reorder** — same options, randomized order. Reuses the `reverse` / `cyclic` machinery in `bernoulli/debias.py`.
  - **Reword** — up to 3 paraphrased question stems per benchmark, authored by hand per benchmark subfolder.

Reported as a single `(reorder%, reword%)` pair.

### Baselines

Every academic benchmark reports Bernoulli against (where applicable):

1. **Same-model generative** — same backbone, same input, parse the completion string. Isolates the ECE / NLL gap that the logit path closes without confounding model choice.
2. **DeBERTa-v3-zeroshot** — [`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0). The strongest open zero-shot encoder. Fast, cheap, and widely cited — the "did we need an LLM at all?" check.
3. **BGE-m3 + logistic regression** — embed with [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3), train a per-task LR head on the labeled train split. Honest baseline for "could we have used embeddings + a classifier?"
4. **Fine-tuned encoder (ceiling)** — a known-good public fine-tune on the task where one exists (per-benchmark; see each subfolder's `README.md`). Skipped where it would require training a new encoder from scratch.

Use-case benchmarks (M9) swap in domain-specific baselines (Llama Guard, ShieldGemma, WildGuard for guardrails) instead of DeBERTa-zeroshot. See each benchmark's `README.md`.

### Stability protocol

For each benchmark:

1. Record the Bernoulli top-class answer under the canonical option order and canonical question stem — call this `ref`.
2. **Reorder test:** re-run with option order permuted. For `choice` with N ≤ 6 options, use all N! permutations and take the modal answer. For N > 6 use 20 random permutations. The example is **stable under reorder** iff the modal answer equals `ref`.
3. **Reword test:** re-run with each of the up-to-3 paraphrased stems (same intent, same answer space). The example is **stable under reword** iff all paraphrased answers equal `ref`.
4. Report `stability_reorder = fraction stable under reorder` and `stability_reword = fraction stable under reword`.

The reword set is hand-authored per benchmark and committed next to the loader so results are reproducible. If reword is skipped for a benchmark, note it in that benchmark's `results/latest.md`.

### Reproducibility

Each benchmark's `results/<date>.md` must record:

- **Model id + revision** — pinned commit SHA from HF (not a tag).
- **vLLM / transformers version** — from `uv.lock`.
- **Bernoulli git commit SHA** — the exact commit that produced the numbers.
- **Dataset identifier** — HF dataset path + revision, or local snapshot hash.
- **Random seed** — for any sampling (reword draws, split subsampling, etc.).
- **Hardware** — AWS instance type + GPU.

A result missing any of these six is not reproducible and should not be cited externally.

### What is deliberately not here

- **BoolQ** — dropped. It's a reading-comprehension task, not a decision task, and the 63% zero-shot number hurts the story without testing what the product claims. Historical [`evals/reports/boolq.md`](../evals/reports/boolq.md) stays in place as a snapshot.
- **Any benchmark that requires training Bernoulli first** — the whole point is zero-shot calibration. The LoRA path lives in M7 and is reported separately there.
