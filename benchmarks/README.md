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

Last run: **2026-10-07**, N=100 eval per benchmark, dev-tier backbone Qwen2.5-VL-7B-Instruct on g5.xlarge/A10G 24GB. Per-benchmark result markdowns are linked from the Benchmark column. See [Honest caveats](#honest-caveats) below for what this snapshot is NOT (no DeBERTa column yet, N=100 is small, WildGuardTest pending auth, arXiv-sample is placeholder).

Each cell is `accuracy` unless noted; **bold** = best in row. ECE is 10-bin (JevBench convention). Stability is `(reorder, reword)` for choice; `(—, reword)` for binary (reorder is semantic n/a for Yes/No labels).

### Academic (M8)

| Benchmark | Bernoulli | Generative | BGE-m3 + LR | Bernoulli NLL | Generative NLL | NLL ratio | Bernoulli ECE | Stability |
|---|---|---|---|---|---|---|---|---|
| [SST-2](academic/sst2/) | **0.95** | **0.95** | 0.90 | **0.18** | 1.38 | 7.6× | 0.027 | (1.00, 0.95) |
| [AG News](academic/ag_news/) | **0.79** | 0.72 | 0.57 | **1.05** | 7.76 | 7.4× | 0.158 | (0.95, 0.94) |
| [Banking77](academic/banking77/) | 0.57 | 0.48 | **0.98** | **1.65** | 13.9 | 8.4× | 0.116 | (0.13 ⚠️, 0.59) |
| [TweetEval-emotion](academic/tweeteval_emotion/) | 0.77 | 0.76 | **0.80** | **0.63** | 6.63 | 10.5× | 0.120 | (0.89, 0.89) |
| [PAWS](academic/paws/) | **0.83** | 0.82 | 0.56 | **0.41** | 4.97 | 12× | 0.076 | (—, 0.94) |
| [arXiv post-cutoff](academic/arxiv_post_cutoff/) | — | — | — | — | — | — | — | — |

- Banking77 reorder = 0.13 ⚠️ is architectural, not a bug — chunked scoring (77 options > 26 letters) is sensitive to which labels fall in which chunk. See the parking-lot item on cross-chunk debias.
- arXiv post-cutoff: dataset is 2 placeholder rows pending M8 task 10b (`make_dataset.py` run). Numbers from the placeholder are intentionally omitted.
- BGE-m3+LR is trained on 500 labeled examples per task. On Banking77 (77-way intent) it dominates — a trained encoder with enough class-balanced data beats zero-shot on fixed label spaces. Honest "when to use what" story: BGE+LR for high-label-count tasks with real training data; Bernoulli everywhere else.

### Use cases (M9)

| Benchmark | Bernoulli | Generative | BGE-m3 + LR | Bernoulli NLL | Generative NLL | NLL ratio | Bernoulli ECE | Headline extra |
|---|---|---|---|---|---|---|---|---|
| [WildGuardTest](guardrails/wildguard_test/) | — | — | — | — | — | — | — | pending HF auth |
| [ToxicChat](guardrails/toxicchat/) | 0.55 | 0.56 | **0.67** | **1.07** | 12.2 | 11× | 0.341 | — |
| [XSTest](guardrails/xstest/) | 0.68 | **0.79** | n/a | **0.53** | 5.80 | 11× | 0.111 | Bernoulli **false_refusal_rate = 0.00** |
| [CLINC150-OOS](triage/clinc150_oos/) | 0.46 | 0.37 | **0.77** | **0.95** | 17.4 | **18×** | 0.267 | BGE+LR oos_auroc 0.75 (best) |
| [Yelp 1-5 stars](ratings/yelp_stars/) | **0.58** | 0.55 | 0.48 | **0.91** | 12.4 | 14× | 0.196 | Bernoulli **MAE 0.44** (lowest) |

- WildGuardTest: dataset `allenai/wildguardmix` is gated on HF. Needs one-time license acceptance + `HF_TOKEN` on the dev box. Pending.
- Per-benchmark extras (false-refusal rate, OOS AUROC, MAE / off-by-one) are reported in each benchmark's full `results/<date>.md`.

### Jev category (M10)

| Benchmark | Bernoulli | Jev | Delta |
|-----------|-----------|-----|-------|
| [JevBench](jevbench/) | TBD       | TBD | TBD   |

### Latency + cost (reported once per backbone, not per benchmark)

Measured over HTTP against the running server; see each benchmark's `results/latest.md` for per-task numbers if they diverge.

| Backbone                             | p50 (1q)  | p50 (5q)  | p50 (20q) | $ / 1k decisions |
|--------------------------------------|-----------|-----------|-----------|------------------|
| Qwen2.5-VL-7B (g5.xlarge A10G, vLLM) | 76 ms     | 368 ms    | 1466 ms   | TBD              |
| Qwen2.5-VL-32B-AWQ (M4e, parked)     | TBD       | TBD       | TBD       | TBD              |

(Latency numbers above lifted from [`evals/reports/loadtest.md`](../evals/reports/loadtest.md).)

## Honest caveats

What this snapshot is **not**:

- **N=100 eval examples per benchmark.** Headline numbers in a 5-point band could swing 2–3 points at N=500. Direction holds; precision doesn't. A higher-N pass is pending (hours of GPU, not days).
- **No DeBERTa-v3-zeroshot column.** The strongest open zero-shot encoder is missing — CPU latency on 4-way+ is pathological (hours per benchmark), and GPU is held by the Bernoulli server. Fix is either shared-GPU scheduling or an overnight CPU batch. The calibration gap between Bernoulli and Generative is unaffected, but it's the biggest hole in the "did we need an LLM?" story.
- **Banking77 reorder stability = 0.13.** With 77 options we exceed the 26-letter alphabet, so `bernoulli/chunked.py` splits into 3 chunks of ≤26 labels and softmaxes globally. The top-1 therefore shifts when the chunks change — architectural, not a bug. Cross-chunk debias is parked. Any Banking77 citation needs the chunked-stability footnote.
- **XSTest Bernoulli `unsafe_recall = 0.36`.** The 0% false-refusal is a product-grade property, but it under-flags actually-unsafe prompts. For a complete guardrail, Bernoulli is the first-pass "don't over-refuse" gate paired with an escalation path for the long tail.
- **CLINC150-OOS Bernoulli `oos_recall ≈ 0.52`.** Coin flip on the "none of these" story. Likely prompt-framing + stem rewording; investigation pending.
- **Dev-tier backbone.** All numbers are from Qwen2.5-VL-7B-Instruct on an AWS g5.xlarge (A10G 24GB). The production step-up to `Qwen2.5-VL-32B-Instruct-AWQ` on g6e.xlarge (L40S 48GB) is parked on AWS capacity.
- **WildGuardTest / arXiv post-cutoff** absent from the headline rows pending HF auth and real-data generation respectively (see table footnotes).

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
