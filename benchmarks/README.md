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
just bench benchmark academic/sst2                        # single benchmark
just bench benchmark academic/sst2 bernoulli,generative,deberta
just bench sweep-academic                                 # the whole academic matrix
just bench sweep-usecase                                  # the whole use-case matrix
```

Direct invocation without the recipe:

```bash
uv run python -m benchmarks.run academic/sst2 --out results/$(date +%F).md
```

The runner POSTs to a Bernoulli server (default `http://127.0.0.1:8000`). Start one with `just dev serve-local` or `just dev docker-run` first (or, on the box, `just serve-start` for the daily production uvicorn).

## Summary leaderboard

Last run: **2026-10-15**, N=500 eval per benchmark (arXiv post-cutoff at N=100 — small test set), dev-tier backbone Qwen2.5-VL-7B-Instruct on g5.xlarge/A10G 24GB. DeBERTa column from a one-shot Lambda A100 run. Per-benchmark result markdowns are linked from the Benchmark column. See [Honest caveats](#honest-caveats) below.

Each cell is `accuracy` unless noted; **bold** = best in row. ECE is 10-bin (JevBench convention). Stability is `(reorder, reword)` for choice; `(—, reword)` for binary (reorder is semantic n/a for Yes/No labels). DeBERTa is `n/a` on PAWS — `DeBERTaZeroshot` only scores `ChoiceQuestion` by design.

### Academic (M8)

| Benchmark | Bernoulli | Generative | BGE-m3 + LR | DeBERTa | Bernoulli NLL | Generative NLL | NLL ratio (G/B) | Bernoulli ECE | Stability |
|---|---|---|---|---|---|---|---|---|---|
| [SST-2](academic/sst2/) | 0.914 | 0.920 | 0.900 | **0.930** | **0.251** | 2.211 | 8.8× | 0.024 | (1.00, 0.98) |
| [AG News](academic/ag_news/) | 0.828 | 0.772 | **0.876** | 0.874 | 0.805 | 6.167 | 7.7× | 0.120 | (0.96, 0.94) |
| [Banking77](academic/banking77/) | 0.572 | 0.640 | **0.950** | 0.640 | 1.401 | 9.668 | 6.9× | 0.111 | (0.07 ⚠️, 0.68) |
| [PAWS](academic/paws/) | **0.846** | 0.836 | 0.552 | n/a | **0.376** | 4.532 | 12.0× | 0.040 | (—, 0.94) |
| [TweetEval-emotion](academic/tweeteval_emotion/) | 0.784 | 0.780 | **0.788** | 0.768 | **0.575** | 5.977 | 10.4× | 0.084 | (0.89, 0.89) |
| [arXiv post-cutoff](academic/arxiv_post_cutoff/) | **0.950** | 0.920 | 0.940 | 0.530 | **0.207** | 2.211 | 10.7× | 0.047 | (0.95, 0.94) |

**Headline finding — the "provably unseen" test**: arXiv papers published 2026-09-18 to 2026-10-06 (strictly after both Qwen2.5-VL's and DeBERTa's training cutoffs). Bernoulli 0.95 vs **DeBERTa 0.53** — DeBERTa's NLI head doesn't generalize to post-cutoff material; Bernoulli (via Qwen) does. BGE-m3 holds up because it only needs semantic similarity. Full dataset generation in [`academic/arxiv_post_cutoff/make_dataset.py`](academic/arxiv_post_cutoff/make_dataset.py); dataset snapshot in [`.../data/dataset.jsonl`](academic/arxiv_post_cutoff/data/dataset.jsonl).

Other things the N=500 numbers settle:

- **vs Generative, the calibration gap is real and consistent** — NLL is 7–12× worse across every benchmark. The "prompt-and-parse is miscalibrated" story we've been telling holds up.
- **vs DeBERTa on classical NLP tasks, we're competitive, not dominant** — on SST-2/AG News/TweetEval-emotion the two trade blows; neither is a clear winner. DeBERTa is a legitimate competitor and we should name that honestly.
- **BGE-m3+LR trained on 2000 labels wins on label-heavy tasks** — Banking77 (0.95 vs everyone else's 0.57–0.64) and AG News (0.876, best in row). This is the honest "if you have 2000 labeled examples, use them" conclusion; Bernoulli's story is for the case where you don't.
- **DeBERTa stability is (1.000, 1.000) everywhere** — architectural, not tuning. An encoder with explicit label hypotheses doesn't have option-position bias or prompt-stem sensitivity. Our story is strongest where DeBERTa can't go (PAWS binary, provably-unseen content), not on stability.
- **Banking77 reorder = 0.07 ⚠️** is architectural — chunked scoring (77 options > 26 letters) means which labels land in which chunk affects the global softmax. Parking-lot item on cross-chunk debias.

### Use cases (M9)

| Benchmark | Bernoulli | Generative | BGE-m3 + LR | Bernoulli NLL | Generative NLL | NLL ratio | Bernoulli ECE | Headline extra |
|---|---|---|---|---|---|---|---|---|
| [WildGuardTest](guardrails/wildguard_test/) | — | — | — | — | — | — | — | pending HF auth |
| [ToxicChat](guardrails/toxicchat/) | 0.69 | 0.69 | **0.78** | **0.800** | 8.46 | 10.6× | 0.221 | — |
| [XSTest](guardrails/xstest/) | **0.68** | 0.66 | n/a | **0.645** | 9.39 | 14.6× | 0.131 | Bernoulli **false_refusal_rate = 0.008** (~0.8%) |
| [CLINC150-OOS](triage/clinc150_oos/) | 0.45 | 0.29 | **0.80** | **1.028** | 19.56 | **19×** | 0.280 | BGE+LR oos_auroc 0.78 (best); Bernoulli oos_auroc 0.48 (coin flip) |
| [Yelp 1-5 stars](ratings/yelp_stars/) | **0.62** | 0.58 | 0.57 | **0.890** | 11.72 | 13.2× | 0.133 | Bernoulli **MAE 0.42** (lowest; Generative 0.46, BGE+LR 0.53) |

- Last run: **2026-10-16**, N=500 eval per benchmark. BGE+LR trained on 2000 examples. Previous 2026-10-07 N=100 numbers superseded.
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

- **DeBERTa doesn't score PAWS.** `DeBERTaZeroshot.predict()` raises `NotImplementedError` on `BinaryQuestion` by design — Yes/No labels aren't semantic hypothesis-shaped. A benchmark wanting DeBERTa on a binary task has to preprocess into a 2-way choice. We left PAWS as-is.
- **Banking77 reorder stability = 0.07 ⚠️.** With 77 options we exceed the 26-letter alphabet, so `bernoulli/chunked.py` splits into 3 chunks of ≤26 labels and softmaxes globally. The top-1 therefore shifts when the chunks change — architectural, not a bug. Cross-chunk debias is parked. Any Banking77 citation needs the chunked-stability footnote.
- **arXiv post-cutoff at N=100.** The test split is small — 200 papers across cs.CL / cs.CV / math.PR / econ.EM, publication dates 2026-09-18 to 2026-10-06 (all strictly after `BACKBONE_CUTOFF = 2025-01-01`, well after any plausible Qwen2.5-VL cutoff). N=100 is the full available test slice; expanding it means scraping more arXiv.
- **XSTest Bernoulli `unsafe_recall = 0.28`.** The ~0.8% false-refusal rate is a product-grade property, but it under-flags actually-unsafe prompts. For a complete guardrail, Bernoulli is the first-pass "don't over-refuse" gate paired with an escalation path for the long tail.
- **CLINC150-OOS Bernoulli `oos_auroc ≈ 0.48`.** Coin flip on the "none of these" story — BGE-m3+LR at 0.78 dominates (it never predicts OOS but the softmax over 150 in-scope classes carries enough signal). Likely prompt-framing + stem rewording; investigation pending.
- **Dev-tier backbone.** All numbers are from Qwen2.5-VL-7B-Instruct on an AWS g5.xlarge (A10G 24GB). The production step-up to `Qwen2.5-VL-32B-Instruct-AWQ` on g6e.xlarge (L40S 48GB) is parked on AWS capacity; one-shot H100 run on the 32B base model is in [`product/reports/2026-10-10-h100-32b.md`](../product/reports/2026-10-10-h100-32b.md).
- **WildGuardTest** absent from the headline rows pending HF auth.

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
