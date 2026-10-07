# Benchmark primer

A walkthrough of Bernoulli's benchmark story, end to end: what we measure, against what, why we run everything ourselves, and how the results will read once they land. This is the orientation doc. For the full methodology see [`benchmarks/README.md`](../../benchmarks/README.md); for the plan and exit criteria see milestones M8–M10 in [`product/milestones.md`](../milestones.md).

## 1. Why Bernoulli benchmarks at all

The product claim has four clauses. Bernoulli produces **calibrated probabilities from one forward pass**, which is **cheaper and more honest than text-generation-plus-parsing**, **competitive with the model classes that actually compete**, and **stable under option-order and prompt-wording perturbations**.

Each clause is a testable statement, and the benchmark suite exists to test them one by one. Calibration is answered by ECE, Brier, and NLL. "Cheaper and more honest than generation" is answered by running the same backbone both ways. "Competitive" is answered by the external baselines. "Stable" is answered by the reorder/reword protocol. If a benchmark or metric doesn't serve one of these questions, it doesn't belong in the suite. That rule is why BoolQ was dropped: it tests reading comprehension, not decisions.

## 2. The six academic benchmarks (M8)

These form the comparability suite. They overlap with the datasets cited in the Jev paper, so our numbers sit next to the category's own.

**SST-2** (`stanfordnlp/sst2`): 2-way sentiment, 872 eval / 67k train. This is the headline benchmark and the expected easy win. It uses the simplest Choice path, with two options labeled A and B. If Bernoulli isn't well calibrated here, nothing else matters.

**AG News** (`fancyzhx/ag_news`): 4-way topic classification over short news snippets, 7.6k eval / 120k train. It exercises the standard multi-option Choice path (letters A–D), which is the shape most production questions take.

**Banking77** (`mteb/banking77`): 77-way customer-support intent, 3,076 eval / 10k train. This is the stress test for chunked scoring. Seventy-seven options exceed the 26-letter alphabet, so `bernoulli/chunked.py` splits them into three chunks of at most 26 letters each, scores each chunk, and softmaxes globally. It also shows how the architecture degrades at scale, which is useful in its own right.

**TweetEval-emotion** (`cardiffnlp/tweet_eval`, config `emotion`): 4-way emotion (anger / joy / optimism / sadness) over tweets, 1,421 eval / 3,257 train. Its job is text style. Every other academic set is formal prose, and noisy social-media text shows whether calibration survives a domain shift.

**PAWS** (`google-research-datasets/paws`, config `labeled_final`): binary paraphrase detection over adversarial sentence pairs, 8k eval / 49k train. It's the first benchmark on `BinaryQuestion`, and the loader stitches `sentence1` and `sentence2` into one state. The pairs are built to fool lexical-overlap shortcuts, so a good score means the model is actually reading.

**arXiv post-cutoff**: custom-built from the arXiv API by `make_dataset.py`. It's a 4-way category classification (cs.CL / cs.CV / math.PR / econ.EM) over papers published strictly after Qwen2.5-VL's training cutoff (default 2025-01-01). It guards against contamination. Every public dataset above could, in principle, be in the backbone's pretraining data. This one can't be, so it answers the "the model memorized the test set" objection for the whole suite.

## 3. The five use-case benchmarks (M9)

Academic sets prove the mechanism. Use-case sets show that the scenarios on the landing page hold up against public data. Each one backs a specific use-case card (see [`USECASES.md`](../../USECASES.md)). For these, the external baselines shift to domain specialists such as Llama Guard 3, ShieldGemma, and WildGuard where applicable.

**WildGuardTest** (`allenai/wildguardmix`, config `wildguardtest`): binary response-harm judgment over (prompt, response) pairs, 1,725 eval. **Guardrails card.**

**ToxicChat** (`lmsys/toxic-chat`, config `toxicchat1123`): binary toxicity over real user queries to LMSYS/Vicuna, 5,082 eval / 5,083 train. It's known for its distribution shift away from Jigsaw/PerspectiveAPI-style training data, so classifiers tuned on those sources underperform here. **Guardrails card.**

**XSTest** (`natolambert/xstest-v2-copy`): an over-refusal probe of 250 handcrafted safe-vs-contrast prompts. It has no training split by design: it checks that a guardrail doesn't flag "how do I kill a Python process" as violent. **Guardrails card.**

**CLINC150-OOS** (`clinc_oos`, config `plus`): in-scope vs out-of-scope binary, 5,500 eval / 15,250 train. It's the first benchmark that measures a "none of these" probability, which is what a triage system needs when a ticket doesn't belong to any known queue. **Support-triage card.**

**Yelp 1–5 stars** (`yelp_review_full`): 5-way star rating, 50k eval / 650k train. It's the first `RatingQuestion` benchmark, and the headline metrics are MAE and off-by-one accuracy rather than exact match. **Content moderation / rating card.**

## 4. The baselines, and why Generative is the important one

Every Bernoulli number is reported against four baselines:

| Baseline | What it is | Question it answers |
|---|---|---|
| **Generative (same model)** | Same backbone and input, prompted for text, completion parsed | Does reading logits beat parsing text? |
| **DeBERTa-zeroshot** | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`, NLI-style zero-shot | Did we need an LLM at all? |
| **BGE-m3 + LR** | `BAAI/bge-m3` embeddings + per-task logistic regression on the train split | Would embeddings plus a classifier have been enough? |
| **Fine-tuned ceiling** | A known-good public fine-tune per task | How far are we from task-specific training? |

The runner (`benchmarks/run.py`) implements the first three alongside Bernoulli itself. The fine-tuned ceiling is deferred per benchmark until a first full-suite run exists. DeBERTa covers Choice questions only, so its column is `n/a` for Binary and Rating benchmarks such as PAWS.

**The key insight is that Generative can't be looked up anywhere.** It isn't a published model with published scores. It's *our* backbone, *our* prompt, and *our* option set, read the other way. Holding the model fixed is what makes it the cleanest experiment in the suite: any ECE or NLL gap between Bernoulli and Generative comes from logit-read versus text-parse and nothing else. As of commit `d34cf97`, Generative runs over the server's `/v1/generate` endpoint, sharing the already-loaded scorer instead of loading a second copy onto the GPU.

## 5. Why we can't just download published numbers

Asking "aren't these published numbers I can just quote?" is fair, and the answer is mostly no, for three reasons.

**Framing drift.** DeBERTa-zeroshot's published SST-2 accuracy uses a specific hypothesis template over `["positive", "negative"]`. Our harness passes the benchmark's option strings as candidate labels with `"This text is about {}."` as the template. The two numbers are related but not identical, and an apples-to-apples comparison needs identical inputs.

**Metrics not reported upstream.** Our methodology commits to reorder/reword stability, coverage curves at 95/90/80/50% autodecision rates, 10-bin ECE (JevBench-compatible), per-call latency, and cost per 1k decisions. Papers rarely report these, and almost never under our exact protocol. The only way to fill those columns is to run locally.

**Reproducibility footer.** Each `results/<date>.md` records the model revision, Bernoulli git SHA, dataset revision, seed, and hardware. A published number has none of this for our configuration, so it would be the one cell in the table we couldn't defend.

There's one caveat. For a few cells, such as DeBERTa's base SST-2 accuracy on standard GLUE validation, a published figure makes a useful sanity check on our harness. For the full column (stability, coverage, latency, 10-bin ECE) there's no shortcut.

## 6. The DeBERTa-on-CPU scheduling problem

On a single-GPU dev box, the Bernoulli server owns the GPU, so DeBERTa runs on CPU. A zero-shot NLI classifier does one forward pass per hypothesis, which means its latency scales linearly with the number of candidate labels:

| Benchmark | Options | DeBERTa CPU latency | Full run incl. stability |
|---|---|---|---|
| SST-2 | 2 | ~7 s/call | tolerable |
| AG News, TweetEval | 4 | ~14 s/call | multiple hours each |
| Banking77 | 77 | ~270 s/call | multiple days |

The first smoke sweep tried this and had to be killed.

The fix is dual-box scheduling, and it doesn't compromise the methodology:

- **CPU baselines are standalone.** DeBERTa and BGE-m3+LR never call the Bernoulli server. Running them on a separate CPU-heavy box (for example a c7i.4xlarge with 16 vCPUs and AVX-512, about $0.71/hr) is scientifically identical, as long as the model revision, dataset revision, and seed are pinned. The runner already enforces that.
- **The footer records the host.** Each baseline run writes its hardware as `platform.node()`, so splitting across boxes is reported openly and isn't a hidden confound.
- **The GPU box is freed up.** Bernoulli-HTTP and Generative-HTTP sweeps run on the GPU box without contention, in parallel with the CPU sweep.
- **Banking77 is still expensive.** A 16-vCPU AVX-512 box is roughly 5× faster than the g5's CPU, which turns "multi-day" into "hours". When that comparison matters, either accept partial coverage via `--limit` or move DeBERTa to a cheap GPU host.
- **It mirrors a real deployment.** Nobody runs DeBERTa and a 7B VLM on the same production node.

## 7. What the published result will look like

The summary leaderboard in [`benchmarks/README.md`](../../benchmarks/README.md) has one row per benchmark. The four measured columns are the core of each row, with ECE and stability alongside:

| Benchmark | Bernoulli | Generative | DeBERTa-zeroshot | BGE-m3 + LR | ECE (10-bin) | Stability (reorder, reword) |
|---|---|---|---|---|---|---|
| SST-2 | acc | acc | acc | acc | per column | (x%, y%) |
| … | | | | | | |

How to read it:

- **Bernoulli wins calibration** if its ECE(10) is the lowest in the row.
- **Bernoulli wins vs. generation** if its ECE/NLL beats same-model Generative at comparable accuracy. That directly validates logit-read over text-parse.
- **Bernoulli wins necessity** if it beats BGE-m3+LR, which shows embeddings plus a classifier weren't enough.
- **Bernoulli wins model-class** if it beats DeBERTa-zeroshot despite being a general-purpose VLM rather than a purpose-built classifier.

A row can win some of these and lose others, and that's an acceptable, publishable outcome. The point is to know which clause of the claim holds where.

**Further reading**

- [`benchmarks/README.md`](../../benchmarks/README.md): full methodology, metric definitions, stability protocol
- Per-benchmark deep dives: [`academic/`](../../benchmarks/academic/), [`guardrails/`](../../benchmarks/guardrails/), [`triage/`](../../benchmarks/triage/), [`ratings/`](../../benchmarks/ratings/). Each has a `README.md` and `results/latest.md`.
- [`vision_and_roadmap.md` §1](../../vision_and_roadmap.md#1-vision): the original calibration claim ("when Bernoulli says 0.9, it is right ~90% of the time")
- [`product/parking-lot.md`](../parking-lot.md): the Generative GPU-contention item, resolved 2026-10-07 by `/v1/generate`

## Where this fits in the roadmap

Benchmarks are not the endpoint. M8 (foundation and academic suite) and M9 (use-case coverage) produce the baseline run that **M5** (v1.0 hardening) depends on for its regression gate. v1.0 isn't tagged until a benchmark baseline is on record. The same harness then feeds **M10**, the JevBench adapter and leaderboard submission, where the comparison is against Jev itself. Read the numbers here as the evidence base those two milestones stand on, not as a finished scorecard.
