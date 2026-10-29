# Milestones

All milestones live in this file. Each section is one milestone with its goal, status, task checklist, and exit criteria. There are no separate PRD/PLAN files in lite mode — this file is the plan.

Source brief: `vision_and_roadmap.md` at the repo root. Treat that document as the authoritative spec for scope, API shape, architecture, and risks.

## Status Key

```
⏳ pending      — not started
🔄 in progress  — actively being worked
✅ done         — exit criteria met
🚫 blocked      — waiting on something external
```

Task checkboxes:
- `[ ]` not done
- `[x]` done
- `[~]` in progress
- `[-]` skipped / won't do

The active milestone is the first one marked `🔄 in progress`. `/ds-work-continue` and `/ds-work-status` find the next `[~]` or `[ ]` task within it.

**Backbone flexibility (project-wide):** the backbone LLM/VLM is **not pinned**. The vision doc proposes Qwen3.6 family but we may choose a different open-weights model (another Qwen variant, a Llama/Gemma/InternVL VLM, or a smaller dev model) based on availability, licensing, calibration behavior, or hardware. Every milestone that touches the model must preserve `Scorer` as a swappable interface — never hard-code a model id outside `config.py`.

**Dependency ordering vs the written sequence.** Milestones are numbered by the order they were added, not by execution order. The real dependency chain today is: finish **M4** (M4e step-up is parked on capacity) → **M8 → M9 → M10** (benchmarks: foundation, use-case coverage, JevBench) → **M5** (v1.0 hardening, which depends on benchmark baselines being in place for the regression gate) → **M11** (public verification endpoint — depends on M5 hardening since this one takes outside traffic) → **M6** (multimodal) → **M7** (LoRA, optional). Do not tag v1.0 before the benchmark suite exists and has a baseline run on record.

---

## M1 — Scaffold (Phase 0)
**Status**: ✅ done
**Goal**: Repo tooling, config system, and pydantic types for the full API are in place; `pytest` + typecheck green on bernoulli.

- [x] Pick dev-tier backbone: **`Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`** (public, non-gated; fits L4 24GB in bf16). Decision + rationale in `CLAUDE.md`.
- [x] `deploy/` — Terraform for AWS g6.xlarge dev box (`bernoulli`), NVIDIA GPU Base CUDA AMI, `id_nuwire` keypair, encrypted gp3 root, cloud-init installs uv + HF cache on `/opt/dlami/nvme`
- [x] `pyproject.toml` with `uv`, `ruff`, `mypy`, `pytest`; phased optional-dependencies (scorer-hf / eval / serve / train)
- [x] Package skeleton: `bernoulli/{__init__,types,prompt,labels,scorer,debias,calibrate,server,config}.py` — real content in types/config/scorer-interface; docstring-only stubs for the rest
- [x] Pydantic types in `types.py` for the full §3 API (State, discriminated-union Question + Decision, request/response)
- [x] `config.py` reads model id / dtype / device / max_len from env via `pydantic-settings` (`BERNOULLI_*`)
- [x] `CLAUDE.md` seeded with commands, config, decisions log
- [x] CI: `Makefile` with `ci / test / lint / fmt` targets — **runs on bernoulli, not GHA**. Rationale: the interesting tests need a GPU; GHA free runners don't have one, so GHA would gate on nothing meaningful. Clean-env guarantee comes from `terraform destroy && apply` (box is cattle, reproducible from `user_data.sh`).
- [x] Pulled Qwen2.5-VL-7B @ pinned revision to `/opt/dlami/nvme/hf-cache` — 16GB, 1.5 min via hf_transfer
- [x] First `just ci` run green on bernoulli (20 tests, ruff clean, mypy strict clean, 0.28s)
- [-] Dockerfile stub — deferred to M4 when VLLMScorer + FastAPI need it (no value shipping an empty Dockerfile now)

**Exit criteria**: `just ci` green on bernoulli (ruff + mypy + pytest); model weights cached on NVMe; dev box reproducible from `terraform destroy && apply`.

**M1 gate status: READY** — all tasks done or explicitly deferred. The reproducibility clause is proven in parts (NVMe LVM bug caught and fixed in `user_data.sh`; next `terraform destroy && apply` is the full proof — added to parking-lot as a periodic check).

---

## M2 — Zero-shot scorer (Phase 1)
**Status**: ✅ done
**Goal**: Text-only `HFScorer` returning calibrated-shape (uncalibrated) distributions for `choice`, `binary`, `rating` on the dev backbone.

- [x] Prompt builder: chat template, system message, state-first/question-last ordering (`bernoulli/prompt.py`)
- [x] Label-token check — letters A–Z + Yes/No are single-token on Qwen2Tokenizer. **Digits ` 1`…` 9` are TWO tokens** (space + digit) — deviation from vision doc §4. Resolved by using letters A–I for ratings uniformly (A = scale low, B = low+1, ...). Logged in CLAUDE.md.
- [x] `HFScorer` in `bernoulli/scorer.py`: `Qwen2_5_VLForConditionalGeneration`, bf16/cuda, one forward pass, last-position logits, restrict to label tokens, cpu+float32 numpy out
- [x] Choice (2–26 letter-labeled options), binary (`Yes`/`No` single-token), rating (letters A–I + integer-string response keys + `expected = Σ p_i · value_i`)
- [x] Unit + GPU tests — 37 total, all green on bernoulli (CPU-only path 3.6s; full suite with model load 78.7s). Includes deterministic-rerun test.
- [x] CLI `bernoulli decide --state file.txt --question q.json` — verified end-to-end on a tracking/urgent/anger example
- [x] Thinking mode: **N/A for Qwen2.5-VL** (no hybrid reasoning; verified top-logit at answer position is ` A`, not `<think>`). Note deferred to M4 if we move to Qwen3-VL for production.

**Exit criteria**: CLI works end-to-end on the dev backbone across all three question types; tests green. ✓

**M2 gate status: READY** — all tasks done. Observed latency 1201 ms for 3 questions sequentially (no batching yet — M4 adds `questions × permutations` batching).

---

## M3 — Debias, calibration, eval harness (Phase 2)
**Status**: ✅ done
**Goal**: `reverse` + `cyclic` debiasing, per-question-type temperature calibration, and a first eval report comparing raw / debiased / calibrated / generative baseline.

- [x] `reverse` debiaser (default); `cyclic` debiaser for k ≤ 6; average in canonical option order (`bernoulli/debias.py`, M3a)
- [x] Eval dataset loaders — SST-2 (`stanfordnlp/sst2`), AG News (`fancyzhx/ag_news`), Banking77 (`mteb/banking77`), BoolQ (`google/boolq`). Common `EvalExample(state, question, gold, source)` format in `evals/example.py` (M3b–e).
- [-] Multimodal eval stubs — deferred to M4 along with the full image path (same reason).
- [x] Generative baseline — `evals/baselines.py`, same backbone + classifier-style system prompt + text parse. P=1 on winner / uniform on parse-fail. (M3f)
- [x] Temperature scaling per question type — scipy `minimize_scalar` bounded on NLL. Persisted to `calibration/qwen2.5-vl-7b.json` with ISO-date version. (M3g phase A + B)
- [x] Metrics — accuracy, macro-F1, ECE (15 bins), Brier, NLL, latency p50/p95. Pure numpy, no scikit-learn. (M3b)
- [x] Report in `evals/reports/` — six markdown + JSON sidecars (sst2, sst2.calibrated, sst2.generative, ag_news, boolq, banking77).
- [x] Chunked/tournament scoring — `bernoulli/chunked.py` splits >26-option choice into chunks of ≤26, keeps raw logits per option, softmaxes globally. Verified on 77-way Banking77. Debias within chunks is deferred (open question in CLAUDE.md). (M3e)

**Exit criteria**: One report in `evals/reports/` comparing all four configs on at least SST-2, AG News, Banking77, BoolQ. ✓

**M3 gate status: READY → done.** Headline comparison on SST-2 (same model, same examples):

| config                      | accuracy | ECE    | NLL    |
|-----------------------------|----------|--------|--------|
| raw (debias=reverse)        | 0.9174   | 0.0280 | 0.2458 |
| **calibrated** (T=1.35)     | 0.9174   | 0.0253 | 0.2338 |
| generative baseline         | 0.9174   | 0.0826 | 2.2815 |

Full dataset coverage: SST-2 91.7% / AG News 84.8% / BoolQ 63.2% / Banking77 58.2%, all zero-shot on Qwen2.5-VL-7B. Generative baseline reports **9× worse NLL** at identical accuracy — the exact calibration gap the logit path closes.

**Deviations from the vision doc** (all logged in CLAUDE.md and/or commits):
- Letters A–I for rating labels, not digits (Qwen tokenizer makes space-prefixed digits two tokens).
- `bernoulli/decide.py` is new (not in §6 layout) — cleanly separates request→response dispatch from scorer and server.
- Multimodal eval stubs deferred to M4 (processor + torchvision land there).
- Dockerfile deferred to M4 (serving container lives there).
- Chunked scoring doesn't apply debias — within-chunk prob averaging vs cross-chunk raw-logit aggregation are mathematically awkward together. Open question.
- Banking77 loader uses `mteb/banking77` (parquet reupload) because `PolyAI/banking77` dataset-script format no longer loads under `datasets>=3`.

---

## M4 — Production serving (text-only)
**Status**: ✅ done (M4e parked, M4g superseded by M8)
**Goal**: A shippable text-only service: `VLLMScorer` with prefix caching + a FastAPI server + the production backbone + Docker + load numbers. Image path explicitly deferred to M6.

- [ ] 🚫 **M4e — parked, but value demonstrated 2026-10-10:** Pick + load the production backbone (step up from the 7B dev model). Target was `Qwen2.5-VL-32B-Instruct-AWQ` on `g6e.xlarge` (L40S 48GB). **External blocker:** `g6e.xlarge` capacity in `us-east-1` has been intermittent across every AZ since M4 began — see decisions log in `CLAUDE.md`. Fell back to `g5.xlarge` + 7B for all of M4a–d/f, which run fine there. **Value demonstrated 2026-10-10 via a Lambda H100 rental**: base (unquantized) `Qwen2.5-VL-32B-Instruct` lifts the JevBench weighted 231-task cohort from 0.706 → 0.801 (+9.5 pp), with hard-tier reasoning +18 pp (0.423 → 0.604) and Brier roughly halved. Full report: [`product/reports/2026-10-10-h100-32b.md`](reports/2026-10-10-h100-32b.md). The AWQ-vs-fp16 quantization question is separate and still open — would be answered on L40S when available. **DigitalOcean backup**: DO lists `gpu-l40sx1-48gb` at $1.57/hr (~15% cheaper than AWS g6e.xlarge), gated behind a quota request filed 2026-10-10 — see [`research/digitalocean-gpu-options.md`](research/digitalocean-gpu-options.md).
- [x] `VLLMScorer` on the dev backbone: `max_tokens=1`, `logprobs=128`, `allowed_token_ids=label_ids`, `enable_prefix_caching=True`. Verified parity with HFScorer (3-decimal match).
- [x] FastAPI server exposing `/v1/decide`, `/healthz`, `/v1/models`. Thin wrapper around `decide()` and `generative_decide()`; honors `BERNOULLI_*` env config.
- [x] Batch all `questions × permutations` for a request into a single engine call via `score_batch` on the `Scorer` protocol — shared state prefix hits prefix cache.
- [x] Dockerfile for serving — `vllm/vllm-openai:v0.31.0-cu129-ubuntu2404` base, bernoulli layered on top, Docker data-root on NVMe. Dockerized server runs within <1% of bare-metal latency.
- [x] Load test: p50 / p95 / p99 at 1 / 5 / 20 questions per state — report in [`evals/reports/loadtest.md`](../evals/reports/loadtest.md).
- [-] ~~Flip the eval harness to score via HTTP against the running server~~ — **superseded by M8**: `benchmarks/common/baselines.py` will treat Bernoulli-over-HTTP as the first-class baseline, which replaces this refactor cleanly. Parity between the HTTP path and the direct scorer was already validated via spot checks (loadtest + VLLMScorer vs HFScorer, both in CLAUDE.md decisions log).

**Exit criteria**: Server runs in Docker, p50/p95 latency table reported. ✓ (M4e production-backbone step-up is explicitly parked; M4g rolled into M8.)

---

## M5 — Hardening (v1.0)
**Status**: ⏳ pending
**Goal**: v1.0-shippable text-only product. Auth, metrics, offline mode, CI regression gate, release tag.

- [ ] API-key auth
- [ ] Structured logging; state content never logged by default
- [ ] Prometheus `/metrics`
- [ ] Explicit offline mode: `HF_HUB_OFFLINE=1`, models from local path
- [ ] CI eval regression gate: fail if accuracy drops >1pt or ECE rises >0.01 vs stored baseline
- [ ] (Optional) React + Vite playground: paste state, add images (M6+), define questions, see bars
- [ ] v1.0 git tag + release notes

**Exit criteria**: v1.0 tagged with all gates passing (auth works, metrics scrape, offline run succeeds, regression gate green).

---

## M6 — Multimodal (image path)
**Status**: ⏳ pending
**Goal**: Add image support (up to 8 per state) to a service that's already in production.

- [ ] Confirm the production backbone supports a VL processor (or swap to a VL-capable sibling)
- [ ] Add `torchvision` to deps; wire `AutoProcessor` into both `HFScorer` and `VLLMScorer`
- [ ] Image path in `prompt.py`; cap at 8 images per state (`State.images` already validates this)
- [ ] Add two multimodal eval datasets: ScienceQA image subset, small Food-101 sample
- [ ] Multimodal eval reports in `evals/reports/`
- [ ] Load test the image path: p50 / p95 vs text-only at 1, 5, 20 questions

**Exit criteria**: Image-based decisions work end-to-end over HTTP; at least one multimodal eval report in `evals/reports/`; latency numbers include image-path cost.

---

## M7 — LoRA fine-tune (optional)
**Status**: ⏳ pending
**Goal**: LoRA on attention/FFN that beats zero-shot on held-out tasks not in the training mix.

- [ ] SFT mix from M3 train splits with randomized option order + randomized label letters
- [ ] Train on the answer token only; mask loss elsewhere
- [ ] LoRA on attn/FFN projections, vision tower frozen
- [ ] Re-run evals; re-fit calibration
- [ ] Held-out task eval (tasks NOT in SFT mix)
- [ ] Expose `bernoulli calibrate --data my_labeled.jsonl` for domain calibration

**Exit criteria**: LoRA checkpoint beats zero-shot on held-out tasks (OOD). If it only wins in-distribution, do not ship.

---

## M8 — Benchmarks: foundation + academic suite
**Status**: ✅ done (2026-10-15)
**Goal**: Stand up a top-level `benchmarks/` tree with a shared harness, migrate the existing `evals/` datasets, and extend to the 6-task academic suite from the Jev paper so our numbers are directly comparable.

Design decisions (locked here; revisit only with cause):
- **Layout**: `benchmarks/<category>/<name>/` with `README.md`, `run.py` (or shared runner), and `results/` co-located. No separate top-level `results/` folder — maintenance tax of two places to update is not worth it.
- **Common harness**: `benchmarks/common/{metrics,baselines,dataset}.py`. The leaderboard-style summary lives in `benchmarks/README.md` and links into each benchmark's `results/latest.md`.
- **Metrics**: accuracy, macro-F1, Brier, NLL, latency p50/p95 at 1 / 5 / 20 questions, cost per 1k decisions. ECE is reported in **both 10-bin (JevBench convention) and 15-bin (our existing reports)**; 10-bin becomes the headline so comparisons stay fair.
- **Stability**: for every benchmark report the Jev-style stability score — the fraction of examples whose top-class answer is unchanged under (a) option reorder and (b) a paraphrased question stem. This is our debiasing story in a number.
- **Coverage curves**: accuracy @ 95% / 90% / 80% / 50% autodecision rate (the curve behind the landing-page slider). Report as a 4-row sub-table under each benchmark.
- **Dropping BoolQ**: it's a reading-comprehension task, not a decision task, and the 63% hurts the story without testing what the product claims. Keep the `evals/reports/boolq.md` historical report in place; just don't carry BoolQ into `benchmarks/`.

Tasks:
- [x] Create `benchmarks/README.md` with the empty summary leaderboard, the "how to run" one-liner, and the methodology section (metric definitions, baseline conventions, stability protocol).
- [x] `benchmarks/common/dataset.py` — standardized `BenchmarkExample(state, question, gold, source, meta)` loader interface. Allow subclasses to add dataset-specific fields.
- [x] `benchmarks/common/metrics.py` — accuracy, macro-F1, ECE (10 + 15 bin), Brier, NLL, coverage curves at [0.95, 0.90, 0.80, 0.50], stability score. Pure numpy, no sklearn.
- [x] `benchmarks/common/baselines.py` — Bernoulli-via-HTTP + same-model generative baseline (lift from `evals/baselines.py`). **Split into 4a/4b/4c because each heavier baseline adds a new HF model + dependency and is cleaner to isolate per commit:**
  - [x] 4a. `Baseline` Protocol + `BernoulliHTTP` + `Generative` wrapper. No new deps beyond `httpx`.
  - [x] 4b. `DeBERTa` baseline — `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`. Adds a ~800 MB model pull on first use; no new pip deps (transformers already in). Later patched 2026-10-15 with `batch_size=32` default on the HF pipeline — without batching the 77-way Banking77 case was pathological (hours vs minutes).
  - [x] 4c. `BGEm3LR` baseline — `BAAI/bge-m3` + per-task logistic regression. Adds a ~2 GB model pull; scikit-learn added to dev group too (was only in the `eval` extra).
  - [-] Per-benchmark fine-tuned-encoder ceilings — deferred-in-practice. Belongs in each benchmark's subfolder, not here. Done case-by-case where a public fine-tune exists; nothing added yet.
- [x] Migrate SST-2 → `benchmarks/academic/sst2/`.
- [x] Migrate AG News → `benchmarks/academic/ag_news/`.
- [x] Migrate Banking77 → `benchmarks/academic/banking77/` (also fits "triage"; keep in academic for the Jev comparability table).
- [x] Add TweetEval-emotion → `benchmarks/academic/tweeteval_emotion/`.
- [x] Add PAWS → `benchmarks/academic/paws/`.
- [x] Add post-cutoff arXiv classification → `benchmarks/academic/arxiv_post_cutoff/`. Must include a `make_dataset.py` that pins the cutoff rule explicitly (dataset dates strictly later than the backbone's training cutoff) so the "couldn't have seen it" claim is defensible when we swap backbones. **Split into 10a (scaffolding + sample data) and 10b (actual run + full snapshot) because the real run needs the `arxiv` pip dep + network access on the dev box.**
  - [x] 10a. Loader + `make_dataset.py` skeleton + 3-row `data/sample.jsonl` + tests. No new deps yet.
  - [x] 10b. Ran `make_dataset.py` on the dev box 2026-10-07. 200 papers across cs.CL / cs.CV / math.PR / econ.EM, publication dates 2026-09-18 to 2026-10-06 (strictly after `BACKBONE_CUTOFF = 2025-01-01`, well after any plausible Qwen2.5-VL training cutoff). `data/dataset.jsonl` committed. First real-data benchmark run: Bernoulli 0.95 acc / ECE 0.047 / NLL 0.21 vs Generative 0.92 / 0.080 / 2.21 (**NLL 10.7× better** on provably-unseen data). `arxiv>=2.1` added to the `eval` extra + dev group. **2026-10-15 N=500 re-run confirms headline: Bernoulli 0.95 vs DeBERTa 0.53 — the "did the model memorize this?" test.**
- [x] Reword + reorder stability runner — reuse `bernoulli/debias.py` for reorder; add a lightweight paraphrase set per benchmark (3 reworded stems is enough).
- [x] Build each baseline for the 6 tasks (same-model generative, DeBERTa-zeroshot, BGE-m3 + LR). Fine-tuned-encoder ceiling is optional per task — do it where a public fine-tune exists; skip if we'd need to train one. **Split into 12a (runner CLI, no GPU) and 12b (actual sweep across all baselines × 6 tasks on the dev box, GPU-bound).**
  - [x] 12a. `benchmarks/run.py` CLI — load a benchmark, instantiate baselines, predict + compute metrics + stability, write `results/<date>.md`. One benchmark per invocation.
  - [x] 12b. **Closed 2026-10-15.** Full academic sweep at N=500 across all 4 baselines: Bernoulli + Generative + BGE-m3+LR via `benchmarks/scripts/run_main_sweep.sh` on the dev box; DeBERTa column via `benchmarks/scripts/run_deberta_sweep.sh` on a one-shot Lambda A100 (separate box sidesteps evicting the vLLM server). Results in `benchmarks/academic/*/results/2026-10-15.{main,deberta}.md`. First N=100 pass shipped 2026-10-07 (Bernoulli + Generative + BGE-m3+LR only — DeBERTa deferred that round because CPU was pathological beyond 2-way; batching patch + separate GPU resolved both constraints).
  - [x] 12c. **PAWS below-random investigation — resolved 2026-10-07.** Root cause: `reverse` debias on `BinaryQuestion` compounded the model's Yes-token prior instead of canceling position bias (`Yes`/`No` are themselves semantic tokens; reversing them doesn't just swap positions, it inverts meaning). Fix in `bernoulli/debias.py`: short-circuit reverse/cyclic to no-op for `BinaryQuestion` only. Post-fix PAWS N=100: Bernoulli 0.83 (was 0.37), ECE 0.076 (was 0.176). Full writeup + blast-radius analysis: [`product/learnings/paws-below-random.md`](learnings/paws-below-random.md).
- [x] First results run: `benchmarks/README.md` summary table filled on 2026-10-07 with the first-sweep numbers (post-PAWS-fix, post-arXiv-real-data). All 6 academic benchmarks have Bernoulli / Generative / BGE-m3+LR rows. Coverage sub-tables and the full stability column live in each benchmark's `results/<date>.md`; the summary carries headline accuracy + NLL + ECE + stability pair. Honest-caveats section documents the N=100 / no-DeBERTa / Banking77-chunked story.

Exit criteria: `benchmarks/README.md` renders a comparison table with ≥ 5 of the 6 academic tasks × ≥ 3 baselines (us, same-model generative, DeBERTa-zeroshot). Each task's `results/latest.md` has coverage sub-table and stability score. Historical `evals/reports/*.md` left in place as a snapshot of pre-migration numbers.

---

## M9 — Benchmarks: use-case coverage
**Status**: ✅ done (WildGuardTest run deferred — gated on HF)
**Goal**: Match the landing-page use-case cards (guardrails, support triage, content moderation / rating) with benchmarks whose results we can cite in the pitch. Each use case gets a baseline comparison against the model category that actually competes.

Tasks:
- [x] `benchmarks/guardrails/wildguard_test/` — WildGuardTest. Baselines: Llama Guard 3, ShieldGemma, WildGuard-7B (these also read token probabilities, so the comparison is apples-to-apples).
- [x] `benchmarks/guardrails/toxicchat/` — ToxicChat. Same baselines as above where applicable.
- [x] `benchmarks/guardrails/xstest/` — XSTest for over-refusal. Report refusal-rate and accuracy separately; a good guardrail is accurate *without* over-refusing.
- [x] `benchmarks/triage/clinc150_oos/` — CLINC150 with the out-of-scope split. Report OOS detection AUROC alongside in-domain accuracy — this is the "none of these" probability story in a number.
- [x] `benchmarks/ratings/yelp_stars/` — Yelp 1-5 star reviews. Tests the rating question type end-to-end. Report MAE and off-by-one accuracy in addition to the standard metrics.
- [x] **M9 sweep run** — shipped 4 of 5 (ToxicChat, XSTest, CLINC150-OOS, Yelp 1-5) at N=100 on 2026-10-07. WildGuardTest is still gated on HF; see its README for the auth prerequisite. Several loader fixes landed along the way (legacy HF paths namespaced, XSTest split `train` → `prompts`, CLINC load+load_train shuffle for class balance, three `__init__.py` re-exports of `extra_metrics` so the runner's `getattr` picks them up).
- [x] Summary leaderboard tables in `benchmarks/README.md` filled for all 5 benchmarks (minus WildGuardTest) across M8+M9. Added an "Honest caveats" section documenting what the snapshot is NOT (N=100, no DeBERTa column, Banking77 reorder=0.13 chunked footnote, XSTest unsafe_recall=0.36, CLINC OOS recall ≈0.52, dev-tier backbone).
- [x] Cross-reference: the three use-case cards on `web/index.html` now carry a `.uc-proof` footer linking to the matching benchmark on `/benchmarks.html#usecase` with specific numbers inline (CLINC150-OOS 18× NLL, XSTest 0.00 false-refusal, Yelp MAE 0.44).

Exit criteria: Each use-case card on the landing page has at least one benchmark result backing it. Guardrails has ≥ 2 of the 3 datasets with ≥ 1 external baseline each. CLINC150 OOS AUROC reported. Yelp stars MAE + off-by-one reported.

---

## M10 — JevBench adapter + leaderboard submission
**Status**: ✅ done (2026-10-10 — submission queued at FIFO #62; three follow-ups below pending Benchmark Heaven posting the entry)
**Goal**: Run Bernoulli through the official JevBench harness and submit to its public leaderboard. Getting listed is distribution; being measured on their rules is credibility.

Tasks:
- [x] Scaffold `benchmarks/jevbench/` — package, README, `results/` dir. The README spells out the three unknowns (upstream URL, adapter contract, submission flow) and the implementation plan once they're answered. **No adapter code yet** — writing stubs that might not match the real contract would be worse than a clear "here's what we need" doc.
- [x] Resolve the three unknowns — done 2026-10-07 via public-web investigation. Short answers (full detail in `benchmarks/jevbench/README.md`):
  - **Upstream**: hub at [benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models); Python harness mirrored on GitHub (3 identical copies — pick [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench) as canonical).
  - **Contract**: self-runnable harness with `jevbench.cli run --adapter <type> --model <name>`. Adapters shipped: `typesafe` / `openai-compatible` / `local` / `gradio`. We run it, submit the numbers.
  - **Submission**: web form at [benchmarkheaven.com/submit](https://benchmarkheaven.com/submit). Required: model name, access URL, email, benchmark selection. Free FIFO queue; paid 48h fast lane. **Not an API submission.**
### M10 execution plan

Five concrete steps, each with inputs/outputs/effort. Steps 1-4 are hands-on code work (~3-5 hours total); step 5 is wall-clock on Benchmark Heaven's FIFO queue (unknown — free queue is "evaluated in the order received").

1. **[x] Pin + vendor the harness** — `git submodule add https://github.com/fstandhartinger/jevbench benchmarks/jevbench/upstream` at `bb05a335bc`. (Didn't run the `typesafe` reference adapter — it requires a TypeSafe API key. Harness integrity verified by using it to summarise our own runs in step 3.)
2. **[x] Write `benchmarks/jevbench/adapter.py`** — `BernoulliLocalAdapter` registered in `benchmarks/jevbench/run.py`. Chose a bespoke adapter over the shipped `openai_compat` path (which marks `probs_source="verbalized"` — specifically what Bernoulli is NOT). Mapping: `noul` → `BinaryQuestion`, `choice` → `ChoiceQuestion` (rubric appended to prompt per upstream convention), `score` → `RatingQuestion` with 0-indexed ↔ 1-based shift. 11 unit tests in `tests/test_jevbench_adapter.py`.
3. **[x] Run the public cohort locally** — 2026-10-08. All three public splits (`original` 72 + `easy` 48 + `hard` 111 = **231 tasks**, not 242 — the leaderboard count includes sealed items we don't have). Writeup in `benchmarks/jevbench/results/2026-10-08.md`. Headline:
   - `original`: acc 0.833, Brier 0.305, ECE 0.116, p50 92 ms, paraphrase_agreement 0.944
   - `easy`: acc 1.000, Brier 0.0002, p50 92 ms
   - `hard`: acc 0.432, Brier 0.730, ECE 0.264, p50 338 ms
   - 100 % operational_success + schema_validity_strict + coverage across all 231; `probs_source="native"` throughout.
4. **[x] Sanity-check against the live board** — leaderboard tops at 67.37 (Imajev-4B); top-10 clusters 58–65. **Caught + fixed an adapter off-by-one** during this step: our score-task legend was 0-indexed while the `RatingQuestion` server-rendered options are 1-indexed, so the model picked the letter one below correct and `_shape_probs` shifted it back to one below `expected`. Pattern was obvious once per-family numbers were pulled (all 8/8 failures off-by-one in the same direction). One-line fix in `benchmarks/jevbench/adapter.py`; ordinal 0.333 → 1.000, standard tier 0.833 → **0.944**, weighted 231-cohort 0.675 → **0.706**. Now 3.5 pp below the lowest-ranked top-20 system on the live v1.4.2.2 board — plausibly in "7B base model" territory, not "broken adapter" territory.
5. **[x] Submit via the web form** at [benchmarkheaven.com/submit](https://benchmarkheaven.com/submit) — **submitted 2026-10-10**, queued at position **#62** in Benchmark Heaven's free FIFO queue. Submission pinned to repo SHA `09e2dd3` (the ordinal-fix commit that produced the published numbers). No ETA published; expect days to weeks. Full form content captured in `benchmarks/jevbench/SUBMISSION.md`.

Follow-ups once the leaderboard entry posts:

- [ ] Link the entry from the repo `README.md` + `web/benchmarks.html` ("as ranked on JevBench").
- [ ] Append a benchmark-state snapshot documenting where we rank + what the gap to #1 is.
- [ ] If the Capability Score is uncompetitive, treat as learning signal — JevBench scoring axes may surface weaknesses our existing suite doesn't (e.g. cost per 1k decisions on dev-tier A10G is probably not favorable vs. the 4B-parameter leaders).

Exit criteria: Bernoulli listed on the JevBench leaderboard with a reproducible run captured in `benchmarks/jevbench/results/`. The submission references a pinned model revision and a pinned Bernoulli commit.

---

## M11 — Public verification endpoint
**Status**: ⏳ pending
**Goal**: Stand up a bookable, rate-limited public `/v1/decide` endpoint so third parties (and in particular Benchmark Heaven's JevBench harness) can send decisions directly to Bernoulli without the submitter's hardware mediating. This upgrades our JevBench row from "shown but not ranked" (public-items only, per the v1.4.2 self-hosted-endpoint rule) to a **ranked entry measured on the sealed held-out cohort**.

**Why this matters:** JevBench's v1.4.2 release notes make the rule explicit — *"this round stopped sending held-out items to an endpoint a submitter operates"* — because a submitter can see item text (and thus identify held-out items) when it transits their own server. The only way to earn a ranked row is to make the endpoint operated by someone **not** us; in practice that means a bookable public endpoint on infra Benchmark Heaven can hit directly, with no human in the loop on our side. Same logic applies to any independent audit of our benchmark claims — a public endpoint turns self-reported numbers into verifiable ones.

**Scope boundaries (what this is / isn't):**
- *Is:* a thin, rate-limited public wrapper around the existing `/v1/decide` server, with auth, abuse controls, and an uptime commitment Benchmark Heaven's harness can rely on. Could be the production serving infra (M4e production backbone on g6e/L40S) exposed publicly, or a separate smaller demo endpoint.
- *Isn't:* a general public SaaS. The target traffic is benchmark harnesses + occasional evaluators, not production workloads.
- *Isn't:* a decision we can make before M5 is done — exposing an endpoint before the hardening work (regression gate, request validation, resource caps) would be reckless.

### Tasks

- [~] **Decide the shape** — compared in [`research/public-demo-options.md`](research/public-demo-options.md), validated in-flight by the [`research/spikes/hf-space-demo.md`](research/spikes/hf-space-demo.md) spike (closed 2026-10-08). **Working recommendation**: keep the HF Space ([shyamsfo-bernoulli-demo.hf.space](https://shyamsfo-bernoulli-demo.hf.space)) as the ongoing *friends-and-nerds* demo (zero adapter code, exact-parity with AWS, ~$0 idle on prepaid); build M11-ranked as a *separate* hardened endpoint on dedicated AWS infra (Cloudflare Tunnel + auth + rate limits + uptime SLO). HF Spaces is not the right shape for the ranked endpoint — sleep/wake and shared-container runtime work against the uptime SLO a sealed-cohort re-run needs. Still to finalise: API-key issuance process + rate-limit numbers + the auth story.
- [ ] **Pick an auth model** — API key per external evaluator vs IP allowlist vs HMAC-signed requests vs no auth + strict per-IP rate limiting. The constraint: Benchmark Heaven's harness needs to call it without a human in the loop on their side either.
- [ ] **Abuse + cost controls** — token/decision budget per API key, per-IP rate limit (requests/min + burst), timeout per request, max input size, max label-set size. The GPU is finite; one misbehaving caller can starve the sealed re-run.
- [ ] **Observability** — logs that identify each request's caller + task-shape (not content) so we can tell a benchmark re-run apart from abuse. Dashboard or at least `tail -f` story.
- [ ] **Uptime SLO** — Benchmark Heaven's harness needs a stable endpoint; occasional spot preemptions on g5/g6e are not acceptable. Either move to on-demand pricing for the public endpoint, run two spot instances behind a load balancer, or accept failed runs and document the retry story.
- [ ] **Public documentation** — a `HOSTING.md` with the endpoint URL, auth model, rate limits, request/response shape, and the privacy statement (what we log, what we don't). Linked from `README.md` + `web/`.
- [ ] **Resubmit to JevBench** with the public endpoint URL in the submission form so the harness can hit it directly for the sealed re-run.

**Exit criteria**: Benchmark Heaven's harness has run the full JevBench sealed cohort against our public endpoint and we are listed as a **ranked** entry on [benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models). Public endpoint has survived 30 days without a sustained outage or an abuse incident that required an emergency shutdown.

**Dependencies:**
- **Blocks on:** M5 (hardening — regression gate + request validation + resource caps before external traffic), M4e (production backbone step-up — if the public endpoint is going to be the ranked system, it should be the 32B-AWQ production backbone, not the dev-tier 7B).
- **Does not block:** anything downstream. M11 is purely about distribution + verification; the technique itself is proven by M1–M10.

**Open questions parked for when we pick this up:**
- Pay-per-use SaaS vs free public demo? Free is more inviting; pay-per-use is more sustainable. A middle ground: free with a monthly decision cap per key.
- Do we expose `/v1/generate` too, or Bernoulli-only? The baselines use it internally — exposing it publicly means we're also hosting a general LLM text endpoint, which is a different liability profile.

---

## M13 — Daily-serving public endpoint for Salesforce dev
**Status**: ✅ done (2026-10-29 — endpoint live, Salesforce dev-org Apex callout verified, Chrome extension is already a second client, OOM path hardened in b666eec)
**Goal**: A single-tenant, dev-tier public endpoint at `https://bernoulli-api.nuwire.dev` that a Salesforce dev org can call via Apex. Operates with `just serve-up` / `just serve-start` / `just serve-down` daily. **Not** production-hardened — API-key auth only, no TLS cert on our side (TLS terminated by ssd2), no systemd restart-on-crash, no metrics, no rate limiting. Full hardening is M5+M11 territory later.

**Design (locked):**
- **Public URL**: `https://bernoulli-api.nuwire.dev` (ClouDNS → ssd2, static).
- **TLS termination**: ssd2 nginx with the existing `*.nuwire.dev` wildcard cert — same operational pattern as `gr{1,2,3}.nuwire.dev`. One nginx server block, already shipped. Nginx re-resolves `bernoulli-internal.nuwire.dev` every 30s via the `resolver` directive, so EC2 ephemeral-IP rotation is handled by updating one ClouDNS record (via `~/bin/cloudns_mgr`).
- **Backend**: EC2 g5.xlarge (existing dev box), ephemeral public IP, security group opens tcp/8000 only to ssd2's /32. API traffic travels: Salesforce → ssd2 (HTTPS) → EC2 (HTTP:8000). Public internet cannot reach :8000 directly.
- **Server**: `uv run uvicorn bernoulli.server:app --host 0.0.0.0 --port 8000`, launched via `nohup` on the box. **No systemd, no Docker for now** — minimum ceremony. If uvicorn crashes, re-run `just serve-start`.
- **Auth**: `BERNOULLI_API_KEYS` env var — 5 hardcoded 64-char hex tokens stored in `~/.credentials` on laptop, injected into the server at `serve-start` time. API-key middleware already in `bernoulli/server.py` from 2026-10-17. `/healthz` exempt.
- **Scorer**: `BERNOULLI_SCORER=hf` (transformers direct). Simpler install than vLLM, supports both `/v1/decide` and `/v1/generate`, ~70 ms per question. Can flip to vLLM later via env var if we want prefix-caching for multi-question requests.
- **Cost shape**: $1/hr g5.xlarge only when running. Dev-day usage ~4-8 h × 20 workdays = ~$80-160/mo. $0 nights/weekends.

**Already shipped (2026-10-17 commits dcef495 + d082606):**
- API-key middleware in `bernoulli/server.py` + 8 tests.
- Terraform SG ingress for port 8000 from ssd2's /32.
- `just serve-up` / `serve-down` / `serve-status` recipes.
- ssd2 nginx config at `/etc/nginx/sites-enabled/bernoulli-api.nuwire.dev` (reverse-proxy pattern copied from gr-slots).
- ClouDNS records `bernoulli-api.nuwire.dev` (static → ssd2) + `bernoulli-internal.nuwire.dev` (placeholder → RFC 5737).
- `deploy/user_data.sh` symlink of `uv` into `/usr/local/bin` so non-interactive SSH finds it.
- `just model-pull` recipes use `uv run hf download` (bare `hf` was broken on non-interactive SSH).
- 5 API keys generated and stored in `~/.credentials` as `BERNOULLI_API_KEYS=k1,k2,k3,k4,k5`.

### Tasks

- [x] Rewrite `just serve-provision`: sync-mirror + `uv sync` on box + `just model-pull`. **No Docker build.** No systemd unit install. (Shipped as `deploy/bringup.sh` in 926d053 — stepwise + resumable, with explicit per-step timeouts that catch the UEFI-boot-loop class of failures.)
- [x] Add `just serve-start`: launches uvicorn via `setsid --fork` + `deploy/start-serve.sh`, with `BERNOULLI_API_KEYS` sourced from `~/.bernoulli.env` (chmod 600) to keep keys out of `ps`. Waits for local `/healthz`. (a01f793 + 0785f5f)
- [x] Add `just serve-stop`. (a01f793)
- [x] Start EC2, run `just serve-provision` end-to-end.
- [x] Run `just serve-start`. Verify local `/healthz` on the box responds.
- [x] Verify public endpoint: `curl -H "Authorization: Bearer <key>" https://bernoulli-api.nuwire.dev/healthz` returns 200.
- [x] Verify auth works: same curl without the header returns 401.
- [x] Verify a real decision call: POST a small `/v1/decide` request with a bearer token, get a typed response. (Verified repeatedly; `just serve-check` smokes this against three sample fixtures each start.)
- [x] Add Salesforce Remote Site Settings entry for `https://bernoulli-api.nuwire.dev` and write a Named Credential with one of the API keys. Make a working Apex callout from the dev org. (Working as of 2026-10-29 per user; Chrome extension is a second client against the same endpoint.)
- [x] ~~Document the daily workflow in a short `HOSTING.md`~~ — **struck as redundant.** The daily workflow is already covered in `CLAUDE.md` (commands cheat sheet) and the `serve-*` recipe headers in `justfile`. Adding `HOSTING.md` would duplicate without adding value.

**Exit criteria**: Salesforce Apex callout from the dev org successfully receives a typed decision from `https://bernoulli-api.nuwire.dev/v1/decide`, authenticated by an API key, with the EC2 box having been started that morning via `just serve-up` + `just serve-start`.

**Not in scope (deferred to M5 + M11):**
- Systemd unit for restart-on-crash and restart-on-reboot.
- Prometheus `/metrics`.
- Structured logging with content redaction.
- Offline mode (`HF_HUB_OFFLINE=1`).
- CI regression gate.
- Rate limiting on the API side.
- Load balancer / multiple backends.
- Elastic IP (vs ephemeral + ClouDNS update).
- Monitoring + uptime SLO.
- `/v1/decide` benchmark re-measurement against the live endpoint.

**Dependencies:**
- **Blocks on**: nothing. The infra pieces are all shipped except the serve-provision recipe rewrite.
- **Does not block**: anything downstream — this is an independent dev-affordance.
- **Supersedes but doesn't replace**: parts of M11 (public verification endpoint). M11 remains the "hardened + ranked" version to pursue once the Salesforce feature has real users.

---

## M12 — Landing-page visitor analytics (self-hosted Umami on ssd2)
**Status**: ⏳ pending
**Goal**: Visitor traffic to [bernoulli.live](https://www.bernoulli.live/) is visible in a self-hosted, cookieless analytics dashboard at `analytics.bernoulli.live`, with no GDPR cookie banner surface and no third-party JS on the landing page.

**Context (already researched, don't redo):**
- DNS provider: **ClouDNS**, zone `bernoulli.live`. Creds live in `~/.credentials` as `CLOUDNS_AUTH_ID` + `CLOUDNS_AUTH_PASSWORD`. API: `https://api.cloudns.net/dns/add-record.json` (GET, `delete-record.json` to roll back).
- ssd2 public IP: **208.87.128.216** (= `b2` in `~/.ssh/config` = apex `bernoulli.live` A record). Analytics subdomain mirrors this.
- Chosen stack: **Umami + Postgres** via `docker-compose`, reverse-proxied by the existing ssd2 nginx with Let's Encrypt (same pattern as the main site).

**Stack rationale (don't relitigate):**
- Umami over Plausible CE: simpler stack (no ClickHouse), ~300 MB RAM vs ~1.5 GB. UI is less polished but sufficient for a landing page.
- Self-host over hosted Plausible/Fathom (~$9–15/mo): defer paying until there's a reason.
- Cookieless (Umami, Plausible, GoatCounter) over cookie-based (Clarity, HubSpot, GA4): no cookie banner, no GDPR consent surface on the landing page.
- GoatCounter free tier rejected: non-commercial clause is ambiguous once we offer managed hosting (M11-adjacent).

**Tasks:**
- [ ] Verify ssd2 free RAM/disk headroom (`ssh b2 'free -h && df -h /'`). Target: ≥500 MB free RAM, ≥2 GB free disk.
- [ ] Add DNS record `analytics.bernoulli.live A 208.87.128.216` (TTL 3600) via ClouDNS API. Verify with `dig +short analytics.bernoulli.live`.
- [ ] Create `/opt/umami/` on ssd2 with `docker-compose.yml` (Umami + Postgres, Umami bound to `127.0.0.1:3000` only — no public port exposure). Bring up and verify locally with `curl http://127.0.0.1:3000`.
- [ ] Add nginx server block for `analytics.bernoulli.live` reverse-proxying to `127.0.0.1:3000`; issue cert via existing certbot flow; reload nginx. Verify TLS via `curl -I https://analytics.bernoulli.live`.
- [ ] First login (default `admin / umami`), rotate admin password, store under a new key in `~/.credentials` (e.g. `UMAMI_ADMIN_PASSWORD`).
- [ ] Add `bernoulli.live` as a tracked site in Umami; copy the generated `<script>` tag.
- [ ] Insert the script tag in `web/index.html` before `</body>` and `just deploy`. The script is served from `analytics.bernoulli.live` (first-party, no third-party domain).
- [ ] Load the live site in an incognito window; verify the pageview lands in the Umami dashboard within a minute.

**Exit criteria**: `https://analytics.bernoulli.live` loads with a valid TLS cert and the Umami dashboard; it shows at least one real pageview from `www.bernoulli.live`; no cookie banner required.

**Scope boundaries (don't scope-creep):**
- *Is:* traffic counting + referrer + country + device for the landing page.
- *Isn't:* session replay (that's Clarity + a banner — defer).
- *Isn't:* CRM-linked lead tracking from the book-a-demo form (that's HubSpot + a banner — defer; the demo form already writes to a Google Sheet).

**Future upgrades (parked, not in this milestone):**
- Session replay via Microsoft Clarity (would require a GDPR banner — add only if the drop-off story becomes a real question).
- Form-fill → CRM linkage via HubSpot free CRM (would require a GDPR banner — add once book-a-demo volume justifies it).
- Postgres backup cron (`pg_dump` nightly to ssd2's existing backup target) — add as a `/ds-work-recurring` item once Umami is live, not as a task here.

**Dependencies:**
- **Blocks on:** nothing. Fully independent of the core product milestones.
- **Does not block:** anything.

---

## M14 — Jev API-compatibility shim
**Status**: ⏳ pending (parked; pick up when a specific customer or integration needs it)
**Goal**: A `/jev/v1/*` endpoint set on the Bernoulli server that accepts Jev's wire-format requests and returns Jev-shaped responses, so a client written against Jev's SDK can swap to Bernoulli with URL + API key config only (no application-code changes).

**Why this isn't the current pitch:**
Our native `/v1/decide` schema is our own invention — pydantic v2 discriminated unions with `extra='forbid'`, our field names (`state`, `questions`, `options`, `decisions`), our question-type vocabulary (`choice` / `binary` / `rating`). It was designed for machine-friendly strictness, not for mimicry. The public pitch stays "same technique, open source, your infra" — not "drop-in binary-compatible with Jev." A compat shim is an **integration affordance**, not a correctness goal.

**When to pursue:**
- A specific customer with an existing Jev integration asks for migration-without-code-changes.
- JevBench ever stops accepting self-reported numbers and demands a Jev-shaped public endpoint for ranked submission.
- We want to be listed on a third-party "Jev-compatible alternatives" comparison and need the wire-format match for that story.

**Design sketch (not committed; validate at build time):**
- New router under `bernoulli/jev_compat.py` mounted at `/jev/v1/*` on the FastAPI app.
- Thin translation layer: Jev-shaped `DecideRequest` → internal `bernoulli.types.DecideRequest`; internal `DecideResponse` → Jev-shaped response.
- Share the same `decide()` function + scorer; the compat layer only touches request/response shape.
- Pin a specific Jev API version (e.g., `v1.4.2` matching the current JevBench upstream) rather than tracking Jev's live API.
- Error-shape translation: Jev's error envelope → our 422/401/500 surface.
- Jev's auth flow (if different from Bearer token) handled in a middleware wrapper, OR hard-require the same `BERNOULLI_API_KEYS` story since we're not a hosted service.

**Reference material we already have:**
- [`benchmarks/jevbench/adapter.py`](../benchmarks/jevbench/adapter.py) — our `BernoulliLocalAdapter` that goes the *other* direction (JevBench harness → Bernoulli). The field-name and question-type mapping is reversible from this file.
- [`benchmarks/jevbench/upstream/`](../benchmarks/jevbench/upstream/) — the vendored JevBench harness; its `typesafe` adapter is the ground truth for the real Jev wire format.
- Jev public docs at [docs.typesafe.ai](https://docs.typesafe.ai) — for anything not covered by the harness's reference adapter.

### Tasks (deferred; capture only, no action)

- [ ] Pin a specific Jev API version to target (recommend v1.4.2 — current JevBench upstream reference).
- [ ] Reverse-engineer field mapping from `benchmarks/jevbench/upstream/`'s `typesafe` adapter + Jev public docs. Document the mapping in a design note.
- [ ] Add `bernoulli/jev_compat.py` with request/response translator + the three question-type mappers (`noul` ↔ binary, `choice` ↔ choice, `score` ↔ rating).
- [ ] Mount `/jev/v1/decide` (or whatever Jev calls their primary endpoint) on `bernoulli/server.py`.
- [ ] Add a smoke test suite: feed Jev-shaped fixture requests to the compat endpoint, verify the response matches Jev's expected shape at least on the three canonical question types.
- [ ] Document on the landing page + USECASES.md: "Jev-SDK clients can point at `/jev/v1/*` for drop-in migration." Not a headline feature; a footnote.

**Exit criteria**: A client that currently targets Jev's API can swap one URL + API-key config and have their application run against Bernoulli unchanged, verified by a smoke test using Jev's own Python SDK (or a direct curl of Jev-shaped payloads).

**Dependencies:**
- **Blocks on:** nothing structural. The native `/v1/decide` + scorer pipeline doesn't need any changes.
- **Does not block:** anything.

**Expected effort**: ~0.5-1 day for a minimum-viable shim covering the three question types. More if we discover Jev has a lot of surface area (batching options, streaming, callbacks, etc.) we don't currently expose.

---

## M15 — API usage statistics
**Status**: ⏳ pending (defined only — further planning required before execution; run `/ds-work-plan M15` when ready)
**Goal**: Know who's calling the Bernoulli API, how often, what question types, how fast — enough to make product decisions about Chrome-plugin vs Salesforce-plugin usage without flying blind.

**Why now:**
Expected first-wave callers are the "Bernoulli for Terms" Chrome extension and Salesforce plugins (M13's daily-serving endpoint is already live for the latter). Today we have no visibility beyond nginx access logs on ssd2, which can't see the question type, label count, calibration state, or scorer timing — the fields that matter for product calls. Volume is still low enough that we can get the instrumentation in before it starts to matter, which is the right order.

**Recommended shape (from the 2026-10-29 scoping discussion — not binding; validate at planning time):**
- **Approach**: application-level JSONL event log via a FastAPI middleware wrapping `/v1/decide` and `/v1/generate`. Not nginx-only (can't see question type); not a full TSDB (premature at one box, handful of clients).
- **Per-event fields (sketch)**: ts, request_id, hashed api_key_id (NOT raw key), endpoint, status, latency_ms, scorer_ms, model, calibration_version, question_count, question_types, label_counts, state_text_bytes (count only — never the text), state_images, debias, calibrated, origin, user_agent, error_type.
- **Client attribution**: `Origin` header + `User-Agent` is enough to separate Chrome-extension traffic from Salesforce-plugin traffic. Confirm during planning that Salesforce Apex callouts expose a usable UA.
- **Storage + rotation**: `~/events.jsonl` on the serving box's EBS (persists across stop/start). logrotate daily, gzip, retain 30 days.
- **Analytics path**: `just bench stats` recipe rsyncs the event log down, runs a DuckDB query over the JSONL, writes a markdown report into `reports/`. No DB service, no dashboard — just a reproducible weekly report.
- **Later (not this milestone)**: Prometheus `/metrics` endpoint for alerting; per-key quotas / rate limits; shipping events to S3 as parquet; customer-facing usage dashboard.

**Open questions to settle in planning (`/ds-work-plan M15`):**
- Exact field list — are we missing anything a product decision will need? (e.g., truncation flags, cache-hit flags once the Chrome extension's local cache can be measured server-side.)
- Fire-and-forget mechanics — `asyncio.create_task` with a bounded queue vs a background thread. What's the drop policy if the queue fills during a traffic spike?
- Hash scheme for api_key_id — SHA-256 prefix is fine, but confirm it stays stable across key rotation (same key → same id).
- Correlation across layers — do we also need to echo `X-Request-Id` on the response so clients can report "my request `req_...` errored" and we can look it up? Likely yes; cheap to do.
- Report format — one weekly markdown? Or one per client (Chrome / Salesforce)? Pick after a week of real data.

**Exit criteria** (to be refined during planning): one week of real traffic produces a weekly report that answers top-N API keys by request count, question-type mix, p50/p95 latency, 4xx/5xx rate, and the Chrome-vs-Salesforce split.

**Dependencies:**
- **Blocks on:** M13 serving endpoint being up (it is).
- **Does not block:** anything structural. Can land anytime.

**Expected effort**: 0.5-1 day for the writer + middleware + logrotate + a stats recipe. Longer if the "what to capture" conversation keeps expanding scope.

---

## M16 — API key management (SQLite + CLI)
**Status**: ⏳ pending (planned 2026-10-29 — queued behind M17 split so admin.py's OSS vs private location is clean from day one)
**Goal**: Issue, label, revoke, and rotate individual API keys without restarting the server or disrupting other keys. Replace the current `BERNOULLI_API_KEYS` env-list in `~/.bernoulli.env` with a tiny SQLite-backed store that still has zero new runtime deps and no new services.

**Why now:**
Today's model — 5 comma-separated keys in an env file, loaded at startup — works for a handful of trusted recipients but has three pain points that compound as real customers show up:
- **No selective revocation.** Rotating one key means editing the env file and restarting the server, which drops every other key's in-flight requests.
- **No attribution.** We can't tell who got key #3 or when; label-free keys make leaks painful to triage.
- **No join into usage stats.** M15 is going to log `api_key_id = sha256(key)[:12]` as a stand-in. With real labelled ids, the weekly report can say "the Acme Salesforce plugin burned 40% of requests this week" instead of "key_abc12345".

Separate from "we need auth" — we have auth. This is purely a key-lifecycle upgrade.

**Recommended shape (from the 2026-10-29 scoping discussion — not binding; validate at planning time):**
- **Tier choice: Tier 1** — small SQLite file + CLI. Not Tier 0 (env list, current), not Tier 2 (managed service like Auth0 / Kong — overkill for a 1-2 person shop), not Tier 3 (OAuth/JWT — wrong trust model for programmatic server-to-server traffic, and JWTs lose cheap revocation which is the main thing we want).
- **Format convention (Stripe-style)**: keys look like `bnl_live_<32 hex>` (or `bnl_test_...` for a future test tier). The visible prefix means a leaked key is identifiable in logs/stack traces/git grep, and GitHub secret scanning accepts vendor patterns — once we have real external customers, submit `bnl_live_[a-f0-9]{32}` to auto-flag accidental commits.
- **Storage**: `~/.bernoulli-keys.db` on the serving box's EBS. One table — `(id TEXT PK, key_hash TEXT, label TEXT, created_at INT, revoked_at INT, notes TEXT)`. Key material never stored, only `sha256(key)`. Stdlib `sqlite3` only; no SQLAlchemy, no ORM.
- **CLI** (`bernoulli-admin` or a `just` recipe): `issue --label <...> [--notes <...>]` (prints the key ONCE; refuse to show again), `list`, `revoke <id>`, `rotate <id>` (= revoke + issue under same label). Script lives in `bernoulli/admin.py` or `scripts/`.
- **Middleware**: load the active-key set at startup into an in-memory `set[str]` keyed by `sha256(key)`. No DB hit per request. Reload on SIGUSR1 (or on a `POST /admin/refresh-keys` endpoint protected by a bootstrap admin key). CLI issue/revoke sends the signal automatically so the server picks up changes without restart.
- **Migration**: a one-shot script that reads the current 5 `BERNOULLI_API_KEYS` entries and ingests them with placeholder labels (`legacy-env-slot-0` ... `legacy-env-slot-4`) so no customer sees a 401 during cutover.
- **Later (not this milestone)**: scopes/RBAC (we have one endpoint tier), per-key rate limits (nginx `limit_req` or in-process token bucket when needed), web UI for issuing (over-invest until >50 keys or non-engineer admins).

**Standard-libs guidance:**
- Use `secrets.token_urlsafe(32)` / `secrets.token_hex(32)` for key material. Nothing fancier needed.
- Use SHA-256 for the hash column, **not** argon2/bcrypt. Argon2 is for low-entropy human passwords where you want slowness; API keys are high-entropy random bytes where you want fast comparison.
- Avoid the small OSS "fastapi-api-key-auth" style packages — they're 20 lines wrapped in a dependency. Hand-rolled middleware is better.

**Decisions (resolved 2026-10-29):**
- **Reload mechanism**: *none for now* — the server reads the active-key set from the SQLite file at startup only. Issuing or revoking a key takes effect on the next `serve-stop` + `serve-start` cycle. Load is light enough that the ~5-min cold-reload cost is acceptable, and startup-only loading avoids the sharp edges of signal handling inside async FastAPI apps. SIGUSR1 hot-reload parked for later if restart downtime starts hurting.
- **CLI location**: `bernoulli/admin.py` module invoked as `uv run python -m bernoulli.admin ...`, with a `just admin <subcommand>` wrapper recipe for ergonomic invocation from the laptop. Keeps the CLI shippable with the package.
- **Rotation UX**: revoke-immediately. If a caller needs a grace period, the explicit pattern is `issue` new → deploy to the client → `revoke` old (two commands, caller-owned timing). Avoids a `grace_until` column + middleware check for marginal gain.
- **Hashing**: plain SHA-256 of the full key. Keys have ~128 bits of entropy from `secrets.token_hex(32)`, so rainbow tables are infeasible; HMAC-with-DB-secret adds complexity for near-zero security gain.
- **Backup**: nightly cron on the box — `rsync ~/.bernoulli-keys.db ssd2:bernoulli-key-backups/$(date +%F).db`. Retain 30 nights on ssd2. EBS snapshots are a secondary line of defense; this gives off-box redundancy for cheap. Same trust zone as everything else (ssd2 is already our TLS termination + reverse-proxy host).
- **Client attribution**: single `label` column, free-form string (`"salesforce-acme-prod"`, `"chrome-ext-shyam-dev"`). Doubles as the join key into M15's event log. If a single client eventually has multiple keys needing to be grouped, promoting `label` to `(client_id, environment, purpose)` columns is a cheap future migration — parked.

### Tasks

- [ ] Add `bernoulli/admin.py` — SQLite schema `(id TEXT PK, key_hash TEXT NOT NULL, label TEXT NOT NULL, created_at INT NOT NULL, revoked_at INT, notes TEXT)` + `issue` / `list` / `revoke` / `rotate` subcommands via `argparse`. Key format: `bnl_live_<32 hex>` from `secrets.token_hex(32)`. `id` format: short ULID-ish (e.g., `key_` + 10-char token). `issue --label <...> [--notes <...>]` prints the full key once to stdout and refuses to re-print on subsequent `list` calls (list shows id + label + status only). Refuse to issue if label is blank.
- [ ] On first run of any admin subcommand: create the table if missing + ingest existing `BERNOULLI_API_KEYS` env values with placeholder labels (`legacy-env-slot-0` … `legacy-env-slot-N`). Idempotent — safe to re-run without duplicating. DB file lives at `~/.bernoulli-keys.db` chmod 600.
- [ ] Add tests in `tests/test_admin.py` — schema creation, issue+list roundtrip, revoke sets `revoked_at`, rotate = revoke+issue under same label, SHA-256 hash matches `sha256(key).hexdigest()`, legacy env ingest is idempotent. CPU-only (no GPU dependency).
- [ ] Rewrite auth middleware in `bernoulli/server.py` to load the active-key set from `~/.bernoulli-keys.db` at startup into an in-memory `dict[str, dict]` keyed by `sha256(bearer)` → `{id, label}`. Keep `BERNOULLI_API_KEYS` env fallback for one week post-migration (both sources OR'd together) — remove after verifying no accidental 401s.
- [ ] Add `just admin <subcommand>` recipe wrapper in root `justfile` — passes through to `uv run python -m bernoulli.admin ...` on the box via ssh. Document in the recipe doc comment that revoke/issue takes effect on next `serve-stop` + `serve-start`.
- [ ] Deploy to box: run the schema migration once (first `just admin list` triggers it), verify legacy env keys all work, verify a freshly `issue`d key works after `serve-stop` + `serve-start`, then remove `BERNOULLI_API_KEYS` from `~/.bernoulli.env` so there's a single source of truth.
- [ ] Add nightly backup cron on the box: `0 2 * * * rsync -a ~/.bernoulli-keys.db ssd2:bernoulli-key-backups/$(date +\%F).db` with a trailing find+delete for anything older than 30 days on the ssd2 side. Document in the "API key management" CLAUDE.md section.
- [ ] Add a new "API key management" section to `CLAUDE.md`: how to issue/list/revoke/rotate, where the DB lives, the "restart to apply" caveat, backup pattern, and the lost-DB recovery flow (scp the latest ssd2 backup back + restart).
- [ ] Update M15's planning note in `milestones.md`: `api_key_id` field in events now comes from the labelled `id` column directly (not a hash prefix). Blocking-but-not-strictly: M15 can still land independently with hash prefixes and backfill once M16 is live, but the natural order is M16 first.

**Exit criteria**:
- `just admin issue --label test` → paste the printed key into a curl `-H "Authorization: Bearer bnl_live_..."` → `/v1/decide` returns 200 after `serve-stop` + `serve-start`.
- `just admin revoke <id>` → `serve-stop` + `serve-start` → the same curl returns 401.
- `just admin list` shows all active + revoked keys with labels; no key material leaks (only hash prefixes if anything).
- Legacy `BERNOULLI_API_KEYS` entries have been migrated and the env var is gone from `~/.bernoulli.env`.
- Nightly backup lands on ssd2 at `bernoulli-key-backups/YYYY-MM-DD.db` and the 30-day retention sweep works.

**Dependencies:**
- **Blocks on:** nothing structural.
- **Blocks:** M15 benefits from this landing first (events get real labels instead of hash prefixes), but doesn't strictly require it — M15 can ship with hash prefixes and re-key when M16 lands.
- **Does not block:** anything else.

**Expected effort**: ~1 day for schema + CLI + middleware swap + migration script. +0.5 day if we add the HTTP admin endpoint vs. the simpler signal-based reload.

---

## M17 — Split OSS from private surface (open-core structure)
**Status**: 🔄 in progress (planned 2026-10-29 — Decisions + Tasks below)
**Goal**: Separate the public open-source repo (library + API + benchmarks + public docs) from a new private repo for the operational and commercial surface (website, marketing, planning docs, our specific Terraform, ops recipes, upcoming Chrome extension + Salesforce code). After the split, OSS stands alone as something anyone can clone and run; private depends on OSS via a pinned pip reference.

**Why now:**
Three forces pointing the same direction:
- M16 is next, and its artifacts split naturally across OSS (generic `bernoulli/admin.py`) and private (our deployment config). Doing M16 before M17 means re-splitting those artifacts right after writing them.
- The pile of commingled stuff only grows — Chrome extension, Salesforce plugin, more marketing pages, more admin tooling are all queued up.
- **No external users of the OSS repo yet.** Rewriting OSS history is free today; in six months it won't be.

Pattern we're adopting is standard open-core: Grafana, HashiCorp Vault, GitLab, Mattermost all do OSS repo + private repo depending on it.

**Decisions (resolved 2026-10-29):**
- **Private repo name**: `shyamsfo/bernoulli-platform` (confirm before creation; alternatives `bernoulli-ops`, `bernoulli-internal`). GitHub, private visibility, no license file, no README initially.
- **Dependency direction**: private → OSS via pip. Private's `pyproject.toml` pins `bernoulli @ git+https://github.com/shyamsfo/bernoulli.git@<sha>`. Bump the pin manually when OSS changes.
- **No submodules in that direction.** OSS keeps its one existing submodule (`benchmarks/jevbench/upstream` — external harness, unrelated).
- **History rewrite on OSS: aggressive.** `git filter-repo --invert-paths` to strip private directories entirely from OSS history. Tag `pre-m17-split` on main before the rewrite so rollback is one command.
- **Private history**: `git filter-repo --path <...>` on a fresh clone to extract moved paths *with* their history, so blame continuity survives. Private repo is initialized from that extraction, not from a fresh init.
- **M16 code location** (decided together with M17): `bernoulli/admin.py` → OSS (generic key lifecycle useful for any Bernoulli operator); `just admin` wrapper + backup cron + box-specific config → private.
- **What moves to private**:
  - `web/` (marketing site + tester + privacy page)
  - `product/` (milestones, parking-lot, reports, research, how-to)
  - `on-box/` (ops recipes)
  - `deploy/` (our specific Terraform + user_data + bringup.sh — contains our IPs, ssd2 integration specifics)
  - `demo/` (companion-app build brief; internal planning)
  - `samples/` (fixtures used by `serve-check`; ops-only)
  - Root `justfile` serve-* recipes (daily operations)
- **What stays OSS**:
  - `bernoulli/` (server code + future `bernoulli/admin.py`)
  - `benchmarks/` (public results + jevbench submodule)
  - `evals/` (public eval harness)
  - `tests/` (test suite)
  - `Dockerfile` (OSS deploy example)
  - Public docs: `README`, `API`, `HTTP`, `CLI`, `HOW-IT-WORKS`, `USAGE`, `USECASES`, `vision_and_roadmap`
  - `pyproject.toml`, `uv.lock`, `.python-version`, `LICENSE`
  - Root `justfile` (slimmed to default + sync; serve-* removed), `dev.just`, `bench.just`
- **`hf-space/`**: already a separate HF Spaces git repo (its own `.git`). Not tracked by OSS. Leave as-is — it continues to live alongside both repos as a convenient working directory.
- **Box deployment model after split**: bernoulli EC2 box clones the **private** repo. `uv sync` pulls OSS in as a pip dep. All serve-* recipes + admin tooling + deployment config live in private. No separate OSS checkout on the box.
- **CLAUDE.md surgery**: OSS CLAUDE.md loses the operational cheat sheet (depends on serve-* / deploy / terraform); private CLAUDE.md inherits that content and adds the two-repo workflow.

### Tasks

- [ ] Pre-flight audit of OSS history for leak risks. Grep for personal info, real API key material, customer names. Document findings even if clean. (Spot-checked 2026-10-29: nothing substantive — `~/.credentials` and `~/.bernoulli.env` are off-repo.)
- [ ] Confirm private repo name with user, then `gh repo create shyamsfo/<name> --private` (empty — no license, no README).
- [ ] Clone OSS repo into two fresh working dirs: `/tmp/bernoulli-oss-slim/` and `/tmp/bernoulli-platform/`. Install `git-filter-repo` locally if missing.
- [ ] On `/tmp/bernoulli-platform/`: `git filter-repo --path web/ --path product/ --path on-box/ --path deploy/ --path demo/ --path samples/` to retain only those paths plus their history. Add new top-level files: `pyproject.toml` depending on OSS at the pre-split SHA, `justfile` (with serve-* recipes extracted from OSS justfile), `CLAUDE.md` (private-surface guide), `README.md` (one-paragraph "private platform on top of github.com/shyamsfo/bernoulli"). First commit `main`, push to the new private repo.
- [ ] Tag OSS main as `pre-m17-split` and push the tag (safety net before the rewrite).
- [ ] On `/tmp/bernoulli-oss-slim/`: `git filter-repo --invert-paths --path web/ --path product/ --path on-box/ --path deploy/ --path demo/ --path samples/` to strip those paths from OSS history entirely. Trim root `justfile` to OSS-relevant recipes (keep `mod dev` + `mod bench`, drop serve-* + deploy-web). Update OSS `CLAUDE.md` to remove the ops cheat sheet. Update OSS `README.md` to remove private-ops references + add a one-line pointer to the private platform repo (bare name only, no link to the private URL since it's inaccessible).
- [ ] Verify OSS slim-down locally: `uv sync` works, `just dev test-fast` passes, imports clean.
- [ ] Force-push slim OSS to `shyamsfo/bernoulli` main. (`--force-with-lease` — not `--force` outright — against the `pre-m17-split` tag's SHA.)
- [ ] Verify private repo locally: `uv sync` resolves the pinned OSS, root-level `just` lists expected recipes.
- [ ] Update bernoulli EC2 box: clone private repo under `~/bernoulli-platform/` (or re-use `~/bernoulli/` path for continuity), `uv sync` to pull OSS as a dep. Verify `just serve-start` works from the new checkout. Remove the old OSS-only checkout once private is proven.
- [ ] Update `deploy/user_data.sh` (now in private) to clone the private repo on fresh instance bringup instead of the OSS repo.
- [ ] Update both CLAUDE.md files to reflect the new layout.
- [ ] Cross-link in both READMEs: OSS says "see our private platform repo (internal)" with no URL; private says "built on top of OSS Bernoulli" with URL.
- [ ] Session report capturing: pre-split SHA, filter-repo commands run, verification steps, box re-checkout, any surprises.

**Exit criteria**:
- `shyamsfo/bernoulli` contains only library + public docs + tests + benchmarks + evals. `git log --follow web/` on OSS returns empty (path doesn't exist in history).
- `shyamsfo/bernoulli-platform` (private) contains web + product + on-box + deploy + ops recipes + fixtures, and depends on OSS at a pinned SHA. `git log --follow web/index.html` on private shows preserved history.
- bernoulli EC2 box runs from the private repo clone; `just serve-up` + `just serve-start` + `just serve-check` all work.
- `/healthz` on `https://bernoulli-api.nuwire.dev` still returns 200 — no customer-visible break.
- The `pre-m17-split` tag exists on OSS and points to the pre-rewrite SHA (rollback path).

**Dependencies:**
- **Blocks on**: user confirmation of private repo name before `gh repo create`.
- **Blocks**: M16's execution. M15 is unaffected (defined only; will revisit after M16).
- **Does not block**: anything else.

**Expected effort**: ~0.5-1 day end-to-end. Dominated by: (a) the two filter-repo passes, (b) verifying both repos still build + test on both laptop and box, (c) the box re-checkout. Low risk because `pre-m17-split` tag lets us roll back cleanly.

---

## Notes

- Milestones are sequential by dependency, not by calendar priority.
- Each milestone must pass its exit criteria before the next begins.
- **Phase gate protocol** (from §10 of the vision doc): stop at each milestone's exit criteria, write a short gate report (what was built, metrics, deviations), and wait for go-ahead before starting the next milestone.
- Ask before downloading any model over 20GB, and before starting any training run.
- Pin dependency versions. Record exact model revision hashes in eval reports.
- If a milestone grows enough to need a real spec (PRD/PLAN with design decisions worth preserving), run `/ds-work-graduate` to promote the project to full mode.
