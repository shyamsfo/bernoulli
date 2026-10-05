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
**Status**: 🔄 in progress
**Goal**: A shippable text-only service: `VLLMScorer` with prefix caching + a FastAPI server + the production backbone + Docker + load numbers. Image path explicitly deferred to M6.

- [ ] Pick + load the production backbone (step up from the 7B dev model). Candidates: Qwen3.6-27B dense (~54 GB bf16), Qwen3.6-35B-A3B MoE (~70 GB bf16, ~3B active), or another open ~20–40B model. Instance resize decision alongside — probably `g6e.xlarge` (L40S 48GB) or an 80GB-class box. Decision + rationale in `CLAUDE.md`.
- [ ] `VLLMScorer` on the chosen backbone: `max_tokens=1`, `logprobs=20`, `allowed_token_ids=label_ids`, prefix caching on. Verify all three features work together on the pinned vLLM version.
- [ ] FastAPI server exposing `/v1/decide`, `/healthz`, `/v1/models`. Thin wrapper around `decide()` and `generative_decide()`; honors `BERNOULLI_*` env config.
- [ ] Batch all `questions × permutations` for a request into a single engine call (shared state prefix).
- [ ] Dockerfile for serving — multi-stage, slim runtime, vLLM + the chosen backbone cached at `/opt/models/`.
- [ ] Load test: p50 / p95 latency for 1, 5, 20 questions per state at steady state. Report in `evals/reports/`.
- [ ] Flip the eval harness to score via HTTP against the running server (replaces direct scorer wiring with an HTTP client).

**Exit criteria**: Server runs in Docker, the full M3 eval suite passes end-to-end against the HTTP API, p50/p95 latency table reported.

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

## Notes

- Milestones are sequential by dependency, not by calendar priority.
- Each milestone must pass its exit criteria before the next begins.
- **Phase gate protocol** (from §10 of the vision doc): stop at each milestone's exit criteria, write a short gate report (what was built, metrics, deviations), and wait for go-ahead before starting the next milestone.
- Ask before downloading any model over 20GB, and before starting any training run.
- Pin dependency versions. Record exact model revision hashes in eval reports.
- If a milestone grows enough to need a real spec (PRD/PLAN with design decisions worth preserving), run `/ds-work-graduate` to promote the project to full mode.
