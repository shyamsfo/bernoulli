# Bernoulli — Open-Source Multimodal Decision Model

> Working name: **Bernoulli**. A Jev-class "System One" decision model built on open Qwen3.6 weights.
> This document is the brief for Claude Code. Work phase by phase; stop at each phase gate and report.

---

## 1. Vision

LLMs generate text, and then software has to parse that text to make a decision. Bernoulli skips generation entirely:

**unstructured state (text + images) + typed questions → calibrated probabilities you can branch on.**

- One forward pass per question. No generated tokens, no parsing, no JSON repair.
- Probabilities are **calibrated**: when Bernoulli says 0.9, it is right ~90% of the time.
- **Multimodal**: text plus up to 8 images per state (screenshots, documents, photos).
- **Long context**: up to the backbone's native ~262k tokens.
- **Sovereign**: runs fully on-prem. No outbound network calls at inference time.

Reference points: TypeSafe AI's Jev (closed, API-only) and Juspay's XOR (Qwen3.6-35B-A3B based). XOR's pipeline is the one we follow: prompt → forward → last-token hidden state → lm_head logits over candidate tokens → forward + reversed option order → calibrate.

## 2. Goals and non-goals

**Goals**
- Typed decision API with three question types: `choice`, `binary`, `rating`.
- Zero-shot accuracy that is competitive with prompting a generative LLM, at a fraction of the latency.
- Calibration (ECE) measurably better than the raw model.
- Reproducible evaluation harness with regression gates.
- Swappable backbone (Qwen3.6-35B-A3B, Qwen3.6-27B, and a small dev model).

**Non-goals (v1)**
- Text generation, explanations, or rationales.
- Tool calling or agents. Bernoulli is a component that agents call.
- Pretraining or full fine-tuning. LoRA at most.

## 3. API specification

### Request — `POST /v1/decide`
```json
{
  "state": {
    "text": "Customer email body ...",
    "images": ["<base64 or file path>"]
  },
  "questions": [
    {"id": "intent", "type": "choice",
     "prompt": "What does the customer want?",
     "options": ["refund", "exchange", "tracking", "other"]},
    {"id": "urgent", "type": "binary",
     "prompt": "Is this message urgent?"},
    {"id": "anger", "type": "rating",
     "prompt": "How angry is the customer?", "scale": [1, 5]}
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

### Response
```json
{
  "decisions": {
    "intent": {"answer": "refund", "confidence": 0.91,
               "distribution": {"refund": 0.91, "exchange": 0.05, "tracking": 0.03, "other": 0.01}},
    "urgent": {"answer": true, "probability": 0.78},
    "anger":  {"expected": 3.6, "distribution": {"1": 0.02, "2": 0.08, "3": 0.25, "4": 0.48, "5": 0.17}}
  },
  "model": "qwen3.6-35b-a3b", "calibration_version": "2026-10-05",
  "latency_ms": 142
}
```

### Semantics
- `choice`: 2–26 options. Labels are internally mapped to letters A, B, C, … The response returns the user's option strings.
- `binary`: implemented as a 2-option choice (Yes/No). Debiased by swapping the order.
- `rating`: integer scale labels. Returns the full distribution plus the expected value.
- Unknown or invalid input returns 422 with a pydantic error. The service never silently guesses.

## 4. Architecture

```
request ─► Prompt Builder ─► Backbone forward (Qwen3.6, VLM processor)
                                  │
                                  ▼
                       last-position logits
                                  │
              restrict to label token ids ─► softmax(logits / T)
                                  │
            Debias: rerun with permuted option order, map back, average
                                  │
                       Calibrator (per question type)
                                  │
                                  ▼
                         typed Decision objects
```

### Components
1. **Prompt builder** (`bernoulli/prompt.py`)
   - Uses the Qwen chat template. The system message reads: "You are a decision function. Answer with a single letter."
   - The user turn contains the images, then the state text, then the question, then the labeled options.
   - The assistant turn is pre-filled with `Answer:` so the next token is the label.
   - The state goes first and the question last, so multiple questions over one state share a cached prefix.
2. **Label tokens** (`bernoulli/labels.py`)
   - At startup, assert that every label (`" A"`…`" Z"`, `" 1"`…`" 9"`, `" Yes"`/`" No"`) encodes to exactly one token. Fail loudly if not.
3. **Scorer** (`bernoulli/scorer.py`)
   - Two backends behind one interface:
     - `HFScorer`: transformers, for dev and tests.
     - `VLLMScorer`: production. Uses `max_tokens=1`, `logprobs=20`, `allowed_token_ids=label_ids`, and prefix caching enabled.
   - Returns raw logits over the label set.
4. **Debiaser** (`bernoulli/debias.py`)
   - Modes: `none`, `reverse` (default, as in XOR), and `cyclic` (all k cyclic shifts, for k ≤ 6).
   - Averages the probabilities after mapping each run back to the canonical option order.
5. **Calibrator** (`bernoulli/calibrate.py`)
   - Temperature scaling, with one T per question type. Fit with LBFGS on NLL over a held-out set.
   - Stored in `calibration/<model>.json` and versioned.
   - Reports ECE (15 bins), Brier score, and NLL before and after calibration.
6. **Server** (`bernoulli/server.py`)
   - FastAPI with pydantic request/response models.
   - Endpoints: `/v1/decide`, `/healthz`, `/v1/models`.
   - Batches all questions × permutations for one request into a single engine call.

## 5. Tech stack
- Python 3.11+, `uv` for environments, `ruff` + `mypy`, `pytest`.
- PyTorch, `transformers`, `vllm`, `peft` (Phase 4), `datasets`.
- FastAPI + uvicorn, pydantic v2.
- Optional Phase 5: a small React + TypeScript (Vite) playground UI.
- Docker for serving.

## 6. Repo layout
```
bernoulli/
  bernoulli/
    __init__.py
    types.py          # pydantic: Question, State, Decision
    prompt.py
    labels.py
    scorer.py         # HFScorer, VLLMScorer
    debias.py
    calibrate.py
    server.py
    config.py         # model id, max_len, dtype, device
  evals/
    datasets/         # loaders → common (state, question, options, gold) format
    run_eval.py       # accuracy, macro-F1, ECE, Brier, latency p50/p95
    baselines.py      # generative-prompt baseline using the same backbone
    reports/
  training/           # Phase 4
    build_sft_mix.py
    train_lora.py
  calibration/
  tests/
  playground/         # Phase 5 (React)
  CLAUDE.md
  pyproject.toml
  Dockerfile
```

## 7. Roadmap

Each phase ends with a **gate**. Claude Code stops at the gate, reports the results, and waits for go-ahead.

### Phase 0 — Scaffold (½ day)
- Set up the repo, `pyproject`, lint/type/test tooling, `CLAUDE.md`, and the config system.
- Write the pydantic types for the full API spec in §3.
- **Dev backbone:** the smallest available Qwen3.5/3.6 VL checkpoint, so iteration runs on a single consumer GPU or CPU.
- **Gate:** `pytest` passes and type checks are clean.

### Phase 1 — Zero-shot scorer (1–2 days)
- Prompt builder, label-token check, and `HFScorer` for text-only input.
- Implement `choice`, `binary`, and `rating`.
- Unit tests: deterministic outputs, distributions sum to 1, option-string mapping is correct, invalid input is rejected.
- **Gate:** a CLI `bernoulli decide --state file.txt --question q.json` works on the dev model.

### Phase 2 — Debias, calibration, eval harness (2–3 days)
- Implement the `reverse` and `cyclic` debiasers.
- Build the eval harness on public sets:
  - Text: SST-2, AG News, Banking77 (77-way intent; test chunked option sets), BoolQ.
  - Multimodal: a ScienceQA image subset and a small image-classification set (e.g. Food-101 sample).
- Add a generative baseline: the same backbone prompted to answer with a single word, then parsed.
- Fit temperatures and produce a report (markdown + plots) in `evals/reports/`.
- **Gate:** a report comparing raw vs. debiased vs. calibrated vs. generative baseline on accuracy, macro-F1, ECE, and latency.

### Phase 3 — Multimodal + production serving (2–3 days)
- Image support via the Qwen VL processor, up to 8 images per state.
- `VLLMScorer` with prefix caching and a configurable `max_model_len` (default 32k; documented path to 262k).
- FastAPI server, plus multi-question batching against a shared state.
- Load test: p50/p95 latency for 1, 5, and 20 questions per state.
- Switch to the production backbone: **Qwen3.6-35B-A3B** (MoE, ~3B active), with a config flag for **Qwen3.6-27B** dense.
- **Gate:** the server runs in Docker, the full eval suite runs through the HTTP API, and latency numbers are reported.

### Phase 4 — LoRA fine-tune (optional, 3–5 days)
- Build an SFT mix from the Phase 2 datasets' train splits, with randomized option order and randomized label letters.
- Train only on the single answer token: loss on that position, masked everywhere else.
- Use LoRA on the attention/FFN projections and keep the vision tower frozen.
- Re-run the evals and re-fit calibration.
- **Gate:** the fine-tune beats zero-shot on held-out tasks **not** seen in training. If it only wins on in-distribution tasks, it is overfitting and does not ship.

### Phase 5 — Hardening (2–3 days)
- API-key auth, structured logging that never logs state content by default, and a Prometheus metrics endpoint.
- Explicit offline mode: `HF_HUB_OFFLINE=1`, and models load from a local path.
- CI eval regression gate: fail if accuracy drops more than 1pt or ECE rises more than 0.01 against the stored baseline.
- Optional React playground: paste state, add images, define questions, see the distributions as bars.
- **Gate:** v1.0 tag.

## 8. Hardware notes
- **Dev:** small model, any 16–24GB GPU or CPU.
- **Qwen3.6-35B-A3B:** about 70GB of weights in bf16, so it needs a single 80GB GPU or 2×48GB. Quantized builds run in about 22GB, but quantization must be re-evaluated for calibration drift.
- **Qwen3.6-27B dense:** about 54GB in bf16.
- **Long context:** the KV cache dominates. 262k tokens needs a large-memory setup, so default to 32k and document the scaling.

## 9. Risks and open questions
- **Position and label bias:** debiasing costs 2× (or k×) compute. Measure whether `reverse` alone is enough.
- **Many options** (e.g. 77 intents): the letter alphabet runs out and long option lists hurt accuracy. Plan: hierarchical or chunked tournament scoring. Evaluate on Banking77.
- **Thinking mode:** Qwen3.6 is hybrid-reasoning. The scorer must disable thinking in the chat template; verify the next token is the label and not a `<think>` tag.
- **Calibration transfer:** a temperature fitted on public sets may not transfer to a user's domain. Expose a `bernoulli calibrate --data my_labeled.jsonl` command.
- **MoE + vLLM:** confirm that `allowed_token_ids` and prefix caching work with the Qwen3.6 MoE and the VL path in the pinned vLLM version.

## 10. Working agreement for Claude Code
- Execute one phase at a time. Stop at each gate with a short report covering what was built, the metrics, and any deviations.
- Ask before downloading any model over 20GB, and before starting any training run.
- Keep `CLAUDE.md` current: commands, config, decisions made, and gotchas found.
- Every module gets tests. Eval numbers come from scripts, never from hand-typed results.
- Pin dependency versions. Record the exact model revision hashes in the eval reports.
- When something in this document turns out to be wrong (a model id, an API behavior), fix it, note it in `CLAUDE.md`, and flag it in the gate report.
