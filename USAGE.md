# Bernoulli — Usage

How to actually run decisions once the dev box is up and the model is cached. For setup, see [one-time setup in the README](README.md#one-time-setup).

## Interfaces

Two interfaces, same engine (`bernoulli.decide.decide`):

- **`bernoulli` CLI** — `uv run bernoulli decide --state X --question Y` (entry point `bernoulli.cli:main`). Good for scripts and ad-hoc runs.
- **HTTP API** — FastAPI server exposing `POST /v1/decide`, `GET /healthz`, `GET /v1/models`. Start with `just serve` (or `uv run uvicorn bernoulli.server:app`). Request/response shape matches [`vision_and_roadmap.md`](vision_and_roadmap.md) §3 exactly.

Switch the backend via `BERNOULLI_SCORER=hf` (default) or `BERNOULLI_SCORER=vllm` (production). Image state currently returns HTTP 501 — image path lands in M6.

### HTTP example

```bash
# start the server (loads the model at startup; takes ~2 min on cold boot)
just serve

# in another shell
curl -sS -X POST http://127.0.0.1:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "I want my money back now."},
    "questions": [
      {"id": "intent", "type": "choice",
       "prompt": "What does the customer want?",
       "options": ["refund", "exchange", "tracking", "other"]}
    ]
  }'
```

## The decision model

Bernoulli answers **typed questions about state**. State is text today (images land in M4). Questions come in three shapes:

- `choice` — pick one of 2–26 options
- `binary` — yes / no
- `rating` — integer scale with expected value

Every call returns calibrated-shape probabilities. No generated text, no JSON parsing, no retries.

## CLI

```bash
ssh bernoulli
cd ~/bernoulli
```

Two invocation forms.

### Form 1: `--state` + `--question` (single or batched question)

```bash
cat > /tmp/state.txt <<'EOF'
Customer email: Where is my package? I ordered it two weeks ago and nothing has
arrived. This is unacceptable service.
EOF

cat > /tmp/questions.json <<'EOF'
[
  {"id": "intent", "type": "choice",
   "prompt": "What does the customer want?",
   "options": ["refund", "exchange", "tracking", "other"]},
  {"id": "urgent", "type": "binary", "prompt": "Is this urgent?"},
  {"id": "anger",  "type": "rating", "prompt": "How angry?", "scale": [1, 5]}
]
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/questions.json
```

The `--question` file may be a single object or a JSON array.

### Form 2: `--request` with a full DecideRequest payload

```bash
cat > /tmp/request.json <<'EOF'
{
  "state": {"text": "Customer email: Where is my package? ..."},
  "questions": [
    {"id": "intent", "type": "choice", "prompt": "What does the customer want?",
     "options": ["refund", "exchange", "tracking", "other"]}
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
EOF

uv run bernoulli decide --request /tmp/request.json
```

## A real example

Input state:
> Customer email: Where is my package? I ordered it two weeks ago and nothing has arrived. This is unacceptable service.

Three questions: intent (choice), urgent (binary), anger (rating 1–5).

Response:

```json
{
  "decisions": {
    "intent": {
      "type": "choice",
      "answer": "tracking",
      "confidence": 0.9421,
      "distribution": {
        "refund":   0.0152,
        "exchange": 0.0012,
        "tracking": 0.9421,
        "other":    0.0414
      }
    },
    "urgent": {
      "type": "binary",
      "answer": true,
      "probability": 0.9466
    },
    "anger": {
      "type": "rating",
      "expected": 4.2956,
      "distribution": {
        "1": 0.0031, "2": 0.0336, "3": 0.0912, "4": 0.4088, "5": 0.4633
      }
    }
  },
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "calibration_version": null,
  "latency_ms": 1201
}
```

Notes on the shape:
- `rating` distribution keys are integer strings (`"1"`, `"2"`, ...). Internally the model picks a letter A–I (max scale width 9) and we map back to the integer labels — a workaround for Qwen's tokenizer treating `" 1"` as two tokens. The letter choice is invisible to callers.
- `calibration_version` is `null` until M3 ships calibrated temperatures.
- `latency_ms` covers all three questions × two forward passes each (`debias=reverse` default on an L4 24GB).

## Configuration

Env-driven via `BERNOULLI_*`:

| var | default | notes |
|---|---|---|
| `BERNOULLI_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | any transformers VLM whose tokenizer makes `" A"`…`" Z"` + `" Yes"`/`" No"` single-token |
| `BERNOULLI_MODEL_REVISION` | `cc594898...` | HF commit SHA; pin for reproducibility |
| `BERNOULLI_DTYPE` | `bfloat16` | `float16` / `float32` also valid |
| `BERNOULLI_DEVICE` | `cuda` | `cuda:0`, `cpu`, etc. |
| `BERNOULLI_MAX_MODEL_LEN` | `32768` | context limit; backbone supports up to 262k |
| `BERNOULLI_SCORER` | `hf` | `vllm` production backend lands in M4 |
| `BERNOULLI_DEFAULT_DEBIAS` | `reverse` | `none` / `reverse` / `cyclic` |

Precedence: CLI flags (e.g. `--model`) > env > `.env` file in the repo root > defaults in `bernoulli/config.py`.

## Debiasing

The model has a position bias — all else equal it slightly prefers the first option. We correct by rerunning the request with permuted orderings and averaging:

| mode | passes | when to use |
|---|---|---|
| `none`    | 1 | fastest; use when you don't need calibrated probabilities |
| `reverse` | 2 | **default.** catches most of the bias at 2× cost |
| `cyclic`  | k | thorough; k× cost; capped at k ≤ 6 options |

Pass via the request options:

```json
{"state": {...}, "questions": [...], "options": {"debias": "cyclic"}}
```

## Running evals

The eval harness scores the model against public datasets and emits a markdown + JSON report:

```bash
# single dataset end-to-end
uv run python -m evals.run_eval --dataset sst2 --debias reverse \
    --out evals/reports/sst2.md

# or via just
just eval sst2 reverse

# cap for a quick smoke
uv run python -m evals.run_eval --dataset sst2 --limit 10

# pull reports back to the local repo (run on your laptop)
just reports-pull
```

Datasets available today: `sst2`. More land in M3d/M3e. See `evals/reports/` for finished reports.

## Troubleshooting

- **First call after instance stop/start is slow.** The ephemeral NVMe is wiped on stop/start, so the HF cache is empty. First model load re-pulls weights (~90s). Subsequent calls use the cache.
- **CUDA OOM.** Qwen2.5-VL-7B needs ~15GB in bf16. On the L4 24GB there's headroom. If you move to a smaller GPU, lower `BERNOULLI_MAX_MODEL_LEN` or switch to `BERNOULLI_DTYPE=float16`.
- **"Image scoring lands in M4".** Passing images in the request currently raises `NotImplementedError`. Images + the full `AutoProcessor` path land in M4.
- **Multi-token label error.** If you swap in a backbone whose tokenizer breaks the single-token-label invariant (space-prefixed `A`–`Z`, `Yes`, `No`), `HFScorer` fails loudly at construction. Pick a different backbone or expand `bernoulli/labels.py` with a tokenizer-specific label set.
