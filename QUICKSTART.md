# Bernoulli — Quickstart

Three paths, in order of ambition:

1. **Mac / CPU — read & poke.** Clone, run tests, instantiate the types, call the server with a tiny backbone. No GPU required.
2. **GPU laptop — real inference.** Serve the default 7B model and hit it with curl.
3. **GPU cloud — public URL.** See [`DEPLOY.md`](DEPLOY.md).

For the technique and architecture, read [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md). For the HTTP shape, [`HTTP.md`](HTTP.md). This doc is just the "get it running" surface.

---

## 1. Mac / CPU — what's actually possible

### Reality check

The default backbone (`Qwen/Qwen2.5-VL-7B-Instruct`) is ~15 GB in bf16. **It will not run on CPU in any useful sense, and it is tight on a 16 GB Mac even with MPS.** Alice-on-a-Mac should pick a smaller backbone — the Scorer is backbone-agnostic (see [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md) §7), so this is a two-env change.

| What you want | Recommended backbone | Memory | Notes |
|---|---|---|---|
| Read the code, run the CPU-only tests | — | — | No model load needed |
| Try the API end-to-end on a 16 GB Mac | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB fp16 | Fast on MPS; weak accuracy, fine for shape-testing |
| Try the API with better accuracy on a 32 GB+ Mac | `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-7B-Instruct` | ~6 / ~15 GB fp16 | MPS works; CPU is painfully slow |

Any backbone works as long as (a) letters A–I and the tokens "Yes"/"No" are single tokens in its tokenizer, and (b) it's a classical chat-tuned model (not a `<think>`-emitting reasoning model). Qwen 2.5, Llama 3, Mistral, Gemma 2, Phi 3 all satisfy this. The scorer asserts the single-token-labels invariant at startup and fails loudly if a backbone breaks it.

### Clone + sync

```bash
git clone https://github.com/shyamsfo/bernoulli.git
cd bernoulli
uv sync                       # installs everything including heavy torch/transformers deps
```

If `uv` isn't installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

### Run the tests

```bash
just dev test-fast            # ~5 s, CPU-only
```

~100 tests exercise the type system, prompt builder, debias logic, chunked scoring, decide dispatcher, calibration, and FastAPI server with a mock scorer. GPU-marked tests auto-skip when CUDA isn't visible.

### Instantiate the types (no model)

```python
from bernoulli.types import ChoiceQuestion, State, DecideRequest
req = DecideRequest(
    state=State(text="Where is my package?"),
    questions=[ChoiceQuestion(id="x", prompt="Intent?", options=["refund", "tracking"])],
)
print(req.model_dump_json(indent=2))
```

This touches nothing beyond pydantic — no torch, no model weights.

### Serve with a tiny backbone (MPS, 16 GB Mac)

```bash
# Point the scorer at a small, text-only model
export BERNOULLI_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
export BERNOULLI_MODEL_REVISION=""        # leave empty = latest, or pin a commit SHA from the HF model card
export BERNOULLI_DEVICE="mps"             # or "cpu" if MPS gives you trouble
export BERNOULLI_DTYPE="float16"          # bf16 is spotty on MPS; fp16 is safer
export BERNOULLI_MAX_MODEL_LEN=4096

just dev serve-local
```

First launch downloads the model to `~/.cache/huggingface` (~1 GB for 0.5B). Model load takes ~10–30 s on an M-series Mac.

In a second terminal:

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "I ordered a jacket two weeks ago and nothing has arrived."},
    "questions": [{"id": "intent", "type": "choice",
                   "prompt": "What does the customer want?",
                   "options": ["refund", "exchange", "tracking", "other"]}]
  }'
```

Expect a response in a few hundred ms on MPS, a few seconds on CPU. The *shape* of the response is identical to what the 7B/32B models return — only the accuracy of the probabilities changes.

### What you cannot do on Mac

- **Docker path.** The image is built on `vllm/vllm-openai`, which is Linux + CUDA only. `just dev docker-run` is a non-starter on Mac.
- **vLLM scorer.** Same reason — needs CUDA. Stays on `BERNOULLI_SCORER=hf`.
- **Match the published benchmark numbers.** Those were run on `Qwen2.5-VL-7B-Instruct` on a real GPU. Smaller backbones will produce worse-calibrated probabilities. Use the small-backbone path to validate the *integration*, not the *accuracy*.

---

## 2. GPU laptop — real inference locally

If you have an NVIDIA GPU (24 GB class, e.g., 3090 / 4090 / A5000):

```bash
git clone https://github.com/shyamsfo/bernoulli.git
cd bernoulli
uv sync

just dev model-pull           # ~15 GB download, 1–2 min with hf_transfer
just dev test                 # full suite including GPU tests, ~2.5 min warm
just dev serve-local          # defaults: Qwen2.5-VL-7B, cuda, bfloat16, 32k ctx
```

Then exercise it from `HTTP.md`:

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"state":{"text":"GET RICH QUICK!!!"}, "questions":[{"id":"c","type":"choice","prompt":"Category?","options":["safe","spam","abuse"]}]}'
```

First call includes a ~45 s model load. Subsequent calls are tens of ms.

### Or the production vLLM path, in Docker

Needs `nvidia-container-toolkit` installed on the host.

```bash
just dev docker-build         # ~5 min first time
just dev docker-run           # exposes :8000, mounts your HF cache
```

On a 24 GB card, set `BERNOULLI_MAX_MODEL_LEN=8192` (overridden in `docker-run` already); the default of 32k won't fit after the 7B weights load.

---

## 3. GPU cloud — public URL

See [`DEPLOY.md`](DEPLOY.md) for the full walkthrough: AWS instance recommendation, SSH-in checklist, Docker install, Caddy for automatic HTTPS, and enabling API-key authentication.

---

## What you cannot do with this repo alone

Lifted from [`AUDIT.md`](AUDIT.md):

- **No Terraform or `serve-up` recipes.** The operational wiring that runs `bernoulli-api.nuwire.dev` lives in a private sibling repo. [`DEPLOY.md`](DEPLOY.md) gives you a self-serve path that doesn't depend on it.
- **No marketing site / Chrome extension / Salesforce integration.** All private.
- **No PyPI package yet.** For now, `pip install git+https://github.com/shyamsfo/bernoulli.git@main` works — that's how the HF Space installs it.

What *is* here is fully functional: the library, the HTTP server, the CLI, the Dockerfile, the eval harness, the benchmarks, and the Gradio Space source under `hf-space/`.
