# Bernoulli

**An open-source clone of [Jev](https://typesafe.ai/), TypeSafe AI's System One decision model.**

**Live:** [bernoulli.live](https://bernoulli.live) — landing page with the pitch, benchmarks, and architecture overview.

Jev takes unstructured state (text or JSON) plus typed questions and returns calibrated probabilities — no generated prose, no JSON parsing, no retries. Bernoulli replicates this on open-weight VLMs (Qwen2.5-VL-7B for dev, swappable for production) and runs fully on-prem. One forward pass per question.

- Jev reference: [typesafe.ai](https://typesafe.ai/) · [System One concept docs](https://docs.typesafe.ai/concepts/system-one)
- Full brief: [`vision_and_roadmap.md`](vision_and_roadmap.md)
- **Run it yourself**: [`QUICKSTART.md`](QUICKSTART.md) (Mac / GPU laptop) · [`DEPLOY.md`](DEPLOY.md) (public URL on AWS GPU, with auth)
- Usage: [`HTTP.md`](HTTP.md) · [`CLI.md`](CLI.md) · landing: [`USAGE.md`](USAGE.md)
- **How it works (architecture + code walkthrough)**: [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md) — read this if you're an experienced Python/LLM dev who wants to understand the technique, the code flow, and what it takes to swap backbones.
- **Design pitch + realistic use cases**: [`USECASES.md`](USECASES.md)
- **HTTP API reference**: [`API.md`](API.md)
- External benchmark comparisons: [`benchmarks/`](benchmarks/)

**Status.** Benchmarks for the foundation + academic suite closed 2026-10-15 with 6 academic benchmarks × 4 baselines at N=500 — see [`benchmarks/README.md`](benchmarks/README.md) and the live summary at [bernoulli.live/benchmarks](https://www.bernoulli.live/benchmarks.html). Use-case coverage closed except for WildGuardTest (gated on HF auth). JevBench adapter + leaderboard submission done 2026-10-10 — Benchmark Heaven FIFO queue, position #62. In-house numbers for the full 231-task public cohort: **7B weighted 0.706** ([results](benchmarks/jevbench/results/2026-10-08.md)); **32B weighted 0.801** on a one-shot Lambda H100 rental (+9.5 pp overall, +18 pp on the hard tier).

Numbers so far (all zero-shot on Qwen2.5-VL-7B):

| | accuracy | ECE | NLL | reference |
|---|---|---|---|---|
| SST-2 (binary sentiment) | 0.9174 | 0.0280 | 0.2458 | [`evals/reports/sst2.md`](evals/reports/sst2.md) |
| …calibrated (T=1.35) | 0.9174 | 0.0253 | 0.2338 | [`evals/reports/sst2.calibrated.md`](evals/reports/sst2.calibrated.md) |
| …generative baseline | 0.9174 | 0.0826 | 2.2815 | [`evals/reports/sst2.generative.md`](evals/reports/sst2.generative.md) |
| AG News (4-way topic) | 0.8479 | 0.0995 | 0.6065 | [`evals/reports/ag_news.md`](evals/reports/ag_news.md) |
| BoolQ (binary passage) | 0.6318 | 0.0999 | 0.6652 | [`evals/reports/boolq.md`](evals/reports/boolq.md) |
| Banking77 (77-way intent, chunked) | 0.5822 | 0.1272 | 1.6178 | [`evals/reports/banking77.md`](evals/reports/banking77.md) |

HTTP serving latency on the containerized vLLM path (A10G 24GB, reverse debias):

| questions/req | p50 | p95 | p99 |
|---|---|---|---|
| 1  | 76 ms   | 76 ms   | 77 ms   |
| 5  | 368 ms  | 369 ms  | 370 ms  |
| 20 | 1466 ms | 1469 ms | 1470 ms |

See [`evals/reports/loadtest.md`](evals/reports/loadtest.md).

## Workflow

Clone + sync + run tests locally:

```bash
git clone https://github.com/shyamsfo/bernoulli.git
cd bernoulli
uv sync
just dev test-fast         # CPU-only tests, ~5 s
just dev ci                # lint + typecheck + full pytest (GPU tests auto-skip without CUDA)
```

Serve the HTTP API locally for integration work:

```bash
just dev serve-local       # uvicorn on 127.0.0.1:8000, HFScorer path (loads model into the process)
```

The vLLM-backed Docker image is the production path:

```bash
just dev docker-build && just dev docker-run
```

The real tests need a GPU; GHA free runners can't run them, so CI gates nothing meaningful via GHA. Run `just dev ci` on your own GPU box before merging.

Deployment-side recipes (serve-up / serve-start for a cloud-hosted instance, website publishing, etc.) are not in this repo — they live in a sibling `bernoulli-ops` repo and depend on this one as a pip package.

### Serving in Docker

The Docker image (`vllm/vllm-openai` base + a thin bernoulli layer) ships the vLLM backend. On the dev box, Docker's data-root must live on the NVMe (`/opt/dlami/nvme/docker`) — the vLLM image + layers won't fit on the 80 GB EBS root. This is already configured on `bernoulli` by cloud-init / manual setup; see the Gotchas section in [`CLAUDE.md`](CLAUDE.md). Mount the host HF cache so the container doesn't re-download the model:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v $HOME/.cache/huggingface:/data/hf-cache \
  -e BERNOULLI_SCORER=vllm \
  -e BERNOULLI_MAX_MODEL_LEN=8192 \
  bernoulli:latest
```

## Developer notes

- **Language + env**: Python 3.11+ managed with [uv](https://docs.astral.sh/uv/); `uv.lock` is committed.
- **API surface**: pydantic v2 (discriminated-union Question + Decision, strict `extra='forbid'`), served via FastAPI at `/v1/decide`, `/v1/models`, `/healthz`.
- **Model stack**: text-only today. [transformers](https://huggingface.co/docs/transformers/) + Qwen2.5-VL-7B for the HF path; vLLM for the production path (`BERNOULLI_SCORER=vllm`). Image modality is on the roadmap.
- **Scoring technique**: single forward pass, read `logprobs` at the answer position restricted to label tokens (letters A–I for choice/rating, Yes/No for binary); reverse debias (2 passes) is the default. Chunked scoring handles >26-option questions.
- **Tests**: pytest with GPU-marked suites that auto-skip when CUDA isn't visible. 102+ tests covering types, prompt builder, label tokens, debias, chunked scoring, decide dispatcher, generative baseline, calibrate, metrics, FastAPI server.
- **Lint + types**: ruff + mypy strict.
- **Eval datasets**: HF [`datasets`](https://huggingface.co/docs/datasets/) in the eval harness; four public sets wired (SST-2, AG News, BoolQ, Banking77).
- **Task runner**: [`just`](https://just.systems/).
- **Container**: Dockerfile uses `vllm/vllm-openai` as the base.

## License

The Bernoulli **source code** is released under the Apache License 2.0 — see [`LICENSE`](LICENSE). Copyright 2026 DeepStore.

The Apache 2.0 grant covers only the code in this repository. It does **not** cover:

- **Model weights.** Bernoulli loads an external VLM at runtime; the weights ship under their own license. For the default [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) that is Apache 2.0, but other sizes in the Qwen family (notably the 72B variants) use the Qwen License, which has commercial-use restrictions. Check the backbone's model card before swapping `BERNOULLI_MODEL_ID`.
- **Eval datasets.** The datasets wired into `evals/` (SST-2, AG News, BoolQ, Banking77) are pulled from Hugging Face at runtime and each has its own terms of use. Nothing is redistributed from this repo.
- **Container base images.** The Dockerfile builds on top of `vllm/vllm-openai` (Apache 2.0 upstream), which in turn pulls CUDA + PyTorch under NVIDIA / Meta's respective licenses. Review those before shipping a derived image.
