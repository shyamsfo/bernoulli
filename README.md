# Bernoulli

**An open-source clone of [Jev](https://typesafe.ai/), TypeSafe AI's System One decision model.**

Jev takes unstructured state (text or JSON) plus typed questions and returns calibrated probabilities — no generated prose, no JSON parsing, no retries. Bernoulli replicates this on open-weight VLMs (Qwen2.5-VL-7B for dev, swappable for production) and runs fully on-prem. One forward pass per question.

- Jev reference: [typesafe.ai](https://typesafe.ai/) · [System One concept docs](https://docs.typesafe.ai/concepts/system-one)
- Full brief: [`vision_and_roadmap.md`](vision_and_roadmap.md)
- Usage: [`HTTP.md`](HTTP.md) · [`CLI.md`](CLI.md) · landing: [`USAGE.md`](USAGE.md)
- **How it works (architecture + code walkthrough)**: [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md) — read this if you're an experienced Python/LLM dev who wants to understand the technique, the code flow, and what it takes to swap backbones.
- **Design pitch + realistic use cases**: [`USECASES.md`](USECASES.md)
- Current milestone + task list: [`product/milestones.md`](product/milestones.md)
- Dev box + model/hardware sizing table: [`deploy/README.md`](deploy/README.md)
- External benchmark comparisons (M8+): [`benchmarks/`](benchmarks/)
- Session reports: [`product/reports/`](product/reports/)

**Status.** M1–M3 complete. M4 (text-only production serving) is essentially done: vLLM backend, FastAPI server, Docker image, request batching, load test — only the step-up to the production backbone (M4e) is parked, pending `g6e.xlarge` capacity in `us-east-1`. Next up: M5 hardening (auth, metrics, offline mode, CI regression gate, v1.0 tag).

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

### One-time setup

Provision the AWS g5.xlarge dev box (NVIDIA A10G 24GB) and wire your local SSH config:

```bash
just up                                                        # terraform apply (deploy/) — launches g5.xlarge in us-east-1a
terraform -chdir=deploy output -raw ssh_config_stanza >> ~/.ssh/config
ssh bernoulli 'sudo cloud-init status --wait && nvidia-smi'    # ~2 min cloud-init; verifies GPU
just sync-mirror                                               # push repo to the box
ssh bernoulli 'cd ~/bernoulli && uv sync && just model-pull'   # install deps + pull Qwen2.5-VL-7B-Instruct (~15 GB, ~90 s via hf_transfer)
```

If `g5.xlarge` in `us-east-1a` hits `InsufficientInstanceCapacity`, pass `-var 'availability_zone=us-east-1b'` (or `1d`) to `terraform apply`. The `create` timeout is capped at 4 min so capacity failures surface fast.

Prereqs:
- AWS credentials configured (default profile, or `AWS_PROFILE=...`)
- `id_nuwire` keypair imported to AWS in `us-east-1` (nuw convention)
- `~/.ssh/id_nuwire` private key present locally
- NVIDIA GPU-Optimized AMI Marketplace subscription (one-time per AWS account — subscribe link in [`deploy/README.md`](deploy/README.md))
- `just` and `terraform >= 1.5` installed locally

See [`deploy/README.md`](deploy/README.md) for details, cost notes, stop/start caveats, and troubleshooting.

### Day-to-day

```bash
# push code to bernoulli
just sync-mirror

# run CI remotely (lint + typecheck + pytest; GPU tests run on bernoulli)
ssh bernoulli 'cd ~/bernoulli && just ci'

# fire an eval against a dataset
ssh bernoulli 'cd ~/bernoulli && just eval sst2 reverse'
just reports-pull                   # bring the sidecar + markdown back

# serve the HTTP API locally
ssh bernoulli 'cd ~/bernoulli && just serve'            # uvicorn on :8000
# or containerized (vLLM backend)
ssh bernoulli 'cd ~/bernoulli && just docker-build && just docker-run'

# load test the running server
ssh bernoulli 'cd ~/bernoulli && just loadtest'

# shell in
just shell
```

CI runs on the dev box, not GHA — the interesting tests all need a GPU. See `product/milestones.md` M1 notes for the rationale.

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
- **Model stack**: text-only today. [transformers](https://huggingface.co/docs/transformers/) + Qwen2.5-VL-7B for the HF path; vLLM 0.31 for the production path (`BERNOULLI_SCORER=vllm`). Image modality lands in M6.
- **Scoring technique**: single forward pass, read `logprobs` at the answer position restricted to label tokens (letters A–I for choice/rating, Yes/No for binary); reverse debias (2 passes) is the default. Chunked scoring handles >26-option questions (Banking77).
- **Infra**: single AWS g5.xlarge (NVIDIA A10G 24GB) in `us-east-1a`, provisioned via Terraform in [`deploy/`](deploy/). Docker data-root on the instance-store NVMe. Step up to L40S / g6e.xlarge (M4e) is parked on AWS capacity.
- **Tests**: pytest with GPU-marked suites that auto-skip when CUDA isn't visible. 102+ tests covering types, prompt builder, label tokens, debias, chunked scoring, decide dispatcher, generative baseline, calibrate, metrics, FastAPI server.
- **Lint + types**: ruff + mypy strict.
- **Eval datasets**: HF [`datasets`](https://huggingface.co/docs/datasets/) in the eval harness; four public sets wired (SST-2, AG News, BoolQ, Banking77).
- **Task runner**: [`just`](https://just.systems/).
- **Container**: Dockerfile uses `vllm/vllm-openai:v0.31.0-cu129-ubuntu2404` as the base.
- **Project management**: the [ds-work plugin](https://github.com/shyamsfo/ds-work-plugin) — milestones, backlog, parking-lot, and session reports all live under [`product/`](product/).

## License

The Bernoulli **source code** is released under the Apache License 2.0 — see [`LICENSE`](LICENSE). Copyright 2026 DeepStore.

The Apache 2.0 grant covers only the code in this repository. It does **not** cover:

- **Model weights.** Bernoulli loads an external VLM at runtime; the weights ship under their own license. For the default [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) that is Apache 2.0, but other sizes in the Qwen family (notably the 72B variants) use the Qwen License, which has commercial-use restrictions. Check the backbone's model card before swapping `BERNOULLI_MODEL_ID`.
- **Eval datasets.** The datasets wired into `evals/` (SST-2, AG News, BoolQ, Banking77) are pulled from Hugging Face at runtime and each has its own terms of use. Nothing is redistributed from this repo.
- **Container base images.** The Dockerfile builds on top of `vllm/vllm-openai` (Apache 2.0 upstream), which in turn pulls CUDA + PyTorch under NVIDIA / Meta's respective licenses. Review those before shipping a derived image.
