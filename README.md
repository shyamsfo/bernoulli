# Bernoulli

**An open-source clone of [Jev](https://typesafe.ai/), TypeSafe AI's System One decision model.**

Jev takes unstructured state (text or JSON) plus typed questions and returns calibrated probabilities — no generated prose, no JSON parsing, no retries. Bernoulli replicates this on open-weight VLMs (Qwen2.5-VL-7B for dev, swappable for production) and runs fully on-prem. One forward pass per question.

- Jev reference: [typesafe.ai](https://typesafe.ai/) · [System One concept docs](https://docs.typesafe.ai/concepts/system-one)
- Full brief: [`vision_and_roadmap.md`](vision_and_roadmap.md)
- Usage (CLI, config, examples): [`USAGE.md`](USAGE.md)
- Current milestone + task list: [`product/milestones.md`](product/milestones.md)
- Dev box (AWS g6.xlarge): [`deploy/README.md`](deploy/README.md)
- Session reports: [`product/reports/`](product/reports/)

**Status:** M3 (eval harness) in progress. SST-2 at 91.74% accuracy zero-shot — see [`evals/reports/sst2.md`](evals/reports/sst2.md).

## Workflow

### One-time setup

Provision the AWS g6.xlarge dev box and wire your local SSH config:

```bash
just up                                                        # terraform apply (deploy/) — launches g6.xlarge
terraform -chdir=deploy output -raw ssh_config_stanza >> ~/.ssh/config
ssh bernoulli 'sudo cloud-init status --wait && nvidia-smi'    # ~2 min cloud-init; verifies GPU
just sync-mirror                                               # push repo to the box
ssh bernoulli 'cd ~/bernoulli && uv sync && just model-pull'   # install deps + pull Qwen2.5-VL-7B (~90s via hf_transfer)
```

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

# run CI remotely
ssh bernoulli 'cd ~/bernoulli && just ci'

# or shell in
just shell
```

CI (lint + typecheck + pytest) runs on the dev box, not GHA — the interesting tests all need a GPU. See `product/milestones.md` M1 notes.

## Developer notes

Python 3.11+ managed with [uv](https://docs.astral.sh/uv/); pydantic v2 for the request/response surface; [transformers](https://huggingface.co/docs/transformers/) with the Qwen2.5-VL family for the dev backbone (vLLM for production, lands in M4). Infrastructure is a single AWS g6.xlarge (NVIDIA L4 24GB) provisioned via Terraform in [`deploy/`](deploy/). Tests use pytest with GPU-marked suites that auto-skip when CUDA isn't visible; linting via ruff; typechecking via mypy strict. Datasets flow through [`datasets`](https://huggingface.co/docs/datasets/) in the eval harness. Task runner is [`just`](https://just.systems/).

Project management (milestones, backlog, parking-lot, session reports under [`product/`](product/)) is driven by the [ds-work plugin](https://github.com/shyamsfo/ds-work-plugin).
