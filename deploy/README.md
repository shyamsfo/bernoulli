# deploy/ — Bernoulli dev box

Terraform for a single AWS GPU box used as an rsync target for dev + test.

**Current default:** `g5.xlarge` (NVIDIA A10G 24 GB) in `us-east-1a`, running `Qwen/Qwen2.5-VL-7B-Instruct`. See [Picking a different model + machine](#picking-a-different-model--machine) below for other combinations.

## Layout

```
deploy/
├── main.tf         # VPC/SG/AMI data lookups + EC2 instance
├── variables.tf    # region, instance_type, keypair, etc.
├── outputs.tf      # public IP, SSH command, ssh-config stanza
├── user_data.sh    # cloud-init: uv, HF cache on NVMe, env
├── sync.sh         # rsync wrapper
└── .gitignore      # terraform state + plan files
```

**No Python CLI.** For a single long-lived dev box, pure Terraform is simpler than the Click-CLI pattern in `~/nuw/inference/kube-do/nvidia-launcher/`. If this project ever needs multiple instances / spot fallback / per-model presets, graduate to that pattern.

## What this is NOT

- Not production serving — that lives in M4 (`VLLMScorer` + FastAPI in Docker).
- Not auto-shutdown — the instance stays up until you `aws ec2 stop-instances` or `terraform destroy` it.
- Not a shared box — SG is scoped to the caller's /32 at apply time.

## Prerequisites

- Terraform ≥ 1.6
- `awscli` configured, or `AWS_PROFILE=<profile>` set
- `id_nuwire` keypair already imported into AWS in `us-east-1` (nuw already did this — nvidia-launcher uses the same keypair)
- `~/.ssh/id_nuwire` present locally (already is, per `~/.ssh/config`)

## Up

```bash
cd deploy
terraform init
terraform plan      # inspect first
terraform apply     # ~90s to running state, then ~2 min of cloud-init
```

After apply, grab the ssh stanza and paste into `~/.ssh/config`:

```bash
terraform output -raw ssh_config_stanza >> ~/.ssh/config
```

Then wait for cloud-init:

```bash
ssh bernoulli 'sudo cloud-init status --wait && nvidia-smi'
```

## Push code and run tests

```bash
./deploy/sync.sh                # rsync, no delete
./deploy/sync.sh --mirror       # mirror (deletes remote files)

ssh bernoulli 'cd /home/ubuntu/bernoulli && uv sync && uv run pytest'
```

## Pause (save money without destroying)

```bash
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)
```

**NOTE:** Public IP changes on stop/start. After `start`, re-paste the ssh stanza:

```bash
terraform refresh && terraform output -raw ssh_config_stanza
```

Also: the instance-store NVMe is wiped on stop/start, so HF cache is lost. First run after start re-pulls model weights (~9 min for Qwen2.5-VL-7B).

## Down

```bash
terraform destroy
```

## HF token

HF Hub access is set via env on the box:

```bash
ssh bernoulli
export HF_TOKEN=hf_...
# add to ~/.bashrc or ~/.hf-token if you want it persistent
```

For most Qwen VL models no token is needed (they're public, non-gated). Add one only if we move to a gated model.

## Cost

- g5.xlarge on-demand: ~$1.01/hr → ~$727/mo if left on 24/7
- Stopped instance: just EBS root (80 GB gp3) ≈ $6.40/mo
- Rough dev usage estimate: 4 hr/day × 20 days/mo = ~$64/mo

## Troubleshooting

- cloud-init log: `ssh bernoulli 'sudo cat /var/log/bernoulli-cloud-init.log'`
- GPU check: `ssh bernoulli 'nvidia-smi'`
- NVMe mount: `ssh bernoulli 'df -h /opt/nvme'`
- The NVIDIA GPU-Optimized AMI must be subscribed-to once per AWS account via the AWS Marketplace: https://aws.amazon.com/marketplace/pp?sku=7wvc601jyn35n5w8b3kro66vi (already done if nvidia-launcher has run in this account).

---

## Picking a different model + machine

The current default (`g5.xlarge` + 7B) is a dev-tier config. For production or a different quality/cost point, pick a backbone and change `instance_type` in `variables.tf` (or pass `-var 'instance_type=...'` to `terraform apply`) + override `BERNOULLI_MODEL_ID` / `BERNOULLI_MODEL_REVISION` for the service.

| Backbone                                 | Weights         | Min VRAM      | AWS instance                      | ~$/hr   | When to pick                                                                 |
|------------------------------------------|-----------------|---------------|-----------------------------------|---------|------------------------------------------------------------------------------|
| `Qwen2.5-VL-3B-Instruct`                 | ~7 GB (bf16)    | 12 GB         | `g4dn.xlarge` (T4 16 GB), or local Mac | ~$0.53  | Local smoke, cheapest cloud iteration. Quality is a notable step down.      |
| **`Qwen2.5-VL-7B-Instruct` (current)**   | **~15 GB (bf16)** | **24 GB**   | **`g5.xlarge` (A10G 24 GB)**      | **~$1.01** | **Dev + light production. What this repo is tested on.**                   |
| `Qwen2.5-VL-32B-Instruct-AWQ`            | ~20 GB (int4)   | 48 GB         | `g6e.xlarge` (L40S 48 GB)         | ~$1.86  | Mid-tier production. L40S has native int4/int8 math. Target for M4e.         |
| `Qwen2.5-VL-72B-Instruct-AWQ`            | ~40 GB (int4)   | 48 GB         | `g6e.2xlarge` (L40S 48 GB, more CPU/RAM) | ~$2.24  | Big production, quality-first. Fits snug — may need `max_model_len` tuning. |
| `Qwen2.5-VL-72B-Instruct` (bf16)         | ~140 GB (bf16)  | 2× 80 GB or 4× 48 GB | `g6e.12xlarge` (4× L40S)   | ~$10.49 | Rare. Only when quantization-induced calibration drift is unacceptable.      |

Notes:

- Prices are `us-east-1` on-demand. Spot is 60–70% cheaper for the g-family; reservations are much cheaper over 1–3 years.
- vLLM on 24 GB cards needs `BERNOULLI_MAX_MODEL_LEN=8192` — the 32768 default doesn't leave room for the KV cache alongside 7B weights. On 48 GB+ you can raise it toward the backbone's native 262k.
- `g6` and `g6e` capacity in `us-east-1` has been intermittent; the current pin to `g5.xlarge` is the result of hitting `InsufficientInstanceCapacity` on both across every supported AZ. Try `-var 'availability_zone=us-east-1b'` (or `1d`) if the default also runs dry.
- The table above sticks to the Qwen2.5-VL family because that's what the scorer + chat template are tested with. Any other transformers-compatible VLM should work — the single-token-label check in `bernoulli/labels.py` catches tokenizer surprises at `HFScorer` construction, so a bad swap fails loudly rather than silently.
- Memory numbers are approximate — exact footprint depends on KV cache sizing, prefix caching, and your tolerance for max context length. Measure, don't guess.

---

## CPU-only / Graviton deployment (llama.cpp)

**Short answer:** possible for low volume, not currently wired up. The technique Bernoulli uses — one forward pass per question, read logits at the answer position restricted to label tokens — fits llama.cpp cleanly, since `llama.cpp` exposes per-token logprobs over the full vocab natively. There is no fundamental blocker beyond writing a `LlamaCppScorer` that implements the same `Scorer` protocol as `HFScorer` / `VLLMScorer`. Debias, chunked scoring, and calibration all sit above that protocol and would work unchanged.

**What exists today:** nothing CPU-specific. `BERNOULLI_SCORER` has `hf` and `vllm`; both need CUDA. A `llamacpp` scorer would go in `bernoulli/llamacpp_scorer.py`, probably via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) with `logits_all=True`, slicing the final-position logits by the single-token label ids from `bernoulli/labels.py`.

**Rough latency expectations** (Qwen2.5-VL-7B as Q4_K_M GGUF, ~4.5 GB RAM, ~300-token prompt + 1 output token, reverse debias = 2 passes):

| Host                                      | ~$/hr    | Per-question latency (est.) | vs A10G reference |
|-------------------------------------------|----------|-----------------------------|-------------------|
| A10G 24 GB (`g5.xlarge`, current GPU ref) | ~$1.01   | ~75 ms                      | 1×                |
| Graviton3 `c7g.4xlarge` (16 vCPU, SVE)    | ~$0.58   | ~3–6 s                      | ~50×              |
| Graviton4 `c8g.4xlarge` (16 vCPU, SVE2)   | ~$0.64   | ~2–5 s                      | ~40×              |
| Intel `c7i.4xlarge` (16 vCPU, AVX-512)    | ~$0.71   | ~3–5 s                      | ~50×              |
| M3 Max / M4 Pro Mac (Metal)               | laptop   | ~1–2 s                      | ~20×              |

Numbers are back-of-envelope from published llama.cpp benchmarks on 7B Q4_K_M; measure before quoting them.

**When CPU makes sense:**
- Air-gapped / on-prem environments where no GPU is available.
- Hobbyist or demo use at low volume (hundreds of decisions/day, not thousands/minute).
- Local dev on an Apple Silicon Mac — Metal-backed llama.cpp is a genuinely useful fast loop.

**When it does not:**
- Any latency SLO under ~1 s. The GPU path is 20–50× faster per question.
- High-concurrency serving — llama.cpp batching is primitive next to vLLM's continuous batching + prefix caching. A single GPU box will out-serve a rack of Gravitons.
- The multimodal path (M6+). llama.cpp has partial Qwen2.5-VL vision support via `mmproj` files, but it is less mature than the HF/vLLM paths; expect friction.

**Alternative without llama.cpp:** `HFScorer` with `BERNOULLI_DEVICE=cpu` and `BERNOULLI_DTYPE=float32` runs the existing code path on any CPU box, no new scorer required. It is slower than llama.cpp (full-precision transformers on CPU, no quantization) but needs zero new code. Useful as a sanity check or for a one-shot script where you don't care about latency.
