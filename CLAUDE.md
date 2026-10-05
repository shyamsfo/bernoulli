# Bernoulli — repo notes for Claude Code

This file is the authoritative session reference. Keep commands, config, decisions, and gotchas current.

## Workflow

Task runner is `just` (see `justfile`). Run `just` with no args to list available tasks.

```bash
# push to dev box
just sync-mirror

# run full ci on dev box (what you'd want green before any merge)
ssh bernoulli 'cd ~/bernoulli && just ci'

# just tests
ssh bernoulli 'cd ~/bernoulli && uv run pytest -v'

# shell in
just shell
```

**All tests + eval runs happen on bernoulli.** GHA is intentionally not wired up. See `product/milestones.md` M1 notes for the rationale.

## Dev box

- `bernoulli` → **g6.xlarge** in us-east-1, **L4 24GB**, NVIDIA GPU Base CUDA on Ubuntu 24.04. Pinned to **us-east-1a** (set in `deploy/variables.tf`); override with `-var 'availability_zone=us-east-1b'` if that AZ hits InsufficientInstanceCapacity. Instance has a 4-min create timeout so capacity failures surface fast instead of terraform retrying forever.
- SSH key: `~/.ssh/id_nuwire` (via the `Host bernoulli` stanza in `~/.ssh/config`)
- Instance-store NVMe (232GB, ephemeral): `/opt/dlami/nvme`, pre-mounted by the AMI via LVM
- HF cache: `/opt/dlami/nvme/hf-cache` (`HF_HOME` set in `~/.bashrc`)
- EBS root: 80GB gp3 encrypted
- Terraform: `deploy/` — up with `just up`, down with `just down`

**Stop/start caveat:** stopping and starting the instance drops the NVMe contents (HF cache included) and changes the public IP. Re-pulling the model takes ~9 min; re-paste the `ssh_config_stanza` output after start.

## Config

Env-driven via pydantic-settings, prefix `BERNOULLI_`. Defaults in `bernoulli/config.py`:

| var | default |
|---|---|
| `BERNOULLI_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` (dev + early M4; step up to 32B AWQ later when capacity frees up) |
| `BERNOULLI_MODEL_REVISION` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `BERNOULLI_DTYPE` | `bfloat16` |
| `BERNOULLI_DEVICE` | `cuda` |
| `BERNOULLI_MAX_MODEL_LEN` | `32768` |
| `BERNOULLI_SCORER` | `hf` (M2) / `vllm` (M4+) |
| `BERNOULLI_DEFAULT_DEBIAS` | `reverse` |

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-10-05 | Dev-tier backbone: `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc59489...` | Fits L4 24GB in bf16 with headroom, public (no gating), well-tested logit behavior around single-letter answers. Flexible — see `bernoulli/scorer.py` interface. |
| 2026-10-05 | No LiteLLM in the Scorer path | The core technique reads *logits over label tokens* with `allowed_token_ids` + prefix caching. OpenAI-compat APIs don't expose any of that uniformly. Scorer stays as `HFScorer` / `VLLMScorer`. LiteLLM may show up in `evals/baselines.py` later for the generative-baseline comparison in M3. |
| 2026-10-05 | AWS g6.xlarge for dev; no auto-shutdown | User explicitly chose ease > cost. $0.80/hr running, $6.40/mo stopped. |
| 2026-10-05 | CI on bernoulli, not GHA | The real tests need a GPU; GHA free runners can't run them, so GHA would gate on nothing meaningful. Clean-env check comes from `terraform destroy && apply` instead. |
| 2026-10-05 | Letters A–I for rating labels (not digits) | **Deviation from vision doc §4.** Qwen's tokenizer makes `" 1"`…`" 9"` TWO tokens (space + digit), which breaks the single-token-answer invariant. Letters A–I are single-token (one per rating value, up to scale width 9). Ratings map A→low, B→low+1, … and the response distribution keys are integer strings per the API spec. |
| 2026-10-05 | Model class is hard-coded to `Qwen2_5_VLForConditionalGeneration` | Only one backbone in the picture for M2–M3. When M4 introduces the production backbone, generalize via a small registry (dispatch on `AutoConfig(name_or_path).model_type`). Not worth building the registry for one model. |
| 2026-10-05 | `AutoProcessor` deferred to M4 | For Qwen2.5-VL the processor pulls `Qwen2VLVideoProcessor`, which hard-requires `torchvision`. M2 is text-only so we use `AutoTokenizer` directly. Images (and the processor + torchvision dep) land in M4. |
| 2026-10-05 | Milestones re-ordered: text-only shippable product before multimodal | User direction. New order: M4 production serving (text-only) → M5 hardening (v1.0) → M6 multimodal → M7 LoRA (optional). Audio parked. |
| 2026-10-05 | Production backbone: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` @ `66c370b7`, instance `g6e.xlarge` (L40S 48GB) | AWQ int4 is ~20 GB — fits L40S 48GB with ~25 GB headroom for KV cache + batching at 32k context. L40S has native int8/int4 math. Single-GPU keeps vLLM config simple. Instance cost ~$1.86/hr on-demand. Dev-tier 7B is still runnable on the same box via `BERNOULLI_MODEL_ID` override. |
| 2026-10-05 | Deferred the step-up to M4e — stay on g6.xlarge + 7B through M4a-d | g6e.xlarge capacity in us-east-1 is tight today (us-east-1c and 1a both reported InsufficientInstanceCapacity when the plan was fresh). The vLLM scorer, FastAPI server, request batching, Dockerfile, load test, and HTTP eval harness are all backbone-agnostic and run fine on 7B / L4 24GB. Revisit the g6e capacity + 32B pull when the production step-up actually matters. Added `availability_zone` variable (default us-east-1a) + 4-min create timeout so capacity issues surface fast. |

## Gotchas

- **vLLM max_model_len on 24GB GPUs**: default `BERNOULLI_MAX_MODEL_LEN=32768` won't fit on A10G/L4 24GB after loading 7B weights. Set `BERNOULLI_MAX_MODEL_LEN=8192` for vLLM on these cards; HFScorer doesn't care.
- **Zombie `VLLM::EngineCore`**: if the vLLM engine init crashes partway (e.g., config error), the engine subprocess can survive the Python exit holding ~18 GB on the GPU. Check `nvidia-smi --query-compute-apps=pid --format=csv,noheader` and `sudo kill -9 <pid>`.
- **CUDA alignment on torch/vision/audio**: when bumping the pytorch index version (`[tool.uv.sources]` → `pytorch-cu130` etc.), bump all three. `transformers>=5.x` imports torchaudio at import time, so a mismatched CUDA version silently breaks every scorer load with a cryptic error from inside torchaudio.
- **NVMe mount**: the NVIDIA GPU Base AMI already LVM-mounts the instance-store at `/opt/dlami/nvme` on g6/g6e. Don't try to `mkfs` it — see `deploy/user_data.sh` for the correct detection pattern.
- **Qwen VL thinking mode**: Qwen hybrid-reasoning models (3+) can emit `<think>` before the answer. The scorer must disable thinking in the chat template and assert next token is a label. Verify at `HFScorer` construction (M2 task).
- **Label alphabet runs out at 26**: Banking77 (77-way intent) needs chunked / tournament scoring. See parking-lot.

## Commands cheat sheet

```bash
# model pull (one-time, ~15GB, ~90s via hf_transfer)
ssh bernoulli 'cd ~/bernoulli && just model-pull'

# check gpu + cache size + instance state
just status

# full ci
just shell  # then: cd ~/bernoulli && just ci

# run an eval
just eval sst2 reverse       # dataset=sst2, debias=reverse → evals/reports/sst2.md
just reports-pull            # pull generated reports back from bernoulli
```
