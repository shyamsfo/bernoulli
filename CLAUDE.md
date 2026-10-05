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

- `bernoulli` → g6.xlarge in us-east-1, L4 24GB, NVIDIA GPU Base CUDA on Ubuntu 24.04
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
| `BERNOULLI_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` |
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

## Gotchas

- **NVMe mount**: the NVIDIA GPU Base AMI already LVM-mounts the g6.xlarge instance-store at `/opt/dlami/nvme`. Don't try to `mkfs` it — see `deploy/user_data.sh` for the correct detection pattern.
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
