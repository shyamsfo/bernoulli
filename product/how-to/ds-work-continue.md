# Project-specific ds-work-continue additions

> Fill in any project-specific context sources that /ds-work-continue should gather.
> Examples: a CURRENT_STATE.md file, a cluster status check, an infra state API call.
> Delete this file if there are no project-specific additions needed.

## Additional context to gather

- Read `vision_and_roadmap.md` at repo root for the authoritative spec (API shape, architecture, phase gates, risks).
- If `CLAUDE.md` exists, read it for current model choice, config, and any gotchas noted during prior sessions.
- Note the currently-selected backbone (dev or production) — the project is explicitly backbone-agnostic, so this can change between sessions.
- Check instance state: `terraform -chdir=deploy output -raw instance_id` + `aws ec2 describe-instances --instance-ids <id> --query 'Reservations[0].Instances[0].State.Name'`. If `stopped` or `stopping`, follow the resume protocol below before any task that needs the dev box.
- **Scan `product/research/spikes/*.md` for in-progress spikes.** Each spike doc carries a `**Status**: in progress | done | abandoned` marker near the top. If any are `in progress`, surface them in the resume brief **before** the milestone context — spikes are side-tracks that would otherwise be invisible to the active-milestone scan. For an in-progress spike, surface: the spike title, the first unchecked `[ ]` step in its plan, the feeding-milestone it informs, and the path to its doc. The convention: a spike is "in progress" from the moment its plan is written until its findings have been folded back into the feeding milestone's tasks or the spike is explicitly marked `done` / `abandoned`.

## Additional fields in resume brief

- **Current backbone**: dev / production, model id, revision hash (from `CLAUDE.md`).
- **Active phase gate**: when the next phase gate is approaching (last milestone task before an exit-criteria check), flag it so the session can plan to end on a clean gate report.
- **Instance state**: `running` / `stopped`. If stopped, flag that first-run tasks pay a ~90 sec HF re-pull cost (NVMe is ephemeral).

## EC2 resume protocol (dev box stopped between sessions)

Stopping `bernoulli` between sessions saves ~$1/hr but wipes the instance-store NVMe and changes the public IP. To resume:

```bash
# 1. Start the instance (instance_id is stable across stops)
aws ec2 start-instances --instance-ids $(terraform -chdir=deploy output -raw instance_id)

# 2. Wait ~30 sec for IP assignment, then re-paste the ssh stanza
terraform -chdir=deploy refresh
terraform -chdir=deploy output -raw ssh_config_stanza
#    → paste that into ~/.ssh/config, replacing the previous `Host bernoulli` block

# 3. Confirm cloud-init finished + GPU healthy
ssh bernoulli 'sudo cloud-init status --wait && nvidia-smi | head -14'

# 4. Recreate venv + re-pull Qwen weights (~90 sec via hf_transfer)
ssh bernoulli 'cd ~/bernoulli && uv sync && just model-pull'
```

What's lost on stop (all cheap to recreate):

- `/opt/dlami/nvme/hf-cache` — Qwen2.5-VL-7B (~15 GB), BGE-m3 (~2 GB), DeBERTa-v3 (~800 MB) if ever pulled. Re-pulls on first use via `huggingface_hub` + `hf_transfer`.
- `~/bernoulli/.venv` — rebuilt from `uv.lock` by `uv sync` (~1 min).
- Docker data-root at `/opt/dlami/nvme/docker` — only matters if the next task uses `just docker-run`.
- Public IP — hence the ssh-stanza re-paste step.

What's **not** lost: the EBS root (`~/bernoulli/` source tree, bashrc, env). Code state on bernoulli survives stop/start — only the NVMe is ephemeral.

## Known project-specific footguns to re-check

- **`just sync-mirror` has `--delete`.** It will remove files on `bernoulli` that don't exist locally. Before running sync-mirror, make sure any benchmark results on the box have been pulled back with `just benchmarks-pull` first. Lost 4 M9 result files to this once on 2026-10-07.
- **GPU zombies after a crashed `VLLM::EngineCore`.** If `nvidia-smi` shows memory in use but no running python process, kill with `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r sudo kill -9`.
- **Server + Generative baseline on same GPU is fine now.** `/v1/generate` endpoint shipped 2026-10-07; the earlier GPU-contention bug (second HFScorer load → OOM) is resolved.
- **BinaryQuestion debias.** Reverse/cyclic debias is auto-no-op for binary (`Yes`/`No` are semantic, not positional). If you reintroduce manual debiasing on a binary task, you'll hit the PAWS below-random bug again — see `product/learnings/paws-below-random.md`.
