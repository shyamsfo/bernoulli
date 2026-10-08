# Project-specific ds-work-status additions

> These additions run during /ds-work-status prerequisite checking.
> Add any project-specific infrastructure checks needed before starting the next task.
> Delete this file if there are no project-specific prerequisite checks.

## Additional context to gather

- Confirm `uv` env is set up and `pytest` is runnable (`uv run pytest --collect-only` as a quick smoke).
- Check `CLAUDE.md` for the currently-selected backbone — the next task may require it to be downloaded locally.
- If a task touches `VLLMScorer` (M4+), check that `vllm` is importable and the pinned version is installed.
- **Scan `product/research/spikes/*.md` for in-progress spikes.** Each spike doc carries a `**Status**: in progress | done | abandoned` marker near the top. If any are `in progress`, surface them **before** the milestone brief — spikes are side-tracks that would otherwise be invisible to the active-milestone scan. For an in-progress spike, surface: the spike title, the first unchecked `[ ]` step in its plan, and the path to its doc. If multiple spikes are in progress, list all; the user picks which to resume. The convention: a spike is "in progress" from the moment its plan is written until its findings have been folded back into the feeding milestone's tasks.

## Infrastructure prerequisite check

- **Local only.** No cloud resources need waking.
- If the next task requires loading the production backbone (M4+), verify there is enough free GPU memory and the weights are present at the configured local path (offline mode default).
- If the next task is in M5 (LoRA training), warn before kicking off — the vision doc requires asking before starting any training run.
