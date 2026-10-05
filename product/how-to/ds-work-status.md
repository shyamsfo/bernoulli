# Project-specific ds-work-status additions

> These additions run during /ds-work-status prerequisite checking.
> Add any project-specific infrastructure checks needed before starting the next task.
> Delete this file if there are no project-specific prerequisite checks.

## Additional context to gather

- Confirm `uv` env is set up and `pytest` is runnable (`uv run pytest --collect-only` as a quick smoke).
- Check `CLAUDE.md` for the currently-selected backbone — the next task may require it to be downloaded locally.
- If a task touches `VLLMScorer` (M4+), check that `vllm` is importable and the pinned version is installed.

## Infrastructure prerequisite check

- **Local only.** No cloud resources need waking.
- If the next task requires loading the production backbone (M4+), verify there is enough free GPU memory and the weights are present at the configured local path (offline mode default).
- If the next task is in M5 (LoRA training), warn before kicking off — the vision doc requires asking before starting any training run.
