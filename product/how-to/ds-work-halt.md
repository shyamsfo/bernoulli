# Project-specific halt steps

> These steps run AFTER the generic ds-work-halt steps (docs updated, committed, pushed).
> Add any project-specific teardown here — infrastructure hibernation, service shutdown, etc.
> Delete this file if there are no project-specific shutdown steps.

## TODO: Add project-specific shutdown steps

For now:
- No cloud infrastructure to hibernate — all development is local.
- If a vLLM server or FastAPI server was left running in the background during the session, stop it before ending.
- If a long-running training job (M5+) is in flight, do NOT halt — note it in the session report and leave it running.

## Phase-gate reminder

If the session ended on the final task of a milestone's exit criteria, the halt report should include a **gate report**:
- What was built this phase
- Metrics (accuracy, ECE, latency, etc. — whatever the phase gate requires)
- Deviations from the vision doc
- Explicit "waiting for go-ahead on M{N+1}" line

The vision doc §10 is strict about not rolling into the next phase without a go-ahead.
