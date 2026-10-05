# Project-specific ds-work-update additions

> These additions run during /ds-work-update state-gathering, before writing the report.
> Add any project-specific state to capture in mid-session snapshots.
> Delete this file if there are no project-specific additions needed.

## Additional state to gather

- Currently loaded backbone (model id + revision hash) if any inference was run this session.
- Latest eval metrics if an eval report was regenerated (accuracy, macro-F1, ECE, Brier, latency p50/p95).
- Any model downloads kicked off during the session (and their size — the vision doc requires asking before downloads > 20GB).

## Additional fields in the report snapshot

- **Backbone**: `<model id>@<revision>` (dev / production).
- **Metrics (if updated)**: a compact table from the latest `evals/reports/` entry.
- **Open gate-blockers**: anything that would prevent passing the current milestone's exit criteria.
