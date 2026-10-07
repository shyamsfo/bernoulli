"""Build the post-cutoff arXiv classification snapshot.

Not run in CI; invoke manually on the dev box when generating or refreshing
`data/dataset.jsonl`:

    uv run python -m benchmarks.academic.arxiv_post_cutoff.make_dataset \
        --target 200 \
        --out benchmarks/academic/arxiv_post_cutoff/data/dataset.jsonl

Queries the arXiv API for recent papers in each of the four
`CATEGORIES`, filters to publication dates strictly after
`BACKBONE_CUTOFF`, balances per-category counts, splits into
`train`/`test` deterministically, and writes JSONL with one row per line.

**Fields written per row**:

- `id` — the arXiv identifier (e.g. `2501.12345`).
- `title` — raw paper title.
- `abstract` — raw paper abstract.
- `primary_category` — one of the four `CATEGORIES`.
- `publication_date` — ISO date string, strictly after `BACKBONE_CUTOFF`.
- `split` — `"train"` or `"test"`.

**Why this script exists:**
The post-cutoff claim is load-bearing — the whole benchmark is pointless
if any example could have been in the backbone's pretraining data. This
script makes the cutoff rule mechanical (not a judgment call) and keeps
the generation reproducible so the claim survives backbone swaps:

- Update `BACKBONE_CUTOFF` in `loader.py` when swapping to a backbone
  with a later cutoff.
- Re-run this script; commit the regenerated `data/dataset.jsonl`.
- Update the model-cutoff claim in `benchmarks/README.md` and each run's
  `results/latest.md`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, datetime
from pathlib import Path

from benchmarks.academic.arxiv_post_cutoff.loader import BACKBONE_CUTOFF, CATEGORIES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        type=int,
        default=200,
        help="Total number of papers to collect across all categories (balanced).",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.5,
        help="Fraction of rows to assign to the 'test' split; the rest go to 'train'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the JSONL snapshot.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260101,
        help="Deterministic split seed; also controls per-category ordering tiebreaks.",
    )
    p.add_argument(
        "--max-fetch-per-cat",
        type=int,
        default=400,
        help="Upper bound on API results per category (we may discard pre-cutoff ones).",
    )
    return p.parse_args()


def _normalise_arxiv_id(entry_id: str) -> str:
    """`http://arxiv.org/abs/2501.12345v2` → `2501.12345`."""
    short = entry_id.rsplit("/", 1)[-1]
    if "v" in short:
        short = short.split("v", 1)[0]
    return short


def _primary_cat(result: object) -> str:
    """arxiv.Result.primary_category can be a full code (e.g. `cs.CL.X`); keep the top two parts."""
    code: str = result.primary_category  # type: ignore[attr-defined]
    parts = code.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else code


def _fetch_category(
    client: object, category: str, cutoff: date, needed: int, max_fetch: int
) -> list[dict[str, str]]:
    """Fetch recent papers in `category`, keep ones strictly after `cutoff`, cap at `needed`."""
    import arxiv

    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_fetch,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    rows: list[dict[str, str]] = []
    for result in client.results(search):  # type: ignore[attr-defined]
        pub: date = result.published.date()
        if pub <= cutoff:
            continue
        # arXiv's "primary_category" may be more specific than our list; keep the top.
        prim = _primary_cat(result)
        if prim != category:
            continue
        rows.append(
            {
                "id": _normalise_arxiv_id(result.entry_id),
                "title": result.title.strip(),
                "abstract": result.summary.strip().replace("\n", " "),
                "primary_category": category,
                "publication_date": pub.isoformat(),
            }
        )
        if len(rows) >= needed:
            break
    return rows


def main() -> None:  # pragma: no cover — invoked manually on the dev box, not in CI
    args = _parse_args()

    try:
        import arxiv
    except ImportError:
        sys.exit(
            "make_dataset: missing `arxiv` package. Install with `uv add arxiv` first."
        )

    cutoff = datetime.fromisoformat(BACKBONE_CUTOFF).date()
    per_cat = args.target // len(CATEGORIES)
    print(f"Target: {args.target} papers = {per_cat} per category × {len(CATEGORIES)}", file=sys.stderr)
    print(f"Cutoff: publication_date strictly after {cutoff.isoformat()}", file=sys.stderr)
    print(f"Output: {args.out}", file=sys.stderr)

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    all_rows: list[dict[str, str]] = []
    for cat in CATEGORIES:
        print(f"[fetch] {cat} → aiming for {per_cat} ...", file=sys.stderr)
        got = _fetch_category(
            client, cat, cutoff, needed=per_cat, max_fetch=args.max_fetch_per_cat
        )
        print(f"[fetch] {cat} → got {len(got)}", file=sys.stderr)
        if len(got) < per_cat:
            print(
                f"[fetch] WARN: {cat} returned only {len(got)}/{per_cat}; raise --max-fetch-per-cat or widen the cutoff",
                file=sys.stderr,
            )
        all_rows.extend(got)

    # Deterministic shuffle, per-category train/test split so each split is class-balanced.
    rng = random.Random(args.seed)
    test_rows: list[dict[str, str]] = []
    train_rows: list[dict[str, str]] = []
    for cat in CATEGORIES:
        cat_rows = [r for r in all_rows if r["primary_category"] == cat]
        rng.shuffle(cat_rows)
        n_test = round(len(cat_rows) * args.test_fraction)
        for row in cat_rows[:n_test]:
            row["split"] = "test"
            test_rows.append(row)
        for row in cat_rows[n_test:]:
            row["split"] = "train"
            train_rows.append(row)

    # Combine with test rows first so the file ordering is stable + inspectable.
    final = test_rows + train_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fp:
        for row in final:
            fp.write(json.dumps(row) + "\n")

    print(
        f"[done] wrote {len(final)} rows → {args.out} "
        f"({len(test_rows)} test, {len(train_rows)} train)",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
