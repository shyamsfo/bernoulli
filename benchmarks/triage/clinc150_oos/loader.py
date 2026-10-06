"""CLINC150-OOS — out-of-scope detection benchmark loader.

Dataset: `clinc_oos` config `plus` — 150 in-scope intents across 10
domains + an explicit out-of-scope class. Each example is a short user
utterance; `intent` is an integer label where one specific id
corresponds to the OOS class.

Framed here as a **BinaryQuestion** on in-scope-ness: "Is this message
addressed by any of our supported intents?" with gold = `Yes` for
in-scope rows and `No` for OOS. This matches the use-case card on the
landing page: *the probability flags "none of these"*. The 150-way
intent classification is parked — it would need a schema change
(BenchmarkExample.question → questions:list) for the natural
multi-head framing of "in-scope?" + "which intent?".

OOS detection convention: this loader looks up the OOS label by name
via `features['intent'].names` (the entry equal to `"oos"`). If a
future dataset revision renames the OOS label we'll fail loudly at
load time rather than silently mis-score.

- `load(limit=None)` → `Iterator[BenchmarkExample]` over `test` (5,500 examples: 4,500 in-scope + 1,000 OOS).
- `load_train(limit=None)` → `Iterator[BenchmarkExample]` over `train` (15,250 examples; 15,000 in-scope + 250 OOS).
- `REWORD_STEMS` — three paraphrased stems for the stability runner.
"""

from __future__ import annotations

from collections.abc import Iterator

from benchmarks.common.dataset import BenchmarkExample
from bernoulli.types import BinaryQuestion

SOURCE = "clinc150_oos"
_HF_PATH = "clinc_oos"
_HF_CONFIG = "plus"
_OOS_NAME = "oos"

QUESTION = BinaryQuestion(
    id="in_scope",
    prompt=(
        "Does this user message fall within our supported intents, or is it "
        "out-of-scope (none of the above)? Answer Yes if in-scope, No if out-of-scope."
    ),
)

REWORD_STEMS: tuple[str, ...] = (
    "Is this message something our assistant knows how to handle?",
    "Would any of our intent handlers match this user message?",
    "Is this message in-scope for our supported intent set?",
)


def _oos_label_id(ds: object) -> int:
    """Resolve the integer label corresponding to the OOS class."""
    features = ds.features  # type: ignore[attr-defined]
    names: list[str] = list(features["intent"].names)
    try:
        return names.index(_OOS_NAME)
    except ValueError as exc:
        raise RuntimeError(
            f"clinc_oos: expected an intent named {_OOS_NAME!r} in features; got {names[:5]}..."
        ) from exc


def _yield_split(split: str, limit: int | None) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    ds = load_dataset(_HF_PATH, _HF_CONFIG, split=split)
    oos_id = _oos_label_id(ds)
    features = ds.features  # type: ignore[attr-defined]
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        is_oos = row["intent"] == oos_id
        yield BenchmarkExample(
            state_text=row["text"].strip(),
            question=QUESTION,
            gold="No" if is_oos else "Yes",
            source=SOURCE,
            meta={
                "example_id": f"clinc-{split}-{i}",
                "intent_id": int(row["intent"]),
                "intent_name": features["intent"].int2str(row["intent"]),
                "is_oos": bool(is_oos),
            },
        )


def load(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the evaluation split (`test`, 5,500 examples)."""
    return _yield_split("test", limit)


def load_train(limit: int | None = None) -> Iterator[BenchmarkExample]:
    """Stream the training split (15,250 examples) — for trainable baselines (`BGEm3LR`)."""
    return _yield_split("train", limit)


def extra_metrics(
    gold: list[str],
    preds: list[dict[str, float]],
    examples: list[BenchmarkExample],
) -> dict[str, float]:
    """Benchmark-specific metrics the generic runner doesn't emit.

    - `oos_auroc` — treat P(gold == "No") as the OOS-detection score and
      `meta.is_oos` as the ground-truth flag. Standard binary ROC-AUC.
      Degenerate (one class only) → returns 0.5.
    - `in_scope_accuracy` — accuracy on the in-scope-only subset.
    - `oos_recall` — fraction of OOS examples the baseline flags as "No".
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    is_oos = np.asarray([bool(ex.meta.get("is_oos", False)) for ex in examples], dtype=bool)
    scores = np.asarray([float(p.get("No", 0.0)) for p in preds], dtype=np.float64)
    top1 = [max(p.items(), key=lambda kv: (kv[1], kv[0]))[0] for p in preds]

    out: dict[str, float] = {}
    if is_oos.any() and (~is_oos).any():
        out["oos_auroc"] = float(roc_auc_score(is_oos.astype(int), scores))
    else:
        out["oos_auroc"] = 0.5  # degenerate — one class only

    in_scope_mask = ~is_oos
    if in_scope_mask.any():
        correct_in = sum(1 for i in range(len(gold)) if in_scope_mask[i] and top1[i] == gold[i])
        out["in_scope_accuracy"] = correct_in / int(in_scope_mask.sum())
    else:
        out["in_scope_accuracy"] = 0.0

    if is_oos.any():
        flagged = sum(1 for i in range(len(gold)) if is_oos[i] and top1[i] == "No")
        out["oos_recall"] = flagged / int(is_oos.sum())
    else:
        out["oos_recall"] = 0.0
    return out
