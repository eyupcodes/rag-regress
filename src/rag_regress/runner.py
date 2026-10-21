"""Run a user's pipeline over the golden set and record per-question results.

The pipeline is any callable ``f(query: str)`` that returns one of:

- a list of retrieved document ids;
- a mapping with ``retrieved_ids`` and optionally ``answer``;
- an object with ``retrieved_ids`` and optionally ``answer`` attributes.

A question whose call raises, or returns something unusable, is recorded with
its error and scored as an empty retrieval: a crash is a regression, not a
reason to abort the whole run.
"""

from __future__ import annotations

import hashlib
import importlib
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .golden import GoldenItem
from .judge import Judge
from .metrics import DEFAULT_KS, metric_sort_key, question_metrics

Pipeline = Callable[[str], Any]


class QuestionResult(BaseModel):
    id: str
    tags: tuple[str, ...] = ()
    retrieved_ids: list[str]
    answer: str | None = None
    latency_ms: float
    error: str | None = None
    judge_error: str | None = None
    metrics: dict[str, float]


class RunResult(BaseModel):
    created_at: str
    golden_sha256: str | None = None
    ks: list[int]
    repeats: int = 1
    questions: list[QuestionResult]
    aggregates: dict[str, float]
    aggregate_samples: dict[str, list[float]] = {}
    errors: int
    latency_ms_p50: float
    latency_ms_p95: float

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        """Read a run file; raises ``ValueError`` naming the file if it is invalid."""
        text = Path(path).read_text(encoding="utf-8")
        try:
            return cls.model_validate_json(text)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            reason = f"{where}: {first['msg']}" if where else first["msg"]
            raise ValueError(f"{path}: not a valid run file ({reason})") from None


class PipelineLoadError(ValueError):
    """The ``module:function`` reference cannot be imported or called."""


def load_pipeline(reference: str) -> Pipeline:
    module_name, sep, attr = reference.partition(":")
    if not sep or not module_name or not attr:
        raise PipelineLoadError(f"{reference!r}: expected 'package.module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"{reference!r}: cannot import {module_name}: {exc}"
        raise PipelineLoadError(msg) from None
    target: Any = module
    for part in attr.split("."):
        if not hasattr(target, part):
            raise PipelineLoadError(f"{reference!r}: {module_name} has no {attr}")
        target = getattr(target, part)
    if not callable(target):
        raise PipelineLoadError(f"{reference!r}: {attr} is not callable")
    return target


def _parse_output(output: Any) -> tuple[list[str], str | None]:
    if isinstance(output, Mapping):
        ids, answer = output.get("retrieved_ids"), output.get("answer")
    elif isinstance(output, Sequence) and not isinstance(output, str | bytes):
        ids, answer = output, None
    elif hasattr(output, "retrieved_ids"):
        ids, answer = output.retrieved_ids, getattr(output, "answer", None)
    else:
        raise TypeError(f"unsupported pipeline output: {type(output).__name__}")
    if not isinstance(ids, Sequence) or isinstance(ids, str | bytes):
        raise TypeError("retrieved_ids must be a list of strings")
    ids = list(ids)
    if not all(isinstance(doc_id, str) for doc_id in ids):
        raise TypeError("retrieved_ids must be a list of strings")
    if answer is not None and not isinstance(answer, str):
        raise TypeError("answer must be a string or None")
    return ids, answer


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(q) - 1]


def file_sha256(path: str | Path) -> str:
    """Fingerprint of a text file that ignores CRLF vs LF, so a Windows
    checkout and a Linux CI runner agree on the same golden set."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _parse_contexts(output: Any) -> list[str] | None:
    """Optional passage texts, used only by the LLM judge's faithfulness score."""
    if isinstance(output, Mapping):
        contexts = output.get("contexts")
    else:
        contexts = getattr(output, "contexts", None)
    if contexts is None:
        return None
    if isinstance(contexts, str | bytes) or not all(
        isinstance(c, str) for c in contexts
    ):
        raise TypeError("contexts must be a list of strings")
    return list(contexts)


@dataclass
class _Attempt:
    ids: list[str]
    answer: str | None
    contexts: list[str] | None
    latency_ms: float
    error: str | None


def _call(item: GoldenItem, pipeline: Pipeline, clock: Callable[[], float]) -> _Attempt:
    started = clock()
    error: str | None = None
    try:
        output = pipeline(item.query)
        ids, answer = _parse_output(output)
        contexts = _parse_contexts(output)
    except Exception as exc:  # the pipeline is user code: record, don't abort
        ids, answer, contexts = [], None, None
        error = f"{type(exc).__name__}: {exc}"
    return _Attempt(ids, answer, contexts, (clock() - started) * 1000, error)


def _score(
    item: GoldenItem, attempt: _Attempt, ks: Sequence[int], judge: Judge | None
) -> tuple[dict[str, float], str | None]:
    metrics = question_metrics(
        attempt.ids,
        item.relevant_ids,
        answer=attempt.answer,
        reference_answer=item.reference_answer,
        ks=ks,
    )
    if judge is None or attempt.answer is None or attempt.error:
        return metrics, None
    try:
        metrics.update(judge.metrics(item.query, attempt.answer, attempt.contexts))
    except Exception as exc:  # a judge failure must not look like a 0 score
        return metrics, f"{type(exc).__name__}: {exc}"
    return metrics, None


def run_pipeline(
    items: Sequence[GoldenItem],
    pipeline: Pipeline,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    repeats: int = 1,
    golden_sha256: str | None = None,
    judge: Judge | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunResult:
    """Score every question, ``repeats`` times for nondeterministic pipelines.

    With repeats, a question's metrics are the mean over its attempts, ids and
    answer come from the first attempt, latency is the mean, and the first
    error (if any) is kept. ``aggregate_samples`` holds one mean per repeat so
    that :func:`compare_runs` can tell run-to-run noise from a real drop.
    """
    if not items:
        raise ValueError("golden set is empty")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    ks = sorted(set(ks))

    attempts: list[list[QuestionResult]] = [[] for _ in items]
    for _ in range(repeats):
        for slot, item in zip(attempts, items, strict=True):
            attempt = _call(item, pipeline, clock)
            metrics, judge_error = _score(item, attempt, ks, judge)
            slot.append(
                QuestionResult(
                    id=item.id,
                    tags=item.tags,
                    retrieved_ids=attempt.ids,
                    answer=attempt.answer,
                    latency_ms=round(attempt.latency_ms, 3),
                    error=attempt.error,
                    judge_error=judge_error,
                    metrics=metrics,
                )
            )

    results = [_merge_attempts(slot) for slot in attempts]
    names = sorted({name for r in results for name in r.metrics}, key=metric_sort_key)
    samples: dict[str, list[float]] = {}
    for name in names:
        per_repeat = (
            [slot[i].metrics[name] for slot in attempts if name in slot[i].metrics]
            for i in range(repeats)
        )
        # A repeat where no question has this metric (say, the judge failed
        # everywhere) contributes no sample rather than a fake zero.
        samples[name] = [statistics.fmean(v) for v in per_repeat if v]
    latencies = [r.latency_ms for r in results]
    return RunResult(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        golden_sha256=golden_sha256,
        ks=ks,
        repeats=repeats,
        questions=results,
        aggregates={name: statistics.fmean(v) for name, v in samples.items()},
        aggregate_samples=samples if repeats > 1 else {},
        errors=sum(r.error is not None for r in results),
        latency_ms_p50=round(_percentile(latencies, 50), 3),
        latency_ms_p95=round(_percentile(latencies, 95), 3),
    )


def _merge_attempts(slot: list[QuestionResult]) -> QuestionResult:
    first = slot[0]
    if len(slot) == 1:
        return first
    return first.model_copy(
        update={
            "latency_ms": round(statistics.fmean(a.latency_ms for a in slot), 3),
            "error": next((a.error for a in slot if a.error), None),
            "judge_error": next((a.judge_error for a in slot if a.judge_error), None),
            "metrics": {
                name: statistics.fmean(
                    a.metrics[name] for a in slot if name in a.metrics
                )
                for name in sorted(
                    {n for a in slot for n in a.metrics}, key=metric_sort_key
                )
            },
        }
    )
