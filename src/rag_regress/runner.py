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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .golden import GoldenItem
from .metrics import DEFAULT_KS, metric_sort_key, question_metrics

Pipeline = Callable[[str], Any]


class QuestionResult(BaseModel):
    id: str
    tags: tuple[str, ...] = ()
    retrieved_ids: list[str]
    answer: str | None = None
    latency_ms: float
    error: str | None = None
    metrics: dict[str, float]


class RunResult(BaseModel):
    created_at: str
    golden_sha256: str | None = None
    ks: list[int]
    questions: list[QuestionResult]
    aggregates: dict[str, float]
    errors: int
    latency_ms_p50: float
    latency_ms_p95: float

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


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


def run_pipeline(
    items: Sequence[GoldenItem],
    pipeline: Pipeline,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    golden_sha256: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunResult:
    if not items:
        raise ValueError("golden set is empty")
    ks = sorted(set(ks))
    results: list[QuestionResult] = []
    for item in items:
        started = clock()
        error: str | None = None
        try:
            ids, answer = _parse_output(pipeline(item.query))
        except Exception as exc:  # the pipeline is user code: record, don't abort
            ids, answer = [], None
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (clock() - started) * 1000
        results.append(
            QuestionResult(
                id=item.id,
                tags=item.tags,
                retrieved_ids=ids,
                answer=answer,
                latency_ms=round(latency_ms, 3),
                error=error,
                metrics=question_metrics(
                    ids,
                    item.relevant_ids,
                    answer=answer,
                    reference_answer=item.reference_answer,
                    ks=ks,
                ),
            )
        )

    names = sorted({name for r in results for name in r.metrics}, key=metric_sort_key)
    aggregates = {
        name: statistics.fmean(r.metrics[name] for r in results if name in r.metrics)
        for name in names
    }
    latencies = [r.latency_ms for r in results]
    return RunResult(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        golden_sha256=golden_sha256,
        ks=ks,
        questions=results,
        aggregates=aggregates,
        errors=sum(r.error is not None for r in results),
        latency_ms_p50=round(_percentile(latencies, 50), 3),
        latency_ms_p95=round(_percentile(latencies, 95), 3),
    )
