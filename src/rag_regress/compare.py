"""Compare a candidate run with a baseline and decide pass or fail.

A metric regresses when the candidate mean is lower than the baseline mean by
more than its tolerance (absolute, default 0.02). Every metric present in
both runs is gated unless the caller names a subset.

Averages can hide a question that went from right to wrong while another went
the other way, so the comparison also lists per-question changes, judged by
``hit@K`` at the largest cut-off both runs share.

Runs made with ``--repeats`` carry one mean per repeat. The allowed drop then
widens to ``NOISE_Z`` combined standard errors of the two means when that is
larger than the tolerance, so run-to-run noise in a nondeterministic pipeline
is not reported as a regression, while a drop well beyond it still is.
Questions whose hit rate is strictly between 0 and 1 in the candidate are
listed as flaky.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping

from pydantic import BaseModel

from .metrics import metric_sort_key
from .runner import RunResult

DEFAULT_TOLERANCE = 0.02
NOISE_Z = 2.0


class MetricDelta(BaseModel):
    name: str
    baseline: float
    candidate: float
    delta: float
    tolerance: float
    noise: float = 0.0
    regressed: bool


class Comparison(BaseModel):
    passed: bool
    hit_metric: str
    metrics: list[MetricDelta]
    newly_failing: list[str]
    newly_passing: list[str]
    newly_erroring: list[str]
    flaky: list[str] = []
    only_in_baseline: list[str]
    only_in_candidate: list[str]
    warnings: list[str]

    @property
    def regressions(self) -> list[MetricDelta]:
        return [m for m in self.metrics if m.regressed]


def _standard_error(samples: list[float] | None) -> float:
    """Standard error of the mean of per-repeat aggregates (0 for one run)."""
    if not samples or len(samples) < 2:
        return 0.0
    return statistics.stdev(samples) / math.sqrt(len(samples))


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    tolerances: Mapping[str, float] | None = None,
    gate: Iterable[str] | None = None,
) -> Comparison:
    if tolerance < 0 or any(t < 0 for t in (tolerances or {}).values()):
        raise ValueError("tolerances must not be negative")
    tolerances = dict(tolerances or {})

    shared = sorted(
        set(baseline.aggregates) & set(candidate.aggregates), key=metric_sort_key
    )
    unknown = [n for n in tolerances if n not in shared]
    if unknown:
        raise ValueError(f"tolerance set for unknown metrics: {', '.join(unknown)}")
    names = shared if gate is None else list(dict.fromkeys(gate))
    missing = [n for n in names if n not in shared]
    if missing:
        raise ValueError(f"metrics not in both runs: {', '.join(missing)}")

    deltas = []
    for name in names:
        base, cand = baseline.aggregates[name], candidate.aggregates[name]
        noise = NOISE_Z * math.hypot(
            _standard_error(baseline.aggregate_samples.get(name)),
            _standard_error(candidate.aggregate_samples.get(name)),
        )
        allowed = max(tolerances.get(name, tolerance), noise)
        delta = cand - base
        deltas.append(
            MetricDelta(
                name=name,
                baseline=base,
                candidate=cand,
                delta=delta,
                tolerance=allowed,
                noise=noise,
                # Rounding keeps float noise (0.30000000000000004) from gating.
                regressed=round(-delta, 12) > allowed,
            )
        )

    shared_k = sorted(set(baseline.ks) & set(candidate.ks))
    if not shared_k:
        raise ValueError("the runs share no @k cut-off")
    hit = f"hit@{shared_k[-1]}"

    base_q = {q.id: q for q in baseline.questions}
    cand_q = {q.id: q for q in candidate.questions}
    newly_failing, newly_passing, newly_erroring = [], [], []
    for qid in (q for q in base_q if q in cand_q):
        before, after = base_q[qid], cand_q[qid]
        if before.metrics[hit] == 1.0 and after.metrics[hit] == 0.0:
            newly_failing.append(qid)
        elif before.metrics[hit] == 0.0 and after.metrics[hit] == 1.0:
            newly_passing.append(qid)
        if after.error and not before.error:
            newly_erroring.append(qid)

    warnings = []
    if (
        baseline.golden_sha256
        and candidate.golden_sha256
        and baseline.golden_sha256 != candidate.golden_sha256
    ):
        warnings.append(
            "the golden set changed between the runs; averages may not be comparable"
        )

    return Comparison(
        passed=not any(d.regressed for d in deltas),
        hit_metric=hit,
        metrics=deltas,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
        newly_erroring=newly_erroring,
        flaky=[q.id for q in candidate.questions if 0.0 < q.metrics[hit] < 1.0],
        only_in_baseline=sorted(set(base_q) - set(cand_q)),
        only_in_candidate=sorted(set(cand_q) - set(base_q)),
        warnings=warnings,
    )


def format_summary(result: Comparison) -> str:
    """Plain-text verdict for the terminal."""
    lines = [
        "PASS: no metric regressed" if result.passed else "FAIL: quality regressed"
    ]
    for m in result.regressions:
        lines.append(
            f"  {m.name}: {m.baseline:.3f} -> {m.candidate:.3f} "
            f"({m.delta:+.3f}, tolerance {m.tolerance:.3f})"
        )
    for label, ids in (
        (f"newly failing ({result.hit_metric})", result.newly_failing),
        (f"newly passing ({result.hit_metric})", result.newly_passing),
        ("newly erroring", result.newly_erroring),
        (f"flaky ({result.hit_metric} varies between repeats)", result.flaky),
        ("only in baseline", result.only_in_baseline),
        ("only in candidate", result.only_in_candidate),
    ):
        if ids:
            lines.append(f"  {label}: {', '.join(ids)}")
    lines.extend(f"  warning: {w}" for w in result.warnings)
    return "\n".join(lines)
