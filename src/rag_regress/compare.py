"""Compare a candidate run with a baseline and decide pass or fail.

A metric regresses when the candidate mean is lower than the baseline mean by
more than its tolerance (absolute, default 0.02). Every metric present in
both runs is gated unless the caller names a subset.

Averages can hide a question that went from right to wrong while another went
the other way, so the comparison also lists per-question changes, judged by
``hit@K`` at the largest cut-off both runs share.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel

from .runner import RunResult

DEFAULT_TOLERANCE = 0.02


class MetricDelta(BaseModel):
    name: str
    baseline: float
    candidate: float
    delta: float
    tolerance: float
    regressed: bool


class Comparison(BaseModel):
    passed: bool
    hit_metric: str
    metrics: list[MetricDelta]
    newly_failing: list[str]
    newly_passing: list[str]
    newly_erroring: list[str]
    only_in_baseline: list[str]
    only_in_candidate: list[str]
    warnings: list[str]

    @property
    def regressions(self) -> list[MetricDelta]:
        return [m for m in self.metrics if m.regressed]


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

    shared = sorted(set(baseline.aggregates) & set(candidate.aggregates))
    names = shared if gate is None else list(dict.fromkeys(gate))
    missing = [n for n in names if n not in shared]
    if missing:
        raise ValueError(f"metrics not in both runs: {', '.join(missing)}")

    deltas = []
    for name in names:
        base, cand = baseline.aggregates[name], candidate.aggregates[name]
        allowed = tolerances.get(name, tolerance)
        delta = cand - base
        deltas.append(
            MetricDelta(
                name=name,
                baseline=base,
                candidate=cand,
                delta=delta,
                tolerance=allowed,
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
        ("only in baseline", result.only_in_baseline),
        ("only in candidate", result.only_in_candidate),
    ):
        if ids:
            lines.append(f"  {label}: {', '.join(ids)}")
    lines.extend(f"  warning: {w}" for w in result.warnings)
    return "\n".join(lines)
