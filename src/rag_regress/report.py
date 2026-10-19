"""Render a comparison for people (Markdown, short enough for a PR comment)
and for tools (JSON).

The Markdown starts with a hidden marker so a CI job can find and update its
own comment instead of posting a new one on every push.
"""

from __future__ import annotations

import json

from .compare import Comparison

MARKER = "<!-- rag-regress-report -->"
COLLAPSE_AFTER = 10


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _question_list(title: str, ids: list[str]) -> list[str]:
    if not ids:
        return []
    items = ", ".join(f"`{qid}`" for qid in ids)
    if len(ids) <= COLLAPSE_AFTER:
        return [f"**{title}** ({len(ids)}): {items}", ""]
    return [
        "<details>",
        f"<summary><b>{title}</b> ({len(ids)})</summary>",
        "",
        items,
        "",
        "</details>",
        "",
    ]


def render_markdown(result: Comparison, *, title: str = "rag-regress") -> str:
    regressed = result.regressions
    if result.passed:
        verdict = f"✅ **{title}: pass**. No gated metric dropped beyond its tolerance."
    else:
        names = ", ".join(f"`{m.name}`" for m in regressed)
        verdict = f"❌ **{title}: regression** in {names}."

    lines = [MARKER, verdict, ""]
    lines += [f"> ⚠️ {w}" for w in result.warnings]
    if result.warnings:
        lines.append("")

    lines += [
        "| Metric | Baseline | Candidate | Δ | Tolerance | |",
        "|---|---:|---:|---:|---:|:-:|",
    ]
    # Regressions first, then the rest in name order, so the problem is on top.
    ordered = regressed + [m for m in result.metrics if not m.regressed]
    for m in ordered:
        status = "❌" if m.regressed else ("⬆️" if m.delta > 0 else "")
        lines.append(
            f"| `{m.name}` | {_fmt(m.baseline)} | {_fmt(m.candidate)} "
            f"| {m.delta:+.3f} | {_fmt(m.tolerance)} | {status} |"
        )
    lines.append("")
    if any(m.noise > 0 for m in result.metrics):
        lines += [
            "Tolerances widen to two standard errors of run-to-run noise "
            "where that is larger.",
            "",
        ]

    lines += _question_list(
        f"Newly failing ({result.hit_metric})", result.newly_failing
    )
    lines += _question_list(
        f"Newly passing ({result.hit_metric})", result.newly_passing
    )
    lines += _question_list("Newly erroring", result.newly_erroring)
    lines += _question_list(
        f"Flaky ({result.hit_metric} varies between repeats)", result.flaky
    )
    lines += _question_list("Only in baseline", result.only_in_baseline)
    lines += _question_list("Only in candidate", result.only_in_candidate)

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def render_json(result: Comparison) -> str:
    data = result.model_dump()
    data["regressions"] = [m.name for m in result.regressions]
    return json.dumps(data, indent=2) + "\n"
