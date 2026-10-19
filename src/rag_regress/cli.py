"""Command line.

``rag-regress run`` measures a pipeline on a golden set; ``rag-regress compare``
gates a candidate run against a baseline. Exit codes: 0 pass, 1 regression,
2 usage or input error.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from .compare import DEFAULT_TOLERANCE, compare_runs, format_summary
from .golden import GoldenSetError, load_golden_set
from .judge import JudgeError, judge_from_env
from .metrics import DEFAULT_KS
from .report import render_json, render_markdown
from .runner import (
    PipelineLoadError,
    RunResult,
    file_sha256,
    load_pipeline,
    run_pipeline,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Catch quality regressions in RAG pipelines."""


@app.command()
def run(
    golden: Annotated[Path, typer.Option(help="Golden set (JSONL).")],
    pipeline: Annotated[
        str, typer.Option(help="Callable to test, as 'package.module:function'.")
    ],
    out: Annotated[Path, typer.Option(help="Where to write the run (JSON).")],
    k: Annotated[
        list[int] | None, typer.Option(help="Cut-off for @k metrics; repeatable.")
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            min=1, help="Run every question N times (nondeterministic pipelines)."
        ),
    ] = 1,
    judge_model: Annotated[
        str | None,
        typer.Option(help="Add LLM-judge scores with this model (needs an API key)."),
    ] = None,
    judge_base_url: Annotated[
        str | None, typer.Option(help="OpenAI-compatible endpoint for the judge.")
    ] = None,
    judge_cache: Annotated[
        Path, typer.Option(help="Cache of judge replies, so reruns score the same.")
    ] = Path(".rag-regress-cache/judge.json"),
) -> None:
    """Run PIPELINE on every golden question and write a run file."""
    # Resolve 'module:function' against the project being tested.
    sys.path.insert(0, str(Path.cwd()))
    try:
        items = load_golden_set(golden)
        fn = load_pipeline(pipeline)
        judge = (
            judge_from_env(judge_model, base_url=judge_base_url, cache_path=judge_cache)
            if judge_model
            else None
        )
    except (GoldenSetError, PipelineLoadError, JudgeError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    result = run_pipeline(
        items,
        fn,
        ks=k or DEFAULT_KS,
        repeats=repeats,
        golden_sha256=file_sha256(golden),
        judge=judge,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)

    headline = [
        m for m in ("hit@5", "recall@5", "mrr@10", "token_f1") if m in result.aggregates
    ]
    summary = ", ".join(f"{m}={result.aggregates[m]:.3f}" for m in headline)
    typer.echo(f"{len(result.questions)} questions, {result.errors} errors. {summary}")
    typer.echo(f"wrote {out}")


def _parse_overrides(values: list[str]) -> dict[str, float]:
    overrides = {}
    for value in values:
        name, sep, number = value.partition("=")
        try:
            overrides[name] = float(number)
        except ValueError:
            sep = ""
        if not sep or not name:
            raise typer.BadParameter(f"{value!r}: expected METRIC=VALUE")
    return overrides


@app.command()
def compare(
    baseline: Annotated[Path, typer.Option(help="Baseline run (JSON).")],
    candidate: Annotated[Path, typer.Option(help="Candidate run (JSON).")],
    tolerance: Annotated[
        float, typer.Option(help="Allowed drop for every metric (absolute).")
    ] = DEFAULT_TOLERANCE,
    tolerance_for: Annotated[
        list[str] | None,
        typer.Option(help="Per-metric override, e.g. recall@5=0.05; repeatable."),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option(help="Gate only these metrics; repeatable. Default: all."),
    ] = None,
    update_baseline: Annotated[
        bool,
        typer.Option(
            help="If nothing regressed, copy the candidate over the baseline."
        ),
    ] = False,
    report_md: Annotated[
        Path | None, typer.Option(help="Also write a Markdown report (PR comment).")
    ] = None,
    report_json: Annotated[
        Path | None, typer.Option(help="Also write the comparison as JSON.")
    ] = None,
) -> None:
    """Fail (exit 1) if CANDIDATE is worse than BASELINE beyond tolerance."""
    try:
        base, cand = RunResult.load(baseline), RunResult.load(candidate)
        result = compare_runs(
            base,
            cand,
            tolerance=tolerance,
            tolerances=_parse_overrides(tolerance_for or []),
            gate=metric,
        )
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    typer.echo(format_summary(result))
    for path, render in ((report_md, render_markdown), (report_json, render_json)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(result), encoding="utf-8")
    if not result.passed:
        if update_baseline:
            typer.echo("baseline not updated: the candidate regressed")
        raise typer.Exit(1)
    if update_baseline:
        shutil.copyfile(candidate, baseline)
        typer.echo(f"updated {baseline}")
