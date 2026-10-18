"""Command line: ``rag-regress run`` measures a pipeline on a golden set."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from .golden import GoldenSetError, load_golden_set
from .metrics import DEFAULT_KS
from .runner import PipelineLoadError, file_sha256, load_pipeline, run_pipeline

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
) -> None:
    """Run PIPELINE on every golden question and write a run file."""
    # Resolve 'module:function' against the project being tested.
    sys.path.insert(0, str(Path.cwd()))
    try:
        items = load_golden_set(golden)
        fn = load_pipeline(pipeline)
    except (GoldenSetError, PipelineLoadError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    result = run_pipeline(
        items, fn, ks=k or DEFAULT_KS, golden_sha256=file_sha256(golden)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)

    headline = [
        m for m in ("hit@5", "recall@5", "mrr@10", "token_f1") if m in result.aggregates
    ]
    summary = ", ".join(f"{m}={result.aggregates[m]:.3f}" for m in headline)
    typer.echo(f"{len(result.questions)} questions, {result.errors} errors. {summary}")
    typer.echo(f"wrote {out}")
