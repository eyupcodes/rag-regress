import json
from pathlib import Path

from typer.testing import CliRunner

from rag_regress.cli import app
from rag_regress.compare import Comparison, MetricDelta
from rag_regress.report import MARKER, render_json, render_markdown


def delta(name: str, base: float, cand: float, regressed: bool) -> MetricDelta:
    return MetricDelta(
        name=name,
        baseline=base,
        candidate=cand,
        delta=cand - base,
        tolerance=0.02,
        regressed=regressed,
    )


def comparison(**overrides) -> Comparison:
    fields = dict(
        passed=True,
        hit_metric="hit@5",
        metrics=[delta("hit@5", 0.9, 0.9, False), delta("mrr@5", 0.7, 0.75, False)],
        newly_failing=[],
        newly_passing=[],
        newly_erroring=[],
        only_in_baseline=[],
        only_in_candidate=[],
        warnings=[],
    )
    fields.update(overrides)
    return Comparison(**fields)


def test_pass_report_snapshot() -> None:
    assert render_markdown(comparison()) == (
        f"{MARKER}\n"
        "✅ **rag-regress: pass**. No gated metric dropped beyond its tolerance.\n"
        "\n"
        "| Metric | Baseline | Candidate | Δ | Tolerance | |\n"
        "|---|---:|---:|---:|---:|:-:|\n"
        "| `hit@5` | 0.900 | 0.900 | +0.000 | 0.020 |  |\n"
        "| `mrr@5` | 0.700 | 0.750 | +0.050 | 0.020 | ⬆️ |\n"
    )


def test_regression_report_puts_the_problem_first() -> None:
    result = comparison(
        passed=False,
        metrics=[delta("hit@5", 0.9, 0.9, False), delta("recall@5", 0.8, 0.6, True)],
        newly_failing=["q2", "q7"],
        newly_erroring=["q3"],
        warnings=["the golden set changed between the runs"],
    )

    text = render_markdown(result, title="docs-bot")

    lines = text.splitlines()
    assert lines[1] == "❌ **docs-bot: regression** in `recall@5`."
    assert lines[3] == "> ⚠️ the golden set changed between the runs"
    first_row = next(line for line in lines if line.startswith("| `"))
    assert first_row.startswith("| `recall@5` | 0.800 | 0.600 | -0.200")
    assert first_row.endswith("| ❌ |")
    assert "**Newly failing (hit@5)** (2): `q2`, `q7`" in text
    assert "**Newly erroring** (1): `q3`" in text
    assert text.endswith("`q3`\n")


def test_long_question_lists_collapse() -> None:
    ids = [f"q{i}" for i in range(11)]

    text = render_markdown(comparison(newly_passing=ids, only_in_candidate=["q99"]))

    assert "<summary><b>Newly passing (hit@5)</b> (11)</summary>" in text
    assert "**Only in candidate** (1): `q99`" in text


def test_json_lists_regression_names() -> None:
    result = comparison(passed=False, metrics=[delta("hit@5", 1, 0.5, True)])

    data = json.loads(render_json(result))

    assert data["passed"] is False
    assert data["regressions"] == ["hit@5"]
    assert data["metrics"][0]["delta"] == -0.5


def test_cli_writes_both_reports(tmp_path: Path) -> None:
    from tests.test_compare import ALL_RIGHT, make_run

    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    make_run(ALL_RIGHT).save(base)
    make_run({**ALL_RIGHT, "q2": ["x"]}).save(cand)
    md, js = tmp_path / "out" / "report.md", tmp_path / "out" / "report.json"

    result = CliRunner().invoke(
        app,
        ["compare", "--baseline", str(base), "--candidate", str(cand)]
        + ["--report-md", str(md), "--report-json", str(js)],
    )

    assert result.exit_code == 1
    assert md.read_text(encoding="utf-8").startswith(MARKER)
    assert json.loads(js.read_text(encoding="utf-8"))["passed"] is False
