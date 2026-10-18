from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_regress.cli import app
from rag_regress.compare import compare_runs, format_summary
from rag_regress.golden import GoldenItem
from rag_regress.runner import RunResult, run_pipeline

GOLDEN = [
    GoldenItem(id=f"q{i}", query=f"q{i}", relevant_ids=(f"d{i}",)) for i in range(1, 5)
]


def make_run(answers: dict[str, list[str] | Exception], sha: str = "s") -> RunResult:
    def pipeline(query: str) -> list[str]:
        out = answers[query]
        if isinstance(out, Exception):
            raise out
        return out

    return run_pipeline(GOLDEN, pipeline, ks=[1, 3], golden_sha256=sha)


ALL_RIGHT = {f"q{i}": [f"d{i}"] for i in range(1, 5)}


def test_identical_runs_pass() -> None:
    result = compare_runs(make_run(ALL_RIGHT), make_run(ALL_RIGHT))

    assert result.passed
    assert result.regressions == []
    assert result.hit_metric == "hit@3"
    assert format_summary(result) == "PASS: no metric regressed"


def test_one_question_breaking_fails_and_is_named() -> None:
    worse = {**ALL_RIGHT, "q2": ["zzz"]}

    result = compare_runs(make_run(ALL_RIGHT), make_run(worse))

    assert not result.passed
    assert result.newly_failing == ["q2"]
    names = {m.name for m in result.regressions}
    assert {"hit@1", "hit@3", "recall@3"} <= names
    summary = format_summary(result)
    assert summary.startswith("FAIL")
    assert "hit@3: 1.000 -> 0.750 (-0.250, tolerance 0.020)" in summary
    assert "newly failing (hit@3): q2" in summary


def test_tolerance_absorbs_small_drops() -> None:
    worse = {**ALL_RIGHT, "q2": ["zzz"]}

    assert compare_runs(make_run(ALL_RIGHT), make_run(worse), tolerance=0.25).passed
    result = compare_runs(
        make_run(ALL_RIGHT), make_run(worse), tolerance=0.25, tolerances={"hit@3": 0}
    )
    assert [m.name for m in result.regressions] == ["hit@3"]


def test_gate_restricts_metrics_and_rejects_unknown_ones() -> None:
    base = make_run(ALL_RIGHT)
    worse = make_run({**ALL_RIGHT, "q2": ["x", "x2", "d2"]})  # found, but at rank 3

    assert compare_runs(base, worse, gate=["hit@3", "recall@3"]).passed
    assert not compare_runs(base, worse, gate=["mrr@3"]).passed
    with pytest.raises(ValueError, match="not in both runs: token_f1"):
        compare_runs(base, worse, gate=["token_f1"])


def test_improvements_errors_and_question_set_changes_are_reported() -> None:
    base = make_run({**ALL_RIGHT, "q1": ["nope"]})
    cand = make_run({**ALL_RIGHT, "q3": RuntimeError("down")}, sha="other")
    cand.questions = [q for q in cand.questions if q.id != "q4"]
    cand.questions.append(cand.questions[0].model_copy(update={"id": "q9"}))

    result = compare_runs(base, cand, tolerance=1)

    assert result.newly_passing == ["q1"]
    assert result.newly_failing == ["q3"]
    assert result.newly_erroring == ["q3"]
    assert result.only_in_baseline == ["q4"]
    assert result.only_in_candidate == ["q9"]
    assert "golden set changed" in result.warnings[0]
    text = format_summary(result)
    for fragment in ("newly erroring: q3", "only in baseline: q4", "warning:"):
        assert fragment in text


def test_float_noise_does_not_count_as_a_regression() -> None:
    base, cand = make_run(ALL_RIGHT), make_run(ALL_RIGHT)
    cand.aggregates = {k: v - 1e-15 for k, v in cand.aggregates.items()}

    assert compare_runs(base, cand, tolerance=0).passed


def test_invalid_arguments() -> None:
    run = make_run(ALL_RIGHT)
    with pytest.raises(ValueError, match="negative"):
        compare_runs(run, run, tolerance=-0.1)
    other = run.model_copy(update={"ks": [5]})
    with pytest.raises(ValueError, match="share no @k"):
        compare_runs(run, other)


def save(run: RunResult, path: Path) -> Path:
    run.save(path)
    return path


def invoke(*args: str):
    return CliRunner().invoke(app, ["compare", *args])


def test_cli_exit_codes_and_update_baseline(tmp_path: Path) -> None:
    base = save(make_run(ALL_RIGHT), tmp_path / "base.json")
    good = save(make_run(ALL_RIGHT), tmp_path / "good.json")
    bad = save(make_run({**ALL_RIGHT, "q2": ["x"]}), tmp_path / "bad.json")
    original = base.read_text(encoding="utf-8")

    failed = invoke(
        "--baseline", str(base), "--candidate", str(bad), "--update-baseline"
    )
    assert failed.exit_code == 1
    assert "baseline not updated" in failed.output
    assert base.read_text(encoding="utf-8") == original

    passed = invoke(
        "--baseline", str(base), "--candidate", str(good), "--update-baseline"
    )
    assert passed.exit_code == 0, passed.output
    assert "updated" in passed.output
    assert base.read_text(encoding="utf-8") == good.read_text(encoding="utf-8")

    tolerated = invoke(
        "--baseline", str(base), "--candidate", str(bad), "--tolerance", "0.3"
    )
    assert tolerated.exit_code == 0
    gated = invoke(
        "--baseline",
        str(base),
        "--candidate",
        str(bad),
        "--tolerance",
        "0.3",
        "--tolerance-for",
        "hit@3=0",
        "--metric",
        "hit@3",
    )
    assert gated.exit_code == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--tolerance-for", "hit@3"],
        ["--tolerance-for", "hit@3=abc"],
        ["--metric", "nope@1"],
        ["--tolerance", "-1"],
    ],
)
def test_cli_usage_errors_exit_2(tmp_path: Path, extra: list[str]) -> None:
    base = save(make_run(ALL_RIGHT), tmp_path / "base.json")

    result = invoke("--baseline", str(base), "--candidate", str(base), *extra)

    assert result.exit_code == 2


def test_cli_missing_or_invalid_run_file_exits_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")

    assert (
        invoke("--baseline", str(tmp_path / "x"), "--candidate", str(bad)).exit_code
        == 2
    )
    assert invoke("--baseline", str(bad), "--candidate", str(bad)).exit_code == 2
