import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_regress.cli import app
from rag_regress.compare import compare_runs
from rag_regress.golden import GoldenItem
from rag_regress.report import render_markdown
from rag_regress.runner import RunResult, run_pipeline

GOLDEN = [
    GoldenItem(id=f"q{i}", query=f"q{i}", relevant_ids=(f"d{i}",)) for i in range(40)
]


def noisy(p_hit: float, seed: int):
    """Finds the right document with probability p_hit, like a sampled pipeline."""
    rng = random.Random(seed)

    def pipeline(query: str) -> list[str]:
        return [f"d{query[1:]}"] if rng.random() < p_hit else ["other"]

    return pipeline


def run(p_hit: float, seed: int, repeats: int = 10) -> RunResult:
    return run_pipeline(GOLDEN, noisy(p_hit, seed), ks=[1], repeats=repeats)


def test_repeats_average_metrics_and_keep_per_repeat_samples() -> None:
    result = run(0.8, seed=1)

    assert result.repeats == 10
    assert len(result.aggregate_samples["hit@1"]) == 10
    assert result.aggregates["hit@1"] == pytest.approx(
        sum(result.aggregate_samples["hit@1"]) / 10
    )
    assert all(0.0 <= q.metrics["hit@1"] <= 1.0 for q in result.questions)


@pytest.mark.parametrize("seed", range(5))
def test_noise_alone_is_not_a_regression(seed: int) -> None:
    result = compare_runs(run(0.8, seed=seed), run(0.8, seed=100 + seed), tolerance=0)

    assert result.passed, result.regressions
    assert result.metrics[0].noise > 0
    assert result.flaky  # with p=0.8 over 10 repeats some questions vary


def test_a_real_drop_is_still_caught() -> None:
    result = compare_runs(run(0.8, seed=1), run(0.5, seed=2), tolerance=0)

    assert not result.passed
    hit = result.metrics[0]
    assert hit.regressed and -hit.delta > hit.noise


def test_single_runs_keep_the_plain_tolerance() -> None:
    result = compare_runs(run(1.0, 1, repeats=1), run(1.0, 2, repeats=1))

    assert result.metrics[0].noise == 0.0
    assert result.metrics[0].tolerance == 0.02
    assert result.flaky == []


def test_first_error_is_kept_across_repeats() -> None:
    calls = {"n": 0}

    def sometimes_down(query: str) -> list[str]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise TimeoutError("slow")
        return ["d0"]

    result = run_pipeline(GOLDEN[:1], sometimes_down, ks=[1], repeats=3)

    assert result.questions[0].error == "TimeoutError: slow"
    assert result.questions[0].metrics["hit@1"] == pytest.approx(2 / 3)
    assert result.errors == 1


def test_invalid_repeats() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_pipeline(GOLDEN, lambda q: [], repeats=0)


def test_old_run_files_without_repeat_fields_still_load(tmp_path: Path) -> None:
    data = run(1.0, 1, repeats=1).model_dump()
    del data["repeats"], data["aggregate_samples"]
    path = tmp_path / "old.json"
    path.write_text(RunResult.model_validate(data).model_dump_json(), encoding="utf-8")

    assert RunResult.load(path).repeats == 1


def test_report_mentions_noise_and_flaky_questions() -> None:
    text = render_markdown(compare_runs(run(0.8, 1), run(0.8, 2), tolerance=0))

    assert "two standard errors" in text
    assert "Flaky (hit@1 varies between repeats)" in text


def test_cli_repeats_option(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "rag.py").write_text("def f(q):\n    return ['d']\n", encoding="utf-8")
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"id": "q", "query": "x", "relevant_ids": ["d"]}\n')
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "run.json"

    result = CliRunner().invoke(
        app,
        ["run", "--golden", str(golden), "--pipeline", "rag:f"]
        + ["--out", str(out), "--repeats", "3"],
    )

    assert result.exit_code == 0, result.output
    assert RunResult.load(out).repeats == 3
    bad = CliRunner().invoke(
        app,
        ["run", "--golden", str(golden), "--pipeline", "rag:f"]
        + ["--out", str(out), "--repeats", "0"],
    )
    assert bad.exit_code == 2


def test_run_file_with_utf8_bom_loads(tmp_path: Path) -> None:
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + run(1.0, 1).model_dump_json().encode("utf-8"))

    assert RunResult.load(path).repeats == 1
