import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rag_regress.cli import app
from rag_regress.golden import GoldenItem
from rag_regress.runner import (
    PipelineLoadError,
    RunResult,
    file_sha256,
    load_pipeline,
    run_pipeline,
)

ITEMS = [
    GoldenItem(id="q1", query="alpha", relevant_ids=("a",), tags=("t",)),
    GoldenItem(id="q2", query="beta", relevant_ids=("b",), reference_answer="Beta"),
]


def ticking_clock(step: float = 0.01):
    counter = itertools.count()
    return lambda: next(counter) * step


@pytest.mark.parametrize(
    "output",
    [
        ["a", "b"],
        ("a", "b"),
        {"retrieved_ids": ["a", "b"], "answer": "Beta"},
        SimpleNamespace(retrieved_ids=["a", "b"], answer="Beta"),
    ],
)
def test_accepts_list_mapping_and_object_outputs(output: object) -> None:
    result = run_pipeline(ITEMS, lambda q: output, ks=[1, 2], clock=ticking_clock())

    assert result.errors == 0
    assert result.questions[0].metrics["hit@1"] == 1.0
    assert result.questions[1].metrics["hit@2"] == 1.0
    assert result.aggregates["hit@2"] == 1.0
    assert result.questions[0].tags == ("t",)
    assert result.questions[0].latency_ms == pytest.approx(10.0)


def test_answer_metrics_only_average_items_with_references() -> None:
    result = run_pipeline(
        ITEMS, lambda q: {"retrieved_ids": [q[0]], "answer": "beta"}, ks=[1]
    )

    assert result.aggregates["exact_match"] == 1.0
    assert "exact_match" not in result.questions[0].metrics


def test_errors_are_recorded_and_scored_as_empty_retrieval() -> None:
    def flaky(query: str) -> list[str]:
        if query == "beta":
            raise RuntimeError("index offline")
        return ["a"]

    result = run_pipeline(ITEMS, flaky, ks=[1])

    assert result.errors == 1
    failed = result.questions[1]
    assert failed.error == "RuntimeError: index offline"
    assert failed.retrieved_ids == []
    assert failed.metrics["hit@1"] == 0.0
    assert result.aggregates["hit@1"] == 0.5


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (42, "unsupported pipeline output: int"),
        ("a,b", "unsupported pipeline output: str"),
        ({"retrieved_ids": "a"}, "list of strings"),
        ([1, 2], "list of strings"),
        ({"retrieved_ids": ["a"], "answer": 3}, "answer must be a string"),
    ],
)
def test_unusable_outputs_become_errors(output: object, message: str) -> None:
    result = run_pipeline(ITEMS[:1], lambda q: output, ks=[1])

    assert message in (result.questions[0].error or "")


def test_run_round_trips_through_json(tmp_path: Path) -> None:
    result = run_pipeline(ITEMS, lambda q: ["a"], ks=[1], golden_sha256="abc")
    path = tmp_path / "run.json"
    result.save(path)

    assert RunResult.load(path) == result


def test_empty_golden_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_pipeline([], lambda q: [])


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("no_colon", "expected 'package.module:function'"),
        ("missing_module_xyz:f", "cannot import"),
        ("json:not_there", "has no not_there"),
        ("json:__doc__", "is not callable"),
    ],
)
def test_load_pipeline_errors(reference: str, message: str) -> None:
    with pytest.raises(PipelineLoadError, match=message):
        load_pipeline(reference)


def test_load_pipeline_follows_dotted_attributes() -> None:
    assert load_pipeline("json:JSONDecoder.decode") is not None


def test_cli_run_writes_a_run_file(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "my_rag.py").write_text(
        "def search(query):\n    return {'retrieved_ids': ['a'], 'answer': 'Beta'}\n",
        encoding="utf-8",
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(json.dumps(i.model_dump()) for i in ITEMS), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    out = tmp_path / "runs" / "run.json"
    result = CliRunner().invoke(
        app,
        ["run", "--golden", str(golden), "--pipeline", "my_rag:search"]
        + ["--out", str(out), "--k", "1", "--k", "5"],
    )

    assert result.exit_code == 0, result.output
    assert "2 questions, 0 errors" in result.output
    run = RunResult.load(out)
    assert run.ks == [1, 5]
    assert run.golden_sha256 and len(run.golden_sha256) == 64


@pytest.mark.parametrize("bad_k", ["0", "-1"])
def test_cli_run_rejects_non_positive_k(tmp_path: Path, bad_k: str) -> None:
    out = tmp_path / "run.json"
    result = CliRunner().invoke(
        app,
        ["run", "--golden", "g.jsonl", "--pipeline", "m:f", "--out", str(out)]
        + ["--k", bad_k],
    )

    assert result.exit_code == 2
    assert "--k" in result.output
    assert "Traceback" not in result.output
    assert not out.exists()


@pytest.mark.parametrize(
    ("pipeline", "golden_text"),
    [
        ("my_rag:search", "not json\n"),
        ("nope:search", '{"id":"q","query":"x","relevant_ids":["a"]}'),
    ],
)
def test_cli_run_reports_setup_errors(
    tmp_path: Path, monkeypatch, pipeline: str, golden_text: str
) -> None:
    (tmp_path / "my_rag.py").write_text("def search(q):\n    return []\n")
    golden = tmp_path / "golden.jsonl"
    golden.write_text(golden_text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["run", "--golden", str(golden), "--pipeline", pipeline, "--out", "o.json"]
    )

    assert result.exit_code == 2
    assert "error:" in result.output


def test_golden_fingerprint_ignores_line_endings(tmp_path: Path) -> None:
    unix, windows = tmp_path / "u.jsonl", tmp_path / "w.jsonl"
    unix.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    windows.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')

    assert file_sha256(unix) == file_sha256(windows)
    windows.write_bytes(b'{"a": 1}\r\n{"b": 3}\r\n')
    assert file_sha256(unix) != file_sha256(windows)


def test_cli_run_reports_unreadable_golden_set_with_exit_code_2(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["run", "--golden", str(tmp_path), "--pipeline", "my_rag:search"]
        + ["--out", str(tmp_path / "run.json")],
    )

    assert result.exit_code == 2
    assert "error:" in result.output
    assert "cannot read" in result.output
