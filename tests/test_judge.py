import json
from datetime import UTC
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from rag_regress.cli import app
from rag_regress.golden import GoldenItem
from rag_regress.judge import (
    Judge,
    JudgeCache,
    JudgeError,
    OpenAICompatibleClient,
    judge_from_env,
    parse_score,
)
from rag_regress.runner import run_pipeline


class FakeClient:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ('{"score": 0.75, "reason": "ok"}', 0.75),
        ('Sure! {"score": 1, "reason": "x"} hope that helps', 1.0),
        ('{"score": 0}', 0.0),
    ],
)
def test_parse_score(reply: str, score: float) -> None:
    assert parse_score(reply) == score


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("no json here", "no JSON object"),
        ('{"reason": "x"}', "not JSON with a 'score'"),
        ('{"score": "high"}', "not a number"),
        ('{"score": true}', "not a number"),
        ('{"score": 1.5}', "outside"),
        ("{broken", "no JSON object"),
        ("{not: json}", "not JSON"),
    ],
)
def test_parse_score_rejects_bad_replies(reply: str, message: str) -> None:
    with pytest.raises(JudgeError, match=message):
        parse_score(reply)


def test_relevance_only_without_contexts_and_both_with() -> None:
    client = FakeClient('{"score": 0.9}')
    judge = Judge(client, "m")

    assert judge.metrics("q?", "a.", None) == {"judge_relevance": 0.9}
    scores = judge.metrics("q?", "a.", ["passage one", "passage two"])
    assert set(scores) == {"judge_relevance", "judge_faithfulness"}
    assert "[2] passage two" in client.prompts[-1]


def test_malformed_reply_is_retried_then_fails() -> None:
    assert Judge(FakeClient("nope", '{"score": 0.5}'), "m")._score("p") == 0.5
    with pytest.raises(JudgeError, match="after 2 tries"):
        Judge(FakeClient("nope"), "m")._score("p")


def test_cache_makes_reruns_identical_and_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "judge.json"
    first = FakeClient('{"score": 0.6}')
    assert Judge(first, "m", cache=JudgeCache(path))._score("p") == 0.6

    second = FakeClient('{"score": 0.1}')
    assert Judge(second, "m", cache=JudgeCache(path))._score("p") == 0.6
    assert second.prompts == []
    assert Judge(second, "other-model", cache=JudgeCache(path))._score("p") == 0.1


def test_runner_adds_judge_scores_and_records_judge_failures() -> None:
    items = [
        GoldenItem(id="q1", query="q1", relevant_ids=("a",)),
        GoldenItem(id="q2", query="q2", relevant_ids=("a",)),
    ]

    def pipeline(query: str) -> dict:
        return {"retrieved_ids": ["a"], "answer": f"ans {query}", "contexts": ["ctx"]}

    class PickyClient:
        def complete(self, prompt: str) -> str:
            return "garbage" if "ans q2" in prompt else '{"score": 0.8}'

    run = run_pipeline(items, pipeline, ks=[1], judge=Judge(PickyClient(), "m"))

    assert run.questions[0].metrics["judge_faithfulness"] == 0.8
    assert run.questions[1].judge_error.startswith("JudgeError")
    assert "judge_relevance" not in run.questions[1].metrics
    assert run.questions[1].error is None
    assert run.aggregates["judge_relevance"] == 0.8  # averaged over judged items


def test_runner_skips_judge_without_answer_and_validates_contexts() -> None:
    item = GoldenItem(id="q", query="q", relevant_ids=("a",))
    judge = Judge(FakeClient('{"score": 1}'), "m")

    no_answer = run_pipeline([item], lambda q: ["a"], ks=[1], judge=judge)
    assert "judge_relevance" not in no_answer.questions[0].metrics

    bad = run_pipeline(
        [item],
        lambda q: {"retrieved_ids": ["a"], "answer": "x", "contexts": "one string"},
        ks=[1],
        judge=judge,
    )
    assert "contexts must be a list of strings" in bad.questions[0].error


def test_failed_judge_in_every_repeat_leaves_no_fake_sample() -> None:
    item = GoldenItem(id="q", query="q", relevant_ids=("a",))
    run = run_pipeline(
        [item],
        lambda q: {"retrieved_ids": ["a"], "answer": "x"},
        ks=[1],
        repeats=2,
        judge=Judge(FakeClient("garbage"), "m"),
    )

    assert "judge_relevance" not in run.aggregates
    assert run.questions[0].judge_error


def test_openai_compatible_client_sends_a_chat_completion() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        reply = {"choices": [{"message": {"content": '{"score": 1}'}}]}
        return httpx.Response(200, json=reply)

    client = OpenAICompatibleClient(
        "judge-1",
        api_key="k",
        base_url="https://llm.example/v1/",
        transport=httpx.MockTransport(handler),
    )

    assert client.complete("hello") == '{"score": 1}'
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["messages"][0]["content"] == "hello"


def test_judge_from_env_needs_a_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_REGRESS_JUDGE_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="API_KEY"):
        judge_from_env("m")

    monkeypatch.setenv("RAG_REGRESS_JUDGE_API_KEY", "k")
    judge = judge_from_env("m", cache_path=tmp_path / "c.json")
    assert judge.model == "m" and judge.cache is not None
    assert judge_from_env("m").cache is None


def test_cli_without_api_key_is_a_usage_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_REGRESS_JUDGE_API_KEY", raising=False)
    (tmp_path / "rag.py").write_text("def f(q):\n    return ['d']\n", encoding="utf-8")
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"id": "q", "query": "x", "relevant_ids": ["d"]}\n')
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["run", "--golden", str(golden), "--pipeline", "rag:f", "--out", "o.json"]
        + ["--judge-model", "gpt-x"],
    )

    assert result.exit_code == 2
    assert "API_KEY" in result.output


@pytest.mark.parametrize(
    "reply",
    [
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": ["parts"]}}]},
        {"choices": [{"message": {"content": []}}]},
        {"choices": [{"message": {"content": [{"type": "image_url"}]}}]},
        {"error": {"message": "overloaded"}},
    ],
)
def test_openai_compatible_client_rejects_malformed_reply(reply: dict) -> None:
    client = OpenAICompatibleClient(
        "judge-1",
        api_key="k",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)),
    )

    with pytest.raises(JudgeError, match="no completion"):
        client.complete("hello")


def test_openai_compatible_client_joins_typed_content_parts() -> None:
    parts = [
        {"type": "text", "text": '{"score": '},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "1}"},
    ]
    reply = {"choices": [{"message": {"content": parts}}]}
    client = OpenAICompatibleClient(
        "judge-1",
        api_key="k",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)),
    )

    assert client.complete("hello") == '{"score": 1}'


def _retry_client(
    handler: object, slept: list[float], **kwargs: object
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        "judge-1",
        api_key="k",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        sleep=slept.append,
        **kwargs,  # type: ignore[arg-type]
    )


_OK = {"choices": [{"message": {"content": '{"score": 1}'}}]}


def test_client_retries_rate_limit_then_succeeds() -> None:
    replies = [httpx.Response(429), httpx.Response(503), httpx.Response(200, json=_OK)]
    slept: list[float] = []
    client = _retry_client(lambda request: replies.pop(0), slept)

    assert client.complete("hello") == '{"score": 1}'
    assert slept == [1.0, 2.0]


def test_client_honours_retry_after_and_caps_it() -> None:
    replies = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(429, headers={"Retry-After": "900"}),
        httpx.Response(200, json=_OK),
    ]
    slept: list[float] = []
    client = _retry_client(lambda request: replies.pop(0), slept, max_retries=2)

    client.complete("hello")
    assert slept == [7.0, 30.0]


def test_client_honours_retry_after_http_date(monkeypatch) -> None:
    from datetime import datetime, timedelta
    from email.utils import format_datetime

    when = datetime.now(UTC) + timedelta(seconds=20)
    replies = [
        httpx.Response(
            429, headers={"Retry-After": format_datetime(when, usegmt=True)}
        ),
        httpx.Response(200, json=_OK),
    ]
    slept: list[float] = []
    client = _retry_client(lambda request: replies.pop(0), slept)

    client.complete("hello")
    assert 15.0 <= slept[0] <= 20.0


def test_client_treats_past_retry_after_date_as_no_wait() -> None:
    replies = [
        httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        httpx.Response(200, json=_OK),
    ]
    slept: list[float] = []
    client = _retry_client(lambda request: replies.pop(0), slept)

    client.complete("hello")
    assert slept == [0.0]


@pytest.mark.parametrize("header", ["soon", "-3", "nan", "inf"])
def test_client_ignores_unusable_retry_after(header: str) -> None:
    replies = [httpx.Response(429, headers={"Retry-After": header})]
    replies.append(httpx.Response(200, json=_OK))
    slept: list[float] = []
    client = _retry_client(lambda request: replies.pop(0), slept)

    client.complete("hello")
    assert slept == [1.0]


def test_client_gives_up_after_max_retries_with_server_message() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="model overloaded")

    slept: list[float] = []
    client = _retry_client(handler, slept, max_retries=2)

    with pytest.raises(JudgeError, match="HTTP 503: model overloaded"):
        client.complete("hello")
    assert len(calls) == 3
    assert len(slept) == 2


def test_client_does_not_retry_client_errors() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    slept: list[float] = []
    client = _retry_client(handler, slept)

    with pytest.raises(JudgeError, match="HTTP 401"):
        client.complete("hello")
    assert len(calls) == 1
    assert slept == []


def test_client_retries_connection_errors() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=_OK)

    slept: list[float] = []
    client = _retry_client(handler, slept)

    assert client.complete("hello") == '{"score": 1}'
    assert slept == [1.0]


def test_client_reraises_connection_error_when_retries_run_out() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    slept: list[float] = []
    client = _retry_client(handler, slept, max_retries=1)

    with pytest.raises(httpx.ConnectError):
        client.complete("hello")
    assert len(slept) == 1
