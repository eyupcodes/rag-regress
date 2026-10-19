"""Optional LLM-as-judge answer metrics.

Two scores in [0, 1], added only when a judge is configured and the pipeline
returned an answer:

- ``judge_relevance``: does the answer address the question?
- ``judge_faithfulness``: is every claim in the answer supported by the
  retrieved passages? Needs the pipeline to return ``contexts`` (the passage
  texts), so it is skipped otherwise.

The judge must reply with strict JSON. A malformed reply is retried, then the
question gets a ``judge_error`` and no judge scores; it is never scored 0 by
accident. Replies are cached on disk by (model, prompt), so re-running an
unchanged question returns the same score. Scores are still not comparable
across judge models or prompt versions, which is why the deterministic
metrics remain the default CI gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

# The cache key includes the full prompt text, so editing a prompt below
# invalidates its cached replies automatically.
RELEVANCE_PROMPT = """You grade a question-answering system.
Question:
{query}

Answer:
{answer}

Does the answer directly address the question? Ignore whether it is true.
Reply with JSON only: {{"score": <number from 0 to 1>, "reason": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """You grade a question-answering system.
Retrieved passages:
{contexts}

Answer:
{answer}

What fraction of the answer's claims is supported by the passages above?
Use 1 if every claim is supported and 0 if none is.
Reply with JSON only: {{"score": <number from 0 to 1>, "reason": "<one sentence>"}}"""


class JudgeError(RuntimeError):
    """The judge could not produce a valid score."""


class JudgeClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAICompatibleClient:
    """Chat completions over HTTP; works with OpenAI and compatible servers."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        transport: object | None = None,
    ) -> None:
        import httpx  # optional dependency: pip install "rag-regress[judge]"

        self.model = model
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def complete(self, prompt: str) -> str:
        response = self._http.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class JudgeCache:
    """Replies keyed by (model, prompt), stored as one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, str] = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {}
        )

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str) -> str | None:
        return self._data.get(self.key(model, prompt))

    def put(self, model: str, prompt: str, reply: str) -> None:
        self._data[self.key(model, prompt)] = reply
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1), encoding="utf-8")


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_score(reply: str) -> float:
    match = _JSON_OBJECT.search(reply)
    if not match:
        raise JudgeError("reply contains no JSON object")
    try:
        score = json.loads(match.group(0))["score"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise JudgeError("reply is not JSON with a 'score'") from None
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise JudgeError("score is not a number")
    if not 0.0 <= score <= 1.0:
        raise JudgeError(f"score {score} is outside [0, 1]")
    return float(score)


class Judge:
    def __init__(
        self,
        client: JudgeClient,
        model: str,
        *,
        cache: JudgeCache | None = None,
        retries: int = 1,
    ) -> None:
        self.client, self.model, self.cache, self.retries = (
            client,
            model,
            cache,
            retries,
        )

    def _score(self, prompt: str) -> float:
        cached = self.cache.get(self.model, prompt) if self.cache else None
        if cached is not None:
            return parse_score(cached)
        last: JudgeError | None = None
        for _ in range(self.retries + 1):
            reply = self.client.complete(prompt)
            try:
                score = parse_score(reply)
            except JudgeError as exc:
                last = exc
                continue
            if self.cache:
                self.cache.put(self.model, prompt, reply)
            return score
        raise JudgeError(f"no valid score after {self.retries + 1} tries: {last}")

    def metrics(
        self, query: str, answer: str, contexts: Sequence[str] | None
    ) -> dict[str, float]:
        scores = {
            "judge_relevance": self._score(
                RELEVANCE_PROMPT.format(query=query, answer=answer)
            )
        }
        if contexts:
            passages = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
            scores["judge_faithfulness"] = self._score(
                FAITHFULNESS_PROMPT.format(contexts=passages, answer=answer)
            )
        return scores


def judge_from_env(
    model: str, *, base_url: str | None = None, cache_path: str | Path | None = None
) -> Judge:
    """Build a judge for the CLI; the API key comes from the environment."""
    api_key = os.environ.get("RAG_REGRESS_JUDGE_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not api_key:
        raise JudgeError("set RAG_REGRESS_JUDGE_API_KEY or OPENAI_API_KEY")
    client = OpenAICompatibleClient(
        model,
        api_key=api_key,
        base_url=base_url
        or os.environ.get("RAG_REGRESS_JUDGE_BASE_URL", "https://api.openai.com/v1"),
    )
    cache = JudgeCache(cache_path) if cache_path else None
    return Judge(client, model, cache=cache)
