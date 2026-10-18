"""Deterministic per-question metrics: no network, no randomness.

Retrieval metrics use binary relevance against the golden ``relevant_ids``.
A repeated id in the retrieved list counts once, at its first position, so a
pipeline cannot inflate a score by returning the same document twice.

Answer metrics compare normalised text: Unicode case folding, punctuation
removed, whitespace collapsed. Articles are not stripped (unlike SQuAD), so
the same rules hold for every language.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def dedupe(ids: Iterable[str]) -> list[str]:
    """Keep the first occurrence of every id, in order."""
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(doc_id in relevant for doc_id in retrieved[:k]))


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    found = sum(doc_id in relevant for doc_id in retrieved[:k])
    return found / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Relevant results in the top k, divided by k even if fewer came back."""
    found = sum(doc_id in relevant for doc_id in retrieved[:k])
    return found / k


def reciprocal_rank_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalise_answer(prediction) == normalise_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred = normalise_answer(prediction).split()
    ref = normalise_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def question_metrics(
    retrieved_ids: Iterable[str],
    relevant_ids: Iterable[str],
    *,
    answer: str | None = None,
    reference_answer: str | None = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """All metrics for one question, keyed like ``recall@5`` or ``token_f1``.

    Answer metrics appear only when the golden item has a reference answer;
    a pipeline that returns no answer then scores 0 on them.
    """
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must be positive integers")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    retrieved = dedupe(retrieved_ids)

    scores: dict[str, float] = {}
    for k in sorted(set(ks)):
        scores[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        scores[f"mrr@{k}"] = reciprocal_rank_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)

    if reference_answer is not None:
        produced = answer or ""
        scores["exact_match"] = exact_match(produced, reference_answer)
        scores["token_f1"] = token_f1(produced, reference_answer)
    return scores
