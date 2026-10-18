import math

import pytest

from rag_regress.metrics import (
    dedupe,
    exact_match,
    ndcg_at_k,
    normalise_answer,
    question_metrics,
    token_f1,
)

RELEVANT = {"a", "c"}
RETRIEVED = ["x", "a", "y", "c", "z"]


def test_retrieval_metrics_on_hand_computed_case() -> None:
    m = question_metrics(RETRIEVED, RELEVANT, ks=[1, 2, 5])

    assert m["hit@1"] == 0.0
    assert m["hit@2"] == 1.0
    assert m["recall@2"] == 0.5
    assert m["recall@5"] == 1.0
    assert m["precision@2"] == 0.5
    assert m["precision@5"] == pytest.approx(2 / 5)
    assert m["mrr@1"] == 0.0
    assert m["mrr@5"] == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert m["ndcg@5"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)


def test_perfect_ranking_scores_one() -> None:
    m = question_metrics(["a", "c", "x"], RELEVANT, ks=[2])

    assert m["recall@2"] == m["precision@2"] == m["mrr@2"] == m["ndcg@2"] == 1.0


def test_empty_retrieval_scores_zero() -> None:
    m = question_metrics([], RELEVANT, ks=[3])

    assert set(m.values()) == {0.0}


def test_duplicates_count_once_at_first_position() -> None:
    assert dedupe(["a", "a", "b", "a"]) == ["a", "b"]
    m = question_metrics(["a", "a", "a"], RELEVANT, ks=[3])

    assert m["recall@3"] == 0.5
    assert m["precision@3"] == pytest.approx(1 / 3)


def test_precision_divides_by_k_when_fewer_results_come_back() -> None:
    assert question_metrics(["a"], RELEVANT, ks=[5])["precision@5"] == 0.2


def test_ndcg_ideal_is_capped_by_k() -> None:
    assert ndcg_at_k(["a"], {"a", "b", "c"}, 1) == 1.0


def test_answer_metrics_only_with_reference() -> None:
    assert "token_f1" not in question_metrics(["a"], RELEVANT, answer="x")
    m = question_metrics(["a"], RELEVANT, answer=None, reference_answer="Paris")

    assert m["exact_match"] == 0.0
    assert m["token_f1"] == 0.0


def test_normalisation_is_language_neutral() -> None:
    assert normalise_answer("  Hello,   WORLD! ") == "hello world"
    assert exact_match("İstanbul'da.", "i̇stanbul da") == 1.0
    assert exact_match("The answer", "answer") == 0.0


def test_token_f1() -> None:
    assert token_f1("the capital is Ankara", "Ankara") == pytest.approx(0.4)
    assert token_f1("", "") == 1.0
    assert token_f1("x", "") == 0.0
    assert token_f1("a b", "c d") == 0.0


@pytest.mark.parametrize(("ids", "ks"), [(["a"], []), (["a"], [0]), ([], [1])])
def test_invalid_arguments(ids: list[str], ks: list[int]) -> None:
    with pytest.raises(ValueError):
        question_metrics(["a"], ids, ks=ks)
