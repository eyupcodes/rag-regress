"""A deliberately tiny keyword retriever, so the examples run without models.

Swap ``search`` for your own pipeline: it only has to take a query and
return retrieved document ids (and, optionally, an answer).
"""

import re

CORPUS = {
    "rag": "Retrieval-augmented generation grounds answers in retrieved documents.",
    "bm25": "BM25 ranks documents by term frequency and inverse document frequency.",
    "chunking": "Chunking splits long documents into passages before embedding.",
    "rerank": "A reranker reorders retrieved passages with a stronger, slower model.",
    "ankara": "Ankara is the capital of Türkiye.",
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def search(query: str) -> dict:
    q = _words(query)
    ranked = sorted(CORPUS, key=lambda doc_id: -len(q & _words(CORPUS[doc_id])))
    top = ranked[:3]
    return {"retrieved_ids": top, "answer": CORPUS[top[0]]}
