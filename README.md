# rag-regress

Catch quality regressions in a RAG pipeline before they merge.

You keep a small **golden set** of questions with the documents (and, optionally, the answers) they should produce. `rag-regress` runs your pipeline on it, measures retrieval and answer quality, and compares the result with a stored **baseline**. If a metric drops by more than your tolerance, the command exits non-zero, so your CI turns red. The report tells you which questions broke.

```
golden set ──► your pipeline ──► metrics ──► compare with baseline ──► pass / fail + report
```

It is meant for the change you are about to merge, such as a new chunk size, embedding model, prompt or reranker. It does not replace a full benchmark.

> Status: pre-alpha. The roadmap is in the [issues](https://github.com/eyupcodes/rag-regress/issues?q=label%3Aroadmap).

## Why

RAG quality degrades quietly. A chunking tweak that helps one query type can drop recall on another, and nobody notices until users do. Unit tests cannot see this, and full benchmarks are too slow and noisy to run on every pull request. A small, deterministic golden set checked on every PR is the missing middle.

## Quickstart

```bash
pip install -e .
rag-regress run --golden examples/golden.jsonl     --pipeline examples.toy_rag:search --out runs/candidate.json
```

```
4 questions, 0 errors. hit@5=1.000, recall@5=1.000, mrr@10=1.000, token_f1=1.000
```

`--pipeline` takes any callable as `package.module:function`, resolved from the current directory. It is called with each query and may return:

- a list of document ids: `["doc-3", "doc-7"]`
- a mapping: `{"retrieved_ids": [...], "answer": "..."}`
- an object with `retrieved_ids` and, optionally, `answer` attributes

If a question raises or returns something else, the error is recorded for that question and it scores as an empty retrieval. A crash counts as a regression; it does not abort the run. The run file holds per-question ids, answers, latency, errors and metrics, plus the averages and p50/p95 latency.

## Golden set format

One JSON object per line:

```json
{"id": "q1", "query": "What does RAG stand for?", "relevant_ids": ["doc-12"], "reference_answer": "Retrieval-augmented generation", "tags": ["basics"]}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique question id |
| `query` | yes | The question sent to your pipeline |
| `relevant_ids` | yes | Documents that should be retrieved (at least one) |
| `reference_answer` | no | Enables answer metrics |
| `tags` | no | For slicing results, e.g. by topic |

A malformed file fails with the file name and line number, so you fix the data before measuring anything.

## Metrics

For every question and every cut-off `k` (default 1, 3, 5, 10):

| Metric | Meaning |
|---|---|
| `hit@k` | At least one relevant document in the top k |
| `recall@k` | Share of the relevant documents found in the top k |
| `precision@k` | Relevant documents in the top k, divided by k |
| `mrr@k` | 1 / rank of the first relevant document (0 if none in the top k) |
| `ndcg@k` | Rank-aware score with binary relevance |
| `exact_match`, `token_f1` | Only when the item has a `reference_answer` |

A document returned twice counts once, at its first position. Answers are compared after Unicode case folding and punctuation removal; articles are not stripped, so the rules are the same in every language.

## Design principles

- **Deterministic first.** Retrieval metrics and lexical answer metrics need no API key and give the same number on every run. LLM judges are optional and come later.
- **Per-question, not just averages.** A stable mean can hide one question going from right to wrong. Reports list what changed.
- **Your pipeline, unchanged.** You point `rag-regress` at a Python callable. It does not wrap or rewrite your stack.

## License

MIT
