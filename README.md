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

## Design principles

- **Deterministic first.** Retrieval metrics and lexical answer metrics need no API key and give the same number on every run. LLM judges are optional and come later.
- **Per-question, not just averages.** A stable mean can hide one question going from right to wrong. Reports list what changed.
- **Your pipeline, unchanged.** You point `rag-regress` at a Python callable. It does not wrap or rewrite your stack.

## License

MIT
