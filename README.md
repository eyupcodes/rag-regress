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

## Gate a change in CI

Commit a baseline once, then compare every change against it:

```bash
rag-regress run --golden examples/golden.jsonl     --pipeline examples.toy_rag:search_without_bm25 --out runs/candidate.json
rag-regress compare --baseline baseline.json --candidate runs/candidate.json     --metric hit@5 --metric recall@5
```

```
FAIL: quality regressed
  hit@5: 1.000 -> 0.750 (-0.250, tolerance 0.020)
  recall@5: 1.000 -> 0.750 (-0.250, tolerance 0.020)
  newly failing (hit@10): q2
```

| Option | Meaning |
|---|---|
| `--tolerance 0.02` | Allowed absolute drop for every metric |
| `--tolerance-for recall@5=0.05` | Per-metric override, repeatable |
| `--metric hit@5` | Gate only these metrics (default: every metric in both runs) |
| `--update-baseline` | Overwrite the baseline with the candidate, but only if nothing regressed |

Add `--report-md report.md` to get a short Markdown report for a pull request comment: the verdict, a metric table with regressions on top, and the questions that changed. `--report-json report.json` writes the same comparison for other tools.

Exit codes: `0` pass, `1` regression, `2` bad input. A warning is printed when the golden set changed between the two runs.

### Nondeterministic pipelines

If your pipeline samples, or uses an approximate index, the same question can pass on one run and fail on the next. Run every question several times:

```bash
rag-regress run --golden eval/golden.jsonl --pipeline myapp.rag:answer \
    --out runs/candidate.json --repeats 10
```

The run then stores one average per repeat. When both runs have repeats, `compare` widens each metric's allowed drop to two standard errors of that run-to-run noise, if that is larger than the tolerance. It lists questions whose result varies between repeats as **flaky**.

On a simulated pipeline with 40 questions and 10 repeats (each question right 80% of the time), comparing two runs of the same pipeline raised a false alarm in 6 of 200 trials (3%). A real drop from 0.8 to 0.7 was caught in 175 of 200 (87.5%). More questions or more repeats narrow the noise band.

## GitHub Action

```yaml
on: pull_request

permissions:
  contents: read
  pull-requests: write   # for the report comment

jobs:
  rag-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - run: pip install -e .          # your project, so the pipeline imports
      - uses: eyupcodes/rag-regress@main
        with:
          golden: eval/golden.jsonl
          pipeline: myapp.rag:answer
          baseline: eval/baseline.json
          compare-args: --metric hit@5 --metric recall@5 --tolerance 0.03
```

The action runs the golden set and compares the result with the committed baseline. It writes the report to the job summary and keeps a single comment on the pull request up to date, and it fails the job when quality drops. Set `comment: "false"` to skip the comment. Outputs: `passed` and `report`. This repository runs the action on its own example in CI, once on the healthy pipeline (it must pass) and once on a broken one (it must fail).

To accept a change in quality deliberately, regenerate the baseline locally with `rag-regress run ... --out eval/baseline.json` and commit it, so the change is visible in review.

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
