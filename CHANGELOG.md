# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The OpenAI-compatible judge client retries `429`/`5xx` replies and connection errors twice with exponential backoff (honouring a numeric `Retry-After`, capped at 30 s) instead of dropping the question's judge scores on the first hiccup; an HTTP error now reports the server's message in `judge_error` (#53).

### Fixed

- The judge client honours `Retry-After` given as an HTTP date (a past date means no extra wait), not only as seconds, so it no longer retries before the server asked (#57).
- `rag-regress compare` and `run` exit 2 before doing any work when `--report-md`, `--report-json` or `--out` is the same file as an input (`--baseline`, `--candidate`, `--golden`) or as another output, instead of overwriting it (#55).
- `rag-regress run` reports a pipeline module that raises while importing (a `SyntaxError`, a failing module-level statement) as `error: ...` with exit code 2 instead of a traceback with exit code 1, which CI reads as a regression (#51).
- The OpenAI-compatible judge client joins `message.content` returned as typed text parts (`[{"type": "text", "text": ...}]`) instead of rejecting a valid verdict as "no completion" (#47).

### Changed

- Dependabot proposes monthly, grouped updates for GitHub Actions and pip dependencies (#45).
- `mypy` runs in strict mode (#43).

### Fixed

- A judge reply without a completion (empty `choices`, `null` or non-string content, error object) raises `JudgeError` instead of `KeyError`/`TypeError`, so the question is reported as a judge failure (#43).
- `rag-regress run --out` is checked before the pipeline runs, and `run`/`compare` report an unwritable `--out`, `--report-md`, `--report-json` or baseline as `error: cannot write ...` with exit code 2 instead of a traceback after the work is done (#41).
- The GitHub Action no longer fails the job when the pull request comment cannot be posted, for example on fork pull requests with a read-only token; it emits a warning and the report stays in the job summary (#39).
- Tag slices count a question once even if it lists the same tag twice (#35).
- Golden set files and run files with a UTF-8 BOM (Excel, Windows editors) now load instead of failing as invalid JSON (#31).

### Added

- `compare` reports the p50/p95 latency of both runs in `--report-md` and `--report-json` (`latency`), and `--max-latency-increase RATIO` fails the comparison when p95 latency grows by more than that ratio; it is off by default (#49).
- `compare` reports hit-rate slices per golden-set tag (CLI summary, `--report-md`, `--report-json`), worst drop first; they are informational and do not gate (#33).

- Golden set loader for JSONL files with per-line validation (#1).
- Deterministic metrics: hit, recall, precision, MRR and nDCG at k, exact match and token F1 (#2).
- `rag-regress run`: runs a `module:function` pipeline over the golden set and writes a JSON run file, recording per-question errors (#3).
- `rag-regress compare`: gates a candidate run against a baseline with per-metric tolerances, lists newly failing, passing and erroring questions, and can update the baseline when nothing regressed (#4).
- Markdown and JSON reports for `compare` (`--report-md`, `--report-json`), suited to PR comments (#5).
- Composite GitHub Action: runs the golden set, compares with the baseline, updates one PR comment and fails the job on regression (#6).
- `run --repeats N` for nondeterministic pipelines; `compare` widens tolerances to two standard errors of run-to-run noise and lists flaky questions (#8).
- Optional LLM-judge scores (`judge_relevance`, `judge_faithfulness`) via any OpenAI-compatible API, with a reply cache and per-question `judge_error` (#7).
- Fixed: the golden set fingerprint no longer changes between CRLF and LF checkouts (#6).
- Fixed: a golden set that is a directory, unreadable or not UTF-8 now fails `run` with `error:` and exit code 2 instead of a traceback (#19).
- Fixed: `compare` on an invalid run file prints `error: <file>: not a valid run file (<reason>)` and exits 2, instead of a pydantic validation dump that did not name the file (#21).
- Fixed: `compare --tolerance-for` with a name that is not a metric in both runs now exits 2 naming it, instead of being silently ignored (#23).
- Fixed: `run --k 0` or a negative `--k` is rejected as a usage error (exit 2) instead of crashing with a traceback (#25).
- Fixed: golden set lines are split on `\n` only, so a query or answer containing U+2028, U+0085 or similar characters loads instead of failing with `invalid JSON` (#27).
- Fixed: `compare` rejects NaN tolerances (exit 2) instead of silently passing every comparison (#29).
