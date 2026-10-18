# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Golden set loader for JSONL files with per-line validation (#1).
- Deterministic metrics: hit, recall, precision, MRR and nDCG at k, exact match and token F1 (#2).
- `rag-regress run`: runs a `module:function` pipeline over the golden set and writes a JSON run file, recording per-question errors (#3).
- `rag-regress compare`: gates a candidate run against a baseline with per-metric tolerances, lists newly failing, passing and erroring questions, and can update the baseline when nothing regressed (#4).
- Markdown and JSON reports for `compare` (`--report-md`, `--report-json`), suited to PR comments (#5).
