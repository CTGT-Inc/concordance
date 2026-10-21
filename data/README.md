---
license: apache-2.0
pretty_name: Concordance ingestion determinism stage tables
task_categories:
  - other
language:
  - en
size_categories:
  - n<1K
tags:
  - determinism
  - reproducibility
  - llm-inference
  - batch-invariance
  - sglang
  - document-ingestion
configs:
  - config_name: summary
    data_files: ingestion/summary.csv
  - config_name: before_fix
    data_files: ingestion/before_fix.csv
  - config_name: after_fix
    data_files: ingestion/after_fix.csv
  - config_name: repeated_requests_by_module
    data_files: ingestion/repeated_requests_by_module.csv
---

# Concordance ingestion determinism stage tables

## Dataset Summary

Stage hashes and counts from repeated runs of an LLM document-ingestion
pipeline over one document, before and after a fix that removed per-run
values from the pipeline's prompts.

Inference was batch-invariant: in both phases, every identical request that
recurred across runs returned an identical response (74 of 74 before the fix,
89 of 89 after). "Identical" means the full request body was byte-equal, with
no masking; the per-module counts in `ingestion/repeated_requests_by_module.csv`
sum to these totals. Before the fix, a further 8 recurring requests differed
only in values the pipeline generated, and all 8 returned a different answer.
Before the fix the committed result still differed between
runs (31 to 33 obligations, 4 distinct result hashes in 4 runs) because the
pipeline put an upload-derived asset id, an unsorted list and a growing topic
registry into its prompts. After the fix, all 5 runs committed the same 34
obligations with one masked result hash, `8f828e73`.

The dataset contains hashes and counts only. It contains no document text,
no prompts and no model outputs.

## Structure

| File | Rows | Content |
|---|---|---|
| `ingestion/before_fix.csv` | 28 | Runs 2-5 before the fix, one row per (run, stage) |
| `ingestion/after_fix.csv` | 35 | Runs 1-5 after the fix, one row per (run, stage) |
| `ingestion/summary.csv` | 10 | One row per run: committed obligations, result hash, validation violations, graph relationships, model calls, runtime, repeated-request counts |
| `ingestion/repeated_requests_by_module.csv` | 14 | Per phase and pipeline module: calls, identical requests repeated across runs, how many of those changed output |

Stages, in pipeline order: `parse`, `label`, `interpret`, `unitize`,
`extract`, `identity`, `committed_result`. Column definitions, the hash
method and the list of entropy sources are in `ingestion/README.md`.

Headline values:

| Measure | Before fix (runs 2-5) | After fix (runs 1-5) |
|---|---|---|
| Committed obligations | 32, 31, 33, 33 | 34 in all runs |
| Distinct committed-result hashes | 4 | 1 (`8f828e73`, per-upload ids masked) |
| Validation violations | 42, 36, 33, 39 | 34 in all runs |
| Graph relationships | 433, 418, 427, 433 | 383 in all runs |
| Model calls | 95, 91, 92, 91 | 91 in all runs |
| Identical requests repeated across runs | 74 | 89 |
| Of those, output changed | 0 | 0 |

## Collection

- Pipeline: a multi-stage policy-document ingestion pipeline that parses a
  PDF, labels and interprets its nodes with an LLM, groups rule units, and
  extracts obligations into a graph.
- Model: Qwen3.8-27B served by SGLang on a batch-invariant deployment
  (`--enable-deterministic-inference`), temperature 1.0, radix cache on. The
  client sent no seed.
- Runs: September 2026. Before-fix runs 2-5 and after-fix runs 1-5 against
  the same PDF. Before-fix run 1 was excluded because the topic registry
  carried keys from earlier runs.
- Each run's stage outputs were canonicalised (run metadata removed,
  generated ids replaced with source-position references, unordered lists
  sorted) and hashed with SHA-256. The masked hash also replaces per-upload
  ids with placeholders. Hashes are published as their first 8 hex characters.
- Every model call was captured and grouped by its full request body to
  count identical requests repeated across runs. The captures themselves are
  not published.
- Fix: asset ids derived from the per-upload UUID were replaced with
  upload-independent aliases (they appeared in 53 of 459 captured calls), the
  30-item `semantic_route_selections` list was sorted, and topic registry
  growth between runs (21 to 22 keys) was removed.

## Caveats

- **Stable is not correct.** After the fix the pipeline returns the same
  answer every time; it does not follow that the answer is right. Before the
  fix, run 3 classified a group of rules as background text and dropped an
  obligation that other runs produced. Replaying run 3's request returned the
  same wrong classification 8 times out of 8.
- **One document.** All runs use a single 9-page retail banking operations
  manual (17,780 characters). The results say nothing about variance across
  documents, document types or languages.
- **Small sample.** 4 before-fix runs and 5 after-fix runs.
- **Byte-level, not semantic.** The hashes measure whether outputs are
  identical. A differing hash does not say how much the meaning changed, and
  this release does not include a semantic-variance measure.
- **Truncated hashes.** 8 hex characters are enough to compare runs within
  this dataset. They are not suitable as content identifiers.
- **Deployment differs from the benchmark.** Production ran with the radix
  cache on. The inference benchmark in `inference/` ran with it off.
- **Document provenance.** The test document is a synthetic retail-banking
  operations manual used for evaluation. It is not
  included, and none of its text is quoted here.

## License

Apache License 2.0. See `LICENSE` at the repository root.
