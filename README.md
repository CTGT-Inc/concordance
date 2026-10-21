# Concordance

Measurement and reproduction package for one finding: on a batch-invariant
inference server, identical requests returned identical answers every time,
and a document-ingestion pipeline still produced different results run to
run, because the pipeline wrote per-run values into its prompts. Removing
those values made five runs of the same document commit the same result.

The package has four parts. Each directory has its own README.

| Directory | Contents |
|---|---|
| [`compare_calls/`](compare_calls/) | `compare_runs.py`: given captured model calls from two or more runs of any pipeline, groups identical requests, reports whether their outputs changed, finds near-identical requests and the lines that differ, and finds the first pipeline stage whose output diverged. Stdlib only. |
| [`toy_pipeline/`](toy_pipeline/) | A three-stage pipeline over a synthetic document that reproduces the effect against any OpenAI-compatible endpoint: a random id printed into the prompt diverges across runs, an aliased id is stable. Writes call captures in the format `compare_runs.py` reads. |
| [`inference/`](inference/) | The batch-invariant inference benchmark: `benchmark.py`, the SGLang launch commands and version, and the reports behind the inference numbers. |
| [`data/`](data/) | Stage-level hashes and counts from the production ingestion sweep, before and after the fix. No prompts, no captured calls, no document text. |

## The measurements

Inference. Qwen3.8-27B on SGLang `0.5.19.dev20260825` with
`--enable-deterministic-inference`: 800 of 800 replies at concurrency 40
identical to a batch-size-1 control, over the first 1,000 tokens of each
reply. Two speculative-decoding configurations run without deterministic
mode: 0 of 800. See [`inference/SGLANG.md`](inference/SGLANG.md).

Ingestion. Five runs of one document on that deployment, every model call
captured. Before the fix, every request that was byte-identical across runs
returned the same answer, and the committed result still differed in four of
four comparable runs. The requests that changed answers differed only in
values the pipeline generated: a hash of the per-upload UUID, the order of a
30-item list, and a registry that grew between runs. After aliasing the ids,
sorting the list and pinning the registry, all five runs committed the same
result. See [`data/ingestion/`](data/ingestion/).

Stable is not correct. Before the fix, one run's wrong classification of a
block of rules replayed identically 8 of 8 times. Determinism makes an answer
repeatable; it does not make it right.

## Reproduce

Requires a GPU host running SGLang in deterministic mode. The toy pipeline
is estimated to run in under ten minutes with a small model; the inference
benchmark takes longer and needs the 27B model to match the published numbers.

```bash
# 1. inference benchmark: see inference/SGLANG.md for the launch command
# 2. toy pipeline
cd toy_pipeline && ./demo.sh
```

`demo.sh` runs the pipeline three times with random ids and three times with
aliased ids, then runs `compare_runs.py` on each set and prints both reports.

## Scope

The engine that produced the production numbers is not part of this package.
Neither are its prompts or the captured calls from the sweep. The toy
pipeline reproduces the mechanism, not the production pipeline.
