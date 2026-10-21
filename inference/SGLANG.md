# SGLang configuration for the Sep 2 runs

This file records the server and client configuration behind the four reports in
`reports/`. The server launch commands are reconstructed from the `server_args`
blob each report captured from SGLang's `/get_server_info` endpoint. That blob
contains two views of the configuration:

- `server_args` (top level): every argument after SGLang resolved its defaults.
- `server_args.internal_states[0]._raw_input`: the arguments as passed on the
  command line, before SGLang filled in derived values.

Flags listed under "Launch command" are the values in `_raw_input` that differ
from SGLang's defaults. Values listed under "Resolved by SGLang" were not passed;
SGLang chose them at startup and they are recorded here because they affect the
result.

## Version

- SGLang: `0.5.19.dev20260825+g1fa32d50e1` (nightly build; recorded in every report
  as `server.version`).
- GPU configuration: tensor parallel size 2 on one node (`tp_size: 2`, `nnodes: 1`).
  The GPU model is not recorded in the reports.
- CUDA version: not recorded. The nightly wheel is published on both the CUDA 12.9
  and CUDA 13.0 indexes.

Install the same nightly (CUDA 13.0 index):

```sh
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  "sglang==0.5.19.dev20260825+g1fa32d50e1"
```

For CUDA 12.9, replace `cu130` with `cu129`. Both index pages listed
`sglang-0.5.19.dev20260825+g1fa32d50e1-cp310-cp310-linux_x86_64.whl` when this file
was written (2026-10-14). The SGLang install guide notes that 0.5.19 is the last
release line with CUDA 12 wheels.

## Launch commands

`$SGLANG_API_KEY` stands for the key the servers were started with. The reports
store it as `<REDACTED>`. The servers listened on `0.0.0.0:8000`; the benchmark
client connected over `127.0.0.1`.

### qwen38_bi: Qwen3.8-27B, deterministic inference on

Report: `reports/qwen38_bi_20260902T230833Z.md` and `.json`.

```sh
python -m sglang.launch_server \
  --model-path qwen/qwen3.8-27b \
  --tp-size 2 \
  --attention-backend triton \
  --enable-deterministic-inference \
  --disable-radix-cache \
  --random-seed 42 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$SGLANG_API_KEY"
```

Resolved by SGLang: `sampling_backend: pytorch`, `chunked_prefill_size: 8192`,
`dtype: auto`, `context_length: null` (model default; `max_req_input_len: 262138`),
`mem_fraction_static: 0.7657`, `disable_custom_all_reduce: true`,
`enforce_disable_flashinfer_allreduce_fusion: true`, `page_size: 1`,
`schedule_policy: fcfs`, CUDA graphs on (decode batch sizes up to 256).

The model path was passed in lower case (`qwen/qwen3.8-27b`); the Hugging Face
repository is `Qwen/Qwen3.8-27B`.

### qwen36_bi: Qwen3.6-27B, deterministic inference on

Report: `reports/qwen36_bi_20260902T214427Z.md` and `.json`.

```sh
python -m sglang.launch_server \
  --model-path qwen/qwen3.6-27b \
  --tp-size 2 \
  --attention-backend triton \
  --enable-deterministic-inference \
  --disable-radix-cache \
  --random-seed 42 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$SGLANG_API_KEY"
```

Resolved by SGLang: identical to qwen38_bi (`sampling_backend: pytorch`,
`chunked_prefill_size: 8192`, `mem_fraction_static: 0.7657`,
`disable_custom_all_reduce: true`, `max_req_input_len: 262138`).

### q38_dflash2: Qwen3.8-27B, DFlash speculative decoding, deterministic inference off

Report: `reports/q38_dflash2_20260902T190359Z.md` and `.json`.

```sh
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --tp-size 2 \
  --attention-backend triton \
  --disable-radix-cache \
  --random-seed 42 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$SGLANG_API_KEY"
```

Resolved by SGLang: `enable_deterministic_inference: false`,
`sampling_backend: flashinfer`, `chunked_prefill_size: 8192`,
`speculative_num_steps: 1`, `speculative_eagle_topk: 1`,
`speculative_draft_attention_backend: triton`, `speculative_draft_model_revision: main`,
`max_running_requests: 48`, `disable_custom_all_reduce: false`,
`max_req_input_len: 252775`.

### gpt120b_eagle3: gpt-oss-120b, EAGLE3 speculative decoding, deterministic inference off

Report: `reports/gpt120b_eagle3_20260902T212217Z.md` and `.json`.

```sh
python -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tp-size 2 \
  --attention-backend triton \
  --disable-radix-cache \
  --random-seed 42 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path nvidia/gpt-oss-120b-Eagle3-long-context \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$SGLANG_API_KEY"
```

Resolved by SGLang: `enable_deterministic_inference: false`,
`sampling_backend: flashinfer`, `chunked_prefill_size: 8192`, `dtype: bfloat16`,
`moe_runner_backend: triton_kernel`, `max_running_requests: 48`,
`max_req_input_len: 131066`. `--speculative-num-draft-tokens` was passed as 2 and
resolved to 4 (`speculative_num_steps + 1`).

## Benchmark command

The reports record the harness under its earlier file name, `determinism.py`, and
the argument `--round 100`, which argparse expands to `--rounds 100`. Recorded
`argv` for qwen38_bi:

```
determinism.py --name qwen38_bi --round 100 --port 8000 --reasoning-effort medium --api-key <REDACTED> --abort-all --control-max-tokens 1000 --temperature 0 --concurrency 40
```

Equivalent command with the harness in this directory:

```sh
uv run benchmark.py \
  --name qwen38_bi \
  --rounds 100 \
  --port 8000 \
  --reasoning-effort medium \
  --api-key "$SGLANG_API_KEY" \
  --abort-all \
  --control-max-tokens 1000 \
  --temperature 0 \
  --concurrency 40
```

The other three runs used the same arguments with a different `--name`
(`q38_dflash2`, `test_gpt120b_eagle3`, `test_qwen36_bi`). qwen36_bi used
`--rounds 10`.

`--port 8000` overrides the port in the default `--api-base`
(`http://127.0.0.1:30000`), so requests went to `http://127.0.0.1:8000/v1`.

## Sampling and run settings

From each report's `config` and captured `request_body`:

| Setting | Value |
| --- | --- |
| temperature | 0 |
| top_p | 1.0 |
| seed | 42 (sent in every request body) |
| reasoning_effort | medium |
| separate_reasoning | true |
| control max tokens | 1,000 |
| prompts | 8 (built-in set, `prompt_set_version` 1) |
| requests per round | 8 (one per prompt) |
| rounds | 100 (qwen36_bi: 10) |
| concurrency | 40 |
| batch order | shuffle (seeded from `--seed` and the round index) |
| control repeats | 3 |
| early cancel, targeted abort by rid, abort-all between phases | on |
| cache flush between phases | on |

Every control in all four reports used the full 1,000-token allowance
(`completion_tokens: 1000`), so each control is the first 1,000 tokens of the
answer and each measured request was capped at 1,000 tokens. Control lengths range
from 2,390 to 4,564 characters.

## Results

Numbers from each report's `summary`, checked against a recount from the
`measurements` array in the same JSON file. The recount matched in every case.

| Run | Model | Deterministic inference | Speculative decoding | Measured | Identical to control | Median first-difference index (chars) | 95% CI | Median normalized position |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen38_bi | Qwen3.8-27B | on | none | 800 | 800/800 | 4134.5 | 4134.5 to 4134.5 | 1.000 |
| qwen36_bi | Qwen3.6-27B | on | none | 80 | 80/80 | 3921.5 | 3921.5 to 3921.5 | 1.000 |
| q38_dflash2 | Qwen3.8-27B | off | DFlash | 800 | 0/800 | 1253 | 1253 to 1342.5 | 0.310 |
| gpt120b_eagle3 | gpt-oss-120b | off | EAGLE3 | 800 | 0/800 | 364 | 319.5 to 474.5 | 0.099 |

The median first-difference index is the median over prompts of each prompt's
median index of the first character that differs from the control. When a
response is identical, its index is the full control length, so for the two
identical runs the value is the median control length. The markdown reports print
this value rounded to an integer (4134 and 3922 for the two identical runs); the
JSON holds 4134.5 and 3921.5.

Where the first difference fell: q38_dflash2, 100 in reasoning text and 700 in
the answer; gpt120b_eagle3, 0 in reasoning and 800 in the answer. No run had
failed requests, and every control still reproduced when re-checked after the
rounds.

## Caveats

- Radix (prefix) cache was disabled in all four runs (`disable_radix_cache: true`).
  The harness refuses to run against a server with prefix caching on unless
  `--allow-radix-cache` is passed. A deployment that keeps the radix cache on is
  not the configuration measured here.
- Batch occupancy was not verified. The servers were launched without
  `--enable-metrics`, so `/metrics` was unavailable and every report carries the
  warning "Batch occupancy could not be verified". The harness asked for
  concurrency 40 and kept it filled with filler requests, but the batch size the
  server actually ran is not recorded. Launch with `--enable-metrics` to record it.
- Qwen3.6-27B ran 10 rounds (80 measured requests), not 100.
- The two speculative-decoding runs did not have `--enable-deterministic-inference`
  and used the flashinfer sampling backend. They measure speculative decoding
  without batch-invariant kernels; they are not a test of speculative decoding
  combined with deterministic inference. The EAGLE3 run is a different model
  (gpt-oss-120b).
- The two speculative runs also set `--mem-fraction-static 0.9`, and SGLang
  resolved `max_running_requests` to 48 for them; the deterministic runs had no
  `max_running_requests` limit. Concurrency 40 fits under both.
- Each control was capped at 1,000 tokens, so "identical" means the first 1,000
  tokens matched character for character.
- The reports were produced by an earlier revision of the harness (then named
  `determinism.py`). Its JSON has no `controls[].truncated` or
  `server.abort_baseline_chunks` fields, which the version in this directory
  writes. The later revision added truncated-control detection, a baseline for
  the abort self-check, an under-fill warning for batch occupancy, validation of
  `--requests-per-round` and `--control-repeats`, a connection-pool size fix, and
  redaction of the server's own API key at capture time. With
  `--control-max-tokens 1000` and every control using its full allowance, both
  revisions cap measured requests at 1,000 tokens.
- The GPU model and CUDA version are not recorded.
