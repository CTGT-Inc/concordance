# Batch-invariant inference benchmark

`benchmark.py` measures whether an OpenAI-compatible inference server returns the
same output for the same request when the request runs alone and when it runs in a
batch with other requests.

## Method

1. Control phase. Each of 8 built-in prompts is sent at batch size 1 (the harness
   waits for the server to go idle before each request). Each prompt is generated 3
   times; the control is accepted only if all 3 outputs are identical character for
   character. A run stops if a control does not reproduce, unless
   `--allow-nondeterministic-control` is passed.
2. Measured rounds. The same prompts are sent again while the server is held at
   `--concurrency` in-flight requests, using filler requests to keep the batch full.
   Prompt order is shuffled per round from `--seed`, so batch composition varies
   but repeats exactly across runs with the same seed.
3. Comparison. Each streamed response is compared to its control character by
   character as it arrives. The score is the index of the first differing
   character; an identical response scores the full control length. A diverged
   request is cancelled by request id (SGLang `/abort_request`) unless
   `--no-early-cancel` is passed.
4. Re-check. After the rounds, each control is generated again at batch size 1 to
   confirm it still reproduces.

The report gives the share of responses identical to the control, the median over
prompts of the per-prompt median first-difference index with a bootstrap 95%
interval, and whether each divergence fell in reasoning text or in the answer.

The harness also checks its own preconditions: it reads the server arguments from
SGLang's `/get_server_info` and refuses to run with the radix (prefix) cache on
unless `--allow-radix-cache` is passed, captures the request body actually sent to
confirm that `seed` reached the server, checks that streamed and non-streamed output
match, checks that abort by request id stops a request, and samples `/metrics` to
record batch occupancy when the server exposes it.

## Requirements

Python 3.11 or later. Dependencies (`litellm`, `httpx`) are declared as PEP 723
inline metadata at the top of `benchmark.py`, so `uv run` installs them:

```sh
uv run benchmark.py --help
```

Tested with Python 3.13.12, litellm 1.105.1 and httpx 0.28.1.

`tools/fake_server.py` uses only the standard library.

## Running against SGLang

Launch the server with `--enable-deterministic-inference`, `--disable-radix-cache`
and `--enable-metrics`. See `SGLANG.md` for the exact launch commands of the
published runs. Then:

```sh
uv run benchmark.py \
  --name my_run \
  --api-base http://127.0.0.1:30000 \
  --api-key "$SGLANG_API_KEY" \
  --rounds 100 \
  --concurrency 40 \
  --temperature 0 \
  --control-max-tokens 1000 \
  --reasoning-effort medium \
  --abort-all
```

`--api-key` defaults to `$OPENAI_API_KEY`, or `EMPTY` if that is unset.
`--admin-key` (default `$SGLANG_ADMIN_KEY`) is used for admin endpoints if they use
a different key. `--abort-all` clears every request on the server between phases;
use it only on a server dedicated to the benchmark.

Each run writes `<name>_<UTC timestamp>.md` and `.json` to `--out-dir` (default
`reports/`). The JSON holds every measured response. API keys are written as
`<REDACTED>`.

## Running without a GPU

`tools/fake_server.py` serves the endpoints the harness uses and can inject the
failures it is meant to detect:

```sh
python3 tools/fake_server.py --port 30000 &
uv run benchmark.py --name fake_clean --rounds 3 --concurrency 4 --temperature 0

python3 tools/fake_server.py --port 30001 --diverge --diverge-at 120 &
uv run benchmark.py --name fake_diverge --port 30001 --rounds 3 --concurrency 4 --temperature 0
```

The first run reports 24/24 identical. In the second, the fake server alters
character 120 of a response whenever more than one request is in flight; a test
run on 2026-10-14 reported 3/24 identical and a median first-difference index of
120.

## Published reports

`reports/` holds the four runs from 2026-09-02 (UTC) as `.md` and `.json`:

| File | Configuration |
| --- | --- |
| `qwen38_bi_20260902T230833Z` | Qwen3.8-27B, deterministic inference on |
| `qwen36_bi_20260902T214427Z` | Qwen3.6-27B, deterministic inference on, 10 rounds |
| `q38_dflash2_20260902T190359Z` | Qwen3.8-27B, DFlash speculative decoding, deterministic inference off |
| `gpt120b_eagle3_20260902T212217Z` | gpt-oss-120b, EAGLE3 speculative decoding, deterministic inference off |

Results, launch commands, sampling settings and caveats are in `SGLANG.md`.

## References

- SGLang deterministic inference:
  https://docs.sglang.io/advanced_features/deterministic_inference.html
