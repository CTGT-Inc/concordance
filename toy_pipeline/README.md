# Toy pipeline

A three-stage document pipeline that reproduces the ingestion finding on a
document anyone can read: when per-run ids are printed into prompts, requests
differ between runs and so do the replies, even on an inference server that
returns the same reply to the same request every time. When the ids are
replaced with stable aliases, the requests are identical across runs and so is
every stage's output.

It runs against any OpenAI-compatible `/v1/chat/completions` endpoint and
writes call captures in the format `../compare_calls/compare_runs.py` reads.

## Files

| File | Contents |
|---|---|
| `synthetic_doc.md` | Operations manual for a fictional public library district, about 2,700 words: 14 numbered sections, two tables, and a block of six same-shaped conflict-resolution rules (9.1 to 9.6). Written for this package. |
| `run_pipeline.py` | The pipeline. Python 3.11+, one dependency (`httpx`). Also `--validate` and `--summary`. |
| `demo.sh` | 3 runs with random ids, 3 with aliased ids, three comparison reports, key numbers. |
| `tools/mock_server.py` | Stdlib-only stand-in for the inference server, for testing without a GPU. |

## Pipeline

1. `parse` (local). Splits the document at numbered `##` and `###` headings
   into 20 sections and extracts the two tables. Assigns an id to every
   section and table. Writes `stages/parse.json`.
2. `interpret` (one model call per section, 20 calls). The prompt contains the
   section's id and text, the ids and contents of any tables it references, and
   the ids and headings of up to two sections on either side. The model is
   asked for a JSON object listing each statement with `kind` (`rule` or
   `background`) and `cites` (ids from the prompt). Writes `calls.jsonl` and
   `stages/interpret.json`. Replies that cannot be parsed as JSON are kept with
   `parse_error: true` and the raw text; the run continues.
3. `commit` (local). Keeps items of kind `rule`, translates cited ids back to
   document references (`section 6`, `Table 1`), deduplicates by normalised
   text (lower case, collapsed whitespace, trailing punctuation removed) and
   sorts by section number and text. Writes `stages/commit.json`.

Because `commit` uses document references and not ids, its hash can be
compared across both id modes. Two commit hashes are equal only if the same
rules were extracted from the same sections with the same citations.

Id modes:

- `--ids random`: `sha256(f"{upload_uuid}:{section_key}")[:16]`, with a new
  `upload_uuid` per run. This is how the production pipeline built its ids.
  The ids appear in every prompt.
- `--ids aliased`: `section_1` ... `section_20`, `table_1`, `table_2`. The
  mapping from ids to document keys is written to `run_meta.json`, outside
  `stages/` and outside every prompt.

Output for each run, under `--out`:

```
<run_id>/calls.jsonl          one record per model call, contract schema
<run_id>/stages/parse.json
<run_id>/stages/interpret.json
<run_id>/stages/commit.json
<run_id>/run_meta.json        upload_uuid, id map, model, timing (not compared)
```

## Running against a GPU server

Start SGLang with deterministic inference. Qwen3-8B is used here because it is
small; any instruct model the server can load works.

```bash
# radix cache off (as in the inference benchmark)
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 \
  --enable-deterministic-inference --attention-backend triton \
  --disable-radix-cache

# radix cache on (as in production)
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 \
  --enable-deterministic-inference --attention-backend triton
```

The production ingestion deployment ran with radix cache on; the inference
benchmark in `../inference/` ran with it off. Run the demo once against each
configuration to see whether that difference matters on your hardware and
SGLang version. Some SGLang versions restrict which attention backends support
deterministic mode, or deterministic mode together with radix cache; if the
server refuses a combination, see the SGLang documentation for your version.
The benchmark's exact version and flags are in `../inference/SGLANG.md`.

Then, from this directory:

```bash
pip install httpx          # or have uv installed; demo.sh falls back to `uv run`
./demo.sh
```

Options:

- `MODEL=Qwen/Qwen3-8B ./demo.sh` sets the model name. Without it, the
  pipeline uses the first id from `GET /v1/models`.
- `BASE_URL=http://host:port/v1 ./demo.sh` points at another server.
- `OPENAI_API_KEY` is sent as a bearer token if set.
- Arguments after the script name go to `run_pipeline.py`, for example
  `./demo.sh --concurrency 8` or `./demo.sh --seed 1`.

Request settings: temperature 1.0, `max_tokens` 2048,
`chat_template_kwargs: {"enable_thinking": false}` (turns Qwen3 thinking off;
pass `--chat-template-kwargs ''` for servers that reject the field). No seed is
sent unless `--seed` is given, matching the production client. Whether a seed
changes the server's sampling depends on the server.

Expected runtime: 120 model calls of 1.8 to 3.6 kB of prompt each. With an
8B-class model on one GPU at concurrency 4 we expect 5 to 10 minutes; this
estimate has not been measured on this document. Raise `--concurrency` if the
server has capacity.

Running the pipeline directly:

```bash
python run_pipeline.py --ids random --runs 3 --out out --run-prefix random
python run_pipeline.py --validate out
python run_pipeline.py --summary out/random_1 out/random_2 out/random_3
```

`uv run run_pipeline.py ...` also works; the dependency is declared in the
script's inline metadata.

## Running without a GPU

```bash
./demo.sh --mock
```

This starts `tools/mock_server.py` on port 30001 (`MOCK_PORT` to change) and
runs the same demo against it. The mock's reply is a function of the SHA-256 of
the request body: identical requests always get identical replies, which is
the property a batch-invariant server provides. The hash chooses whether the
reply is wrapped in a code fence and how it is indented, and, for section 9.4
only, whether that section's statements are returned as `rule` or as
`background`. The mock is a test fixture for the pipeline and the comparison
tool. Its extraction is a sentence splitter with a keyword list, and its
numbers say nothing about model behaviour.

One mock run of `./demo.sh --mock`:

```
random_1  ids=random  calls=20  rules_committed=78  commit=0460ffca
random_2  ids=random  calls=20  rules_committed=78  commit=0460ffca
random_3  ids=random  calls=20  rules_committed=76  commit=d5afdd0f
aliased_1  ids=aliased  calls=20  rules_committed=78  commit=0460ffca
aliased_2  ids=aliased  calls=20  rules_committed=78  commit=0460ffca
aliased_3  ids=aliased  calls=20  rules_committed=78  commit=0460ffca
```

Random ids draw a new `upload_uuid` each run, so the random-id hashes and
which run drops section 9.4's two rules change from one invocation to the
next. In about one invocation in four, all three random runs agree.

## Reading the reports

`demo.sh` writes three reports with `compare_runs.py --final commit
--stage-order parse,interpret,commit`, plus a JSON file next to each:

| Report | Runs | Normalisation |
|---|---|---|
| `out/REPORT_random.md` | random_1..3 | none |
| `out/REPORT_random_normalised.md` | random_1..3 | `\b[0-9a-f]{16}\b` replaced before hashing |
| `out/REPORT_aliased.md` | aliased_1..3 | none |

`REPORT_random.md`:

- Identical requests repeated across runs: 0. Every prompt contains the run's
  ids, so no request is byte-identical to a request in another run.
- Near-identical requests: 20 pairs per run compared with the baseline, each
  differing in 3 to 7 lines. Every differing line is an id line: the section
  id, neighbouring section ids, table ids. The section text, headings and
  instructions are identical.
- Stage artifacts: `parse` differs first, because it records the ids.
- Committed result: the commit hashes may or may not agree. See the caveat
  below.

`REPORT_random_normalised.md` masks the 16-hex-digit ids in requests,
responses and stage artifacts. The requests become identical (20 repeated
across runs). The responses can still differ, because the model was given
different tokens: masking happens after the fact, in the comparison, not in
what the model saw. This is the state of the production finding before the
fix: requests identical after normalising per-upload ids, outputs not.

`REPORT_aliased.md`:

- Identical requests repeated across runs: 20, with 0 having more than one
  distinct response. If this second number is not 0, the server returned
  different replies to the same request, and the question becomes one about
  the inference configuration (`../inference/`), not the pipeline.
- Near-identical requests: 0. First divergent stage: none.
- Committed result: one hash across all runs.

After the reports, `demo.sh` prints the output of `run_pipeline.py --summary`
for each set, which recomputes the repeated-request counts and commit hashes
without the compare tool.

## Caveats

- The reproduced effect is byte-level divergence: random ids make the requests
  differ, the replies differ as a result, and the aliased runs do not. A
  semantic flip, where a rule is dropped or reclassified in the committed
  result, may or may not appear on this document with a given model. The
  production case showed one (a block of same-shaped rules classified as
  background in one run); this toy document includes a block of that shape in
  section 9, but whether a particular model flips on it is not something this
  package can promise.
- At temperature 1.0 a different prompt is expected to change the sampled
  tokens. The demo does not show that random ids make answers worse, only that
  they make them differ between runs.
- Aliased runs being identical shows the pipeline is repeatable. It does not
  show the committed rules are correct. A wrong classification in an aliased
  run is repeated in every aliased run.
- The commit stage counts each extracted sentence as a rule. The rule counts
  are a property of this toy pipeline and are not comparable to the
  production obligation counts in `../data/`.
