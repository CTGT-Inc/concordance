# compare_calls

`compare_runs.py` compares two or more runs of the same LLM pipeline on the
same input. It reads the model calls each run made and the artifact each stage
committed, and reports:

1. Calls per run and per stage.
2. Requests that were byte-identical across runs, and how many of those got
   more than one distinct response.
3. Requests that were almost identical: same stage, same position (or best
   line-similarity match), different text. For each pair, the number of
   differing lines and a unified diff of the first N of them.
4. The hash of each stage artifact per run, the first stage in pipeline order
   whose artifact differs, and a diff of that artifact between the first two
   runs that differ.
5. With `--final STAGE`, the hash of that stage's artifact per run and whether
   all runs agree.

Item 2 separates two causes of run-to-run variation. If identical requests
always get identical responses, the inference side is reproducible for those
calls, and any change in the committed result comes from the requests
themselves differing. Item 3 shows how they differ.

Single file, Python 3.11 or later, standard library only.

## Usage

```
python compare_runs.py RUN_DIR [RUN_DIR ...] [--out report.md] [--json results.json]
                       [--final STAGE] [--normalize-regex PATTERN]...
                       [--max-diff-lines N] [--stage-order STAGE,STAGE,...]
```

| option | meaning |
|---|---|
| `RUN_DIR` | A run directory in the format described in `SCHEMA.md`. At least two. |
| `--out PATH` | Write the markdown report to PATH. Default: stdout. |
| `--json PATH` | Also write all results as JSON (sorted keys, full hashes). |
| `--final STAGE` | Treat `stages/STAGE.json` as the committed result (item 5). |
| `--normalize-regex PATTERN` | Replace matches with `<normalized>` before hashing. Repeatable. |
| `--max-diff-lines N` | Changed lines shown per diff. Default 20. The differing-line count is always the full count. |
| `--stage-order LIST` | Comma-separated stage order, for pipelines where the call sequence does not give it (for example a stage that writes an artifact but makes no model calls). Stages not listed follow in the default order. |

Exit status is 0 on success and 2 on bad input. Input errors name the file,
the line and the key, for example:

```
error: runs/run_2/calls.jsonl:17: missing required key 'stage'
```

Runs are sorted by `run_id` (numeric parts compared as numbers), so the report
does not depend on argument order. The first run after sorting is the
baseline for the pairwise comparisons in item 3 and the artifact diff in
item 4. The same inputs always produce the same report.

## Examples

`examples/` holds two hand-written sets of three runs each over a toy office
policy document (working hours, meeting rooms, kitchen). There are no real
prompts or model outputs in them. Each run has a `parse` call, three
`interpret` calls (one per section) and a `commit` call.

- `examples/random_ids/`: each run prints a different random 16-hex-digit
  section id into the `interpret` prompts. Those requests differ in exactly one
  line. In `run_2` the meeting-rooms call returns two rules instead of three,
  so `interpret.json` and `commit.json` differ from `run_1` and `run_3`.
  Report: `examples/random_ids/REPORT.md`.
- `examples/aliased_ids/`: the prompts use `section_1`, `section_2`,
  `section_3`. All requests, responses and artifacts are identical across
  runs. Report: `examples/aliased_ids/REPORT.md`.

To regenerate the reports, from `examples/`:

```
python ../compare_runs.py random_ids/run_1 random_ids/run_2 random_ids/run_3 --final commit --out random_ids/REPORT.md
python ../compare_runs.py aliased_ids/run_1 aliased_ids/run_2 aliased_ids/run_3 --final commit --out aliased_ids/REPORT.md
```

The example responses are written by hand. In the random-id set, the dropped
rule in `run_2` stands in for the effect of a changed prompt on a sampled
answer; it is not a measured model output.

## Normalisation

`--normalize-regex` masks per-run values so that requests differing only in
those values count as identical. Matches are replaced with `<normalized>` in
request strings and keys, in response text, and in stage artifacts. When
normalisation is on, item 2 also prints the counts without normalisation.

```
python compare_runs.py examples/random_ids/run_* --final commit \
  --normalize-regex '[0-9a-f]{16}'
```

On the random-id example this moves the six `interpret` pairs out of item 3
and into item 2. One of them, the meeting-rooms call, then shows two distinct
responses for one normalised request. That means the masked text was the only
difference in the requests and the responses still differed: the per-run id
is what changed the output. Normalising also hides the ids in stage artifacts,
so `commit.json` still differs (the missing rule) while the id-only
differences in `interpret.json` no longer count.

Patterns are Python regular expressions applied in the order given. A pattern
that is too broad can merge requests that differ in meaningful ways; check
the "without normalisation" line against the normalised counts. If two object
keys normalise to the same text, the second gets a `#2` suffix so no entry is
lost.

## Producing `calls.jsonl` from your own pipeline

Wrap the model client so every call appends one record, and give each run its
own directory. Example for an OpenAI-compatible client:

```python
import json, os, threading
from openai import OpenAI

class CapturingClient:
    def __init__(self, run_dir, run_id, **kwargs):
        self.client, self.dir, self.run_id, self.seq, self.lock = OpenAI(**kwargs), run_dir, run_id, 0, threading.Lock()
        os.makedirs(os.path.join(run_dir, "stages"), exist_ok=True)

    def chat(self, stage, **request):
        choice = self.client.chat.completions.create(**request).choices[0]
        with self.lock, open(os.path.join(self.dir, "calls.jsonl"), "a", encoding="utf-8") as f:
            self.seq += 1
            f.write(json.dumps({"run_id": self.run_id, "sequence": self.seq, "stage": stage, "request": request,
                "response": {"content": choice.message.content, "finish_reason": choice.finish_reason}}) + "\n")
        return choice.message.content
```

Call `client.chat("interpret", model=..., messages=..., temperature=...)` in
place of `chat.completions.create`, and write each stage's committed output
with `json.dump` to `<run_dir>/stages/<stage>.json`.

Notes on the wrapper:

- `request` should hold every input that reaches the model: model name,
  messages, sampling parameters, and anything passed through `extra_body`.
  Leave out values that do not reach the model, such as API keys and request
  timeouts.
- With concurrent calls, `sequence` records completion order, which can vary
  between runs. Pairing in item 3 falls back to line similarity when positions
  do not match, so this does not produce false differences. Stage order is
  taken from the first sequence of each stage; use `--stage-order` if stages
  overlap.
- The captured files contain your prompts and model outputs. Review them
  before sharing.

## Tests

From the repository root:

```
python -m unittest discover -s compare_calls/tests
```
