# Call capture format

`compare_runs.py` reads one directory per pipeline run. The toy pipeline in
`../toy_pipeline/` writes this format, and any other pipeline can write it
with a small wrapper around its model client (see `README.md`).

## Run directory

```
<run_dir>/
  calls.jsonl          required: one line per model call, in the order the calls were made
  stages/<stage>.json  optional: the artifact a pipeline stage committed, as JSON (any shape)
```

The file name without `.json` is the stage name. Stage artifacts are used to
find the first stage whose output differs between runs, and one of them can be
named as the committed result with `--final`.

## `calls.jsonl` record

```json
{
  "run_id": "run_1",
  "call_id": "run_1-0042",
  "sequence": 42,
  "stage": "interpret",
  "request": {
    "model": "example-model",
    "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "temperature": 1.0,
    "extra": {}
  },
  "response": {
    "content": "...",
    "finish_reason": "stop"
  }
}
```

| key | type | required | meaning |
|---|---|---|---|
| `run_id` | string | yes | Run label. Every line in one `calls.jsonl` must carry the same value, and two run directories passed together must not share one. |
| `call_id` | string | no | Label for the call. Defaults to `<run_id>-<sequence>`. |
| `sequence` | integer | yes | Order of the call within the run. Unique within a run. Stage order is taken from the lowest `sequence` at which each stage appears. |
| `stage` | string | yes | Pipeline stage that made the call. Matches the `stages/<stage>.json` name when the stage writes an artifact. |
| `request` | object | yes | The OpenAI-style chat request body, minus anything the comparison should ignore (API keys, request timestamps, tracing headers). |
| `response` | object | yes | Must contain `content`. |
| `response.content` | string | yes | The full text the model returned. |
| `response.finish_reason` | string | no | Recorded for reference; not compared. |

Extra keys anywhere in a record are allowed and ignored, except inside
`request`, where every key is part of the request identity.

## Identity

- Request identity: SHA-256 of the canonical JSON of `request`:
  `json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`,
  encoded as UTF-8. Key order in the captured file does not matter; any change
  to a value, including whitespace inside a prompt, does.
- Response identity: SHA-256 of `response.content` encoded as UTF-8.
- Stage artifact identity: SHA-256 of the canonical JSON of the parsed
  artifact, using the same `json.dumps` arguments. Formatting and key order in
  the file do not matter.

With `--normalize-regex`, every match of each pattern is replaced with
`<normalized>` in all string values and object keys of the request, in
`response.content`, and in all string values and object keys of stage
artifacts, before the hashes above are computed.

## Rendered request text

For line diffs, a request is rendered as text: top-level keys in sorted order,
each message as a `[messages.<i> role=<role>]` header followed by its
`content` split on newlines, and all other values as pretty-printed JSON. A
change to one line of a long prompt therefore counts as one differing line.
This rendering is used only for diffs and similarity; hashing always uses the
canonical JSON above.
