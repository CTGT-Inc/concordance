# Ingestion sweep: stage hashes and counts

Per-run, per-stage hashes and counts from the September production ingestion
sweep on one test document, before and after a fix that removed per-run values
from the pipeline's prompts. The files hold hashes and counts only.

## Files

| File | Rows |
|---|---|
| `before_fix.csv` | Runs 2-5 before the fix, 7 stages each (28 rows) |
| `after_fix.csv` | Runs 1-5 after the fix, 7 stages each (35 rows) |
| `summary.csv` | One row per run with the headline measures (9 rows) |
| `repeated_requests_by_module.csv` | Model calls and identical repeated requests per pipeline module, per phase |

Run ids are `before_run_N` and `after_run_N`. There is no `before_run_1`: the
first before-fix run started against an empty topic registry, so its prompts
differed from every later run for a reason unrelated to the fix, and it was
excluded from the comparison.

## Stages

Pipeline order.

| Stage | Output | `item_count` counts |
|---|---|---|
| `parse` | Parsed document: nodes and assets | parse nodes |
| `label` | Structural label per node and asset | node labels |
| `interpret` | Role annotation per node (normative, context, ...) | annotations |
| `unitize` | Rule units grouped from normative nodes | rule units |
| `extract` | Obligations extracted per rule unit, plus tags | obligations |
| `identity` | Obligation graph with definitions and relationships | obligations |
| `committed_result` | The committed ingestion result | obligations |

## Columns: `before_fix.csv`, `after_fix.csv`

| Column | Meaning |
|---|---|
| `run_id` | Run identifier |
| `stage` | Stage name, as above |
| `output_hash` | First 8 hex characters of the SHA-256 of the stage output after canonicalisation and with per-upload ids masked. Empty where no masked hash was recorded for that stage (see Hash computation) |
| `output_hash_unmasked` | First 8 hex characters of the SHA-256 of the stage output after canonicalisation, without masking per-upload ids |
| `item_count` | Count of the stage's primary item, named in `item_kind` |
| `item_kind` | What `item_count` counts |
| `model_calls` | Model calls made by the stage in that run. `label` is the sum of label, declaration and reference calls; `extract` is extraction plus tagging calls; `committed_result` is the total for the run |
| `violations` | Validation violations raised by the stage. `committed_result` is the total for the run. `n/a` where the source reports none for the stage |
| `notes` | Secondary counts: call breakdown, role counts, source segments, definitions, graph relationships, runtime |

Per-stage `model_calls` and `violations` sum to the `committed_result` row in
every run.

## Columns: `summary.csv`

| Column | Meaning |
|---|---|
| `run_id`, `phase` | Run identifier and `before_fix` or `after_fix` |
| `included` | Whether the run is in the comparison set |
| `committed_obligations` | Obligations in the committed result |
| `result_hash` | Committed-result hash with per-upload ids masked (8 hex characters) |
| `result_hash_unmasked` | Committed-result hash without masking |
| `validation_violations` | Total validation violations for the run |
| `graph_relationships` | Relationships in the `identity` stage graph |
| `model_calls` | Total model calls for the run |
| `runtime_seconds` | Wall time of the run |
| `identical_repeated_requests_across_runs` | Number of distinct requests (full request body: system prompt, examples, user payload) that occurred in more than one run of the set. A set-level value, repeated on each row of the set |
| `repeated_requests_output_changed` | Of those repeated requests, how many returned a different response in some run. Set-level |
| `notes` | Caveats for the row |

## Hash computation

Each stage output is canonicalised before hashing: run metadata is dropped,
generated graph ids are replaced with references derived from the source
position of the item, and unordered lists (nodes, assets, annotations and
their list fields) are sorted. The canonical JSON is hashed with SHA-256.
Values in this directory are the first 8 hex characters.

The masked hash additionally replaces per-upload ids with placeholders before
hashing. A per-upload id is any identifier derived from the upload UUID, such
as asset ids, which are a SHA-256 over a value that includes the upload's
document version id. These ids change on every upload of the same file, so
two runs that produced the same content still have different unmasked hashes.
After the fix, `label`, `unitize`, `extract`, `identity` and
`committed_result` each have a different unmasked hash per run and one masked
hash across all 5 runs. The unmasked hashes of `parse` and `interpret`
already agree across runs, so no masked hash was recorded for them.

Before the fix, only the unmasked hash was recorded, so `output_hash` and
`result_hash` are empty for before-fix rows. Before-fix stages from `label`
onward also differ in their counts, so masking would not make them agree.

Hashes are published as 8-character prefixes. Full digests were not retained.

## Model and deployment

- Model: Qwen3.8-27B, served by SGLang on a batch-invariant deployment
  (`--enable-deterministic-inference`).
- Temperature 1.0. The client sends no seed; `extra_body` carries only
  thinking off and reasoning effort low.
- Radix cache on. The separate inference benchmark in `inference/` ran with
  the radix cache disabled.

## Entropy sources removed by the fix

1. Asset ids derived from the per-upload UUID (SHA-256 of a value that
   includes the upload's document version id). They appeared in 53 of 459
   captured calls. The fix replaces them in prompts with aliases that do not
   depend on the upload.
2. Order of `semantic_route_selections`, a list of 30 items describing where
   content sits in the document. The items were the same in every run; the
   order was not. The fix sorts the list.
3. Topic registry growth. The topic vocabulary passed to the tagging step grew
   from 21 to 22 keys across runs because each run added to a shared registry.

In both phases, no identical request repeated across runs returned a
different response (0 of 74 before the fix, 0 of 89 after). "Identical" means
the full request body was byte-equal across runs, with no masking of
per-upload ids; `repeated_requests_by_module.csv` gives the per-module
breakdown that sums to each total. Before the fix, 8 further recurring
requests were equal except for values the pipeline generated (asset ids), and
all 8 returned a different answer. The committed result changed before the
fix because the requests themselves differed between runs.

## Not included

- The test document and any of its text.
- Prompts, prompt templates, or rendered requests.
- Captured model calls or responses.
- Engine source code.

The per-stage hashes let a reader check agreement between runs. They do not
allow reconstruction of any of the above.
