# Run comparison

Runs: run_1, run_2, run_3. Baseline for pairwise comparisons: run_1.
Normalisation: none. Hashes are over the captured bytes.

## Summary

- Identical requests repeated across runs: 5; with more than one distinct response: 0.
- Near-identical request pairs (same stage, request text differs): 0.
- First stage whose artifact differs: none.
- Final stage `commit`: all runs agree: yes (1 distinct hash).

## 1. Calls per run

| run | calls | parse | interpret | commit |
|---|---|---|---|---|
| run_1 | 5 | 1 | 3 | 1 |
| run_2 | 5 | 1 | 3 | 1 |
| run_3 | 5 | 1 | 3 | 1 |

Stage columns are in pipeline order, taken from the first call sequence of each stage.

## 2. Identical requests repeated across runs

| measure | count |
|---|---|
| distinct request hashes | 5 |
| seen in more than one run | 5 |
| seen in every run | 5 |
| seen in more than one run, more than one distinct response | 0 |
| seen in one run only | 0 |

## 3. Near-identical requests

Each run is compared with run_1. Within a stage, requests with equal hashes are matched first; remaining requests are paired by position within the stage, then by line similarity (minimum ratio 0.5). Differing lines count changed, removed and inserted lines of the rendered request text.

No near-identical pairs.

## 4. Stage artifacts

| stage | run_1 | run_2 | run_3 | all equal |
|---|---|---|---|---|
| parse | b253e7d75717 | b253e7d75717 | b253e7d75717 | yes |
| interpret | c2b862103d84 | c2b862103d84 | c2b862103d84 | yes |
| commit | 6228b15cdca3 | 6228b15cdca3 | 6228b15cdca3 | yes |

All stage artifacts are identical across runs.

## 5. Committed result

| run | commit.json |
|---|---|
| run_1 | 6228b15cdca3 |
| run_2 | 6228b15cdca3 |
| run_3 | 6228b15cdca3 |

All 3 runs committed the same `commit` artifact.
