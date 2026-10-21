# Run comparison

Runs: run_1, run_2, run_3. Baseline for pairwise comparisons: run_1.
Normalisation: none. Hashes are over the captured bytes.

## Summary

- Identical requests repeated across runs: 2; with more than one distinct response: 0.
- Near-identical request pairs (same stage, request text differs): 7.
- First stage whose artifact differs: interpret.
- Final stage `commit`: all runs agree: no (2 distinct hashes).

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
| distinct request hashes | 12 |
| seen in more than one run | 2 |
| seen in every run | 1 |
| seen in more than one run, more than one distinct response | 0 |
| seen in one run only | 10 |

## 3. Near-identical requests

Each run is compared with run_1. Within a stage, requests with equal hashes are matched first; remaining requests are paired by position within the stage, then by line similarity (minimum ratio 0.5). Differing lines count changed, removed and inserted lines of the rendered request text.

| stage | baseline call | call | paired by | differing lines | request chars | same response |
|---|---|---|---|---|---|---|
| interpret | run_1-0002 | run_2-0002 | position | 1 | 360 / 360 | yes |
| interpret | run_1-0003 | run_2-0003 | position | 1 | 390 / 390 | no |
| interpret | run_1-0004 | run_2-0004 | position | 1 | 356 / 356 | yes |
| commit | run_1-0005 | run_2-0005 | position | 1 | 621 / 569 | no |
| interpret | run_1-0002 | run_3-0002 | position | 1 | 360 / 360 | yes |
| interpret | run_1-0003 | run_3-0003 | position | 1 | 390 / 390 | yes |
| interpret | run_1-0004 | run_3-0004 | position | 1 | 356 / 356 | yes |

### interpret: run_1-0002 vs run_2-0002 (1 differing line)

```diff
--- run_1-0002
+++ run_2-0002
@@ -5 +5 @@
-Section id: 3f9a2c71d0be4e58
+Section id: c04b7e19a8f25d36
```

### interpret: run_1-0003 vs run_2-0003 (1 differing line)

```diff
--- run_1-0003
+++ run_2-0003
@@ -5 +5 @@
-Section id: a61c09f2e7d34b10
+Section id: 7e3d51a0b9c2f846
```

### interpret: run_1-0004 vs run_2-0004 (1 differing line)

```diff
--- run_1-0004
+++ run_2-0004
@@ -5 +5 @@
-Section id: 5d2e8b40c19f7a63
+Section id: e8a1f6c3047d2b95
```

### commit: run_1-0005 vs run_2-0005 (1 differing line)

```diff
--- run_1-0005
+++ run_2-0005
@@ -10 +9,0 @@
-- [Meeting rooms] Rooms seat at most eight people.
```

### interpret: run_1-0002 vs run_3-0002 (1 differing line)

```diff
--- run_1-0002
+++ run_3-0002
@@ -5 +5 @@
-Section id: 3f9a2c71d0be4e58
+Section id: 91d7c2e45fa0b318
```

### interpret: run_1-0003 vs run_3-0003 (1 differing line)

```diff
--- run_1-0003
+++ run_3-0003
@@ -5 +5 @@
-Section id: a61c09f2e7d34b10
+Section id: 0b6f3a8de21c9754
```

### interpret: run_1-0004 vs run_3-0004 (1 differing line)

```diff
--- run_1-0004
+++ run_3-0004
@@ -5 +5 @@
-Section id: 5d2e8b40c19f7a63
+Section id: 4c85e0d7b3a16f29
```

## 4. Stage artifacts

| stage | run_1 | run_2 | run_3 | all equal |
|---|---|---|---|---|
| parse | b253e7d75717 | b253e7d75717 | b253e7d75717 | yes |
| interpret | 470e4234a213 | 48fbf059e931 | 4b49eea102b7 | no |
| commit | 6228b15cdca3 | ffb4517a43ab | 6228b15cdca3 | no |

First divergent stage: `interpret`. Diff of `interpret.json` between run_1 and run_2 (5 differing lines of pretty-printed JSON):

```diff
--- run_1/stages/interpret.json
+++ run_2/stages/interpret.json
@@ -6,5 +6,5 @@
         "Staff record leave in the shared calendar."
       ],
-      "section_id": "3f9a2c71d0be4e58",
+      "section_id": "c04b7e19a8f25d36",
       "title": "Working hours"
     },
@@ -12,8 +12,7 @@
       "rules": [
         "Book rooms in the shared calendar.",
-        "Release a booking if plans change.",
-        "Rooms seat at most eight people."
+        "Release a booking if plans change."
       ],
-      "section_id": "a61c09f2e7d34b10",
+      "section_id": "7e3d51a0b9c2f846",
       "title": "Meeting rooms"
     },
@@ -23,5 +22,5 @@
         "The fridge is cleared every Friday."
       ],
-      "section_id": "5d2e8b40c19f7a63",
+      "section_id": "e8a1f6c3047d2b95",
       "title": "Kitchen"
     }
```

## 5. Committed result

| run | commit.json |
|---|---|
| run_1 | 6228b15cdca3 |
| run_2 | ffb4517a43ab |
| run_3 | 6228b15cdca3 |

Runs disagree: 2 distinct `commit` hashes across 3 runs.
