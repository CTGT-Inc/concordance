#!/usr/bin/env python3
"""Compare captured model calls and stage artifacts across repeated pipeline runs.

Usage:
    python compare_runs.py RUN_DIR [RUN_DIR ...] [--out report.md] [--json results.json]
                           [--final STAGE] [--normalize-regex PATTERN]...
                           [--max-diff-lines N] [--stage-order STAGE,STAGE,...]

Each RUN_DIR holds a calls.jsonl file and, optionally, a stages/ directory of
JSON artifacts. See SCHEMA.md for the record format.

Python 3.11+, standard library only.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PLACEHOLDER = "<normalized>"
DEFAULT_MAX_DIFF_LINES = 20
# Pairs of differing requests below this line-similarity ratio are reported as
# unpaired rather than as near-identical.
MIN_PAIR_SIMILARITY = 0.5
SHORT_HASH = 12
REQUIRED_KEYS = ("run_id", "sequence", "stage", "request", "response")


class InputError(Exception):
    """Raised for malformed or missing input. The message is shown to the user."""


# --------------------------------------------------------------------------
# Hashing and normalisation
# --------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_hash(request: Any) -> str:
    return sha256_text(canonical_json(request))


class Normalizer:
    """Replaces every match of each pattern with PLACEHOLDER in all string values and keys."""

    def __init__(self, patterns: list[str]) -> None:
        self.patterns = list(patterns)
        self._compiled: list[re.Pattern[str]] = []
        for pattern in self.patterns:
            try:
                self._compiled.append(re.compile(pattern))
            except re.error as exc:
                raise InputError(f"--normalize-regex {pattern!r} is not a valid regex: {exc}") from exc

    @property
    def active(self) -> bool:
        return bool(self._compiled)

    def text(self, value: str) -> str:
        for regex in self._compiled:
            value = regex.sub(PLACEHOLDER, value)
        return value

    def value(self, value: Any) -> Any:
        if not self._compiled:
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key in sorted(value):
                new_key = self.text(key)
                # Two keys that normalise to the same text must both survive.
                candidate, suffix = new_key, 2
                while candidate in out:
                    candidate = f"{new_key}#{suffix}"
                    suffix += 1
                out[candidate] = self.value(value[key])
            return out
        return value


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Call:
    run_id: str
    call_id: str
    sequence: int
    stage: str
    request: Any
    response_content: str
    request_hash: str = ""
    response_hash: str = ""
    raw_request_hash: str = ""
    raw_response_hash: str = ""
    _text: str | None = field(default=None, repr=False)

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = request_text(self.request)
        return self._text


@dataclass
class Run:
    run_id: str
    path: str
    calls: list[Call]
    stages: dict[str, Any]  # normalised artifacts
    stage_hashes: dict[str, str]

    def calls_per_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for call in self.calls:
            counts[call.stage] = counts.get(call.stage, 0) + 1
        return counts


def _type_name(value: Any) -> str:
    return type(value).__name__ if value is not None else "null"


def parse_record(record: Any, where: str) -> dict[str, Any]:
    """Validate one decoded calls.jsonl line. `where` is 'path:line' for messages."""
    if not isinstance(record, dict):
        raise InputError(f"{where}: record must be a JSON object, got {_type_name(record)}")
    for key in REQUIRED_KEYS:
        if key not in record:
            raise InputError(f"{where}: missing required key '{key}'")
    run_id = record["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise InputError(f"{where}: key 'run_id' must be a non-empty string, got {_type_name(run_id)}")
    sequence = record["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise InputError(f"{where}: key 'sequence' must be an integer, got {_type_name(sequence)}")
    stage = record["stage"]
    if not isinstance(stage, str) or not stage:
        raise InputError(f"{where}: key 'stage' must be a non-empty string, got {_type_name(stage)}")
    request = record["request"]
    if not isinstance(request, dict):
        raise InputError(f"{where}: key 'request' must be a JSON object, got {_type_name(request)}")
    response = record["response"]
    if not isinstance(response, dict):
        raise InputError(f"{where}: key 'response' must be a JSON object, got {_type_name(response)}")
    if "content" not in response:
        raise InputError(f"{where}: missing required key 'response.content'")
    content = response["content"]
    if not isinstance(content, str):
        raise InputError(f"{where}: key 'response.content' must be a string, got {_type_name(content)}")
    call_id = record.get("call_id")
    if call_id is None:
        call_id = f"{run_id}-{sequence}"
    elif not isinstance(call_id, str) or not call_id:
        raise InputError(f"{where}: key 'call_id' must be a non-empty string, got {_type_name(call_id)}")
    return {
        "run_id": run_id,
        "call_id": call_id,
        "sequence": sequence,
        "stage": stage,
        "request": request,
        "content": content,
    }


def load_run(path: Path, normalizer: Normalizer) -> Run:
    if not path.is_dir():
        raise InputError(f"{path}: not a directory")
    calls_path = path / "calls.jsonl"
    if not calls_path.is_file():
        raise InputError(f"{calls_path}: file not found (each run directory needs a calls.jsonl)")

    calls: list[Call] = []
    run_id: str | None = None
    seen_sequences: dict[int, int] = {}
    with calls_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{calls_path}:{line_no}"
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{where}: invalid JSON: {exc.msg} at column {exc.colno}") from exc
            rec = parse_record(decoded, where)
            if run_id is None:
                run_id = rec["run_id"]
            elif rec["run_id"] != run_id:
                raise InputError(
                    f"{where}: key 'run_id' is {rec['run_id']!r} but earlier lines in this file use "
                    f"{run_id!r}; one calls.jsonl must hold one run"
                )
            if rec["sequence"] in seen_sequences:
                raise InputError(
                    f"{where}: key 'sequence' value {rec['sequence']} already used on line "
                    f"{seen_sequences[rec['sequence']]}"
                )
            seen_sequences[rec["sequence"]] = line_no
            request = normalizer.value(rec["request"])
            content = normalizer.text(rec["content"])
            calls.append(
                Call(
                    run_id=rec["run_id"],
                    call_id=rec["call_id"],
                    sequence=rec["sequence"],
                    stage=rec["stage"],
                    request=request,
                    response_content=content,
                    request_hash=request_hash(request),
                    response_hash=sha256_text(content),
                    raw_request_hash=request_hash(rec["request"]),
                    raw_response_hash=sha256_text(rec["content"]),
                )
            )
    if run_id is None:
        raise InputError(f"{calls_path}: no records")
    calls.sort(key=lambda c: c.sequence)

    stages: dict[str, Any] = {}
    stage_hashes: dict[str, str] = {}
    stages_dir = path / "stages"
    if stages_dir.is_dir():
        for stage_file in sorted(stages_dir.glob("*.json")):
            try:
                artifact = json.loads(stage_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise InputError(
                    f"{stage_file}:{exc.lineno}: invalid JSON: {exc.msg} at column {exc.colno}"
                ) from exc
            artifact = normalizer.value(artifact)
            stages[stage_file.stem] = artifact
            stage_hashes[stage_file.stem] = sha256_text(canonical_json(artifact))
    return Run(run_id=run_id, path=str(path), calls=calls, stages=stages, stage_hashes=stage_hashes)


def natural_key(text: str) -> list[Any]:
    return [(0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", text) if part]


def load_runs(paths: list[str], normalizer: Normalizer) -> list[Run]:
    runs = [load_run(Path(p), normalizer) for p in paths]
    by_id: dict[str, str] = {}
    for run in runs:
        if run.run_id in by_id:
            raise InputError(
                f"run_id {run.run_id!r} appears in both {by_id[run.run_id]} and {run.path}; "
                "each run directory needs a distinct run_id"
            )
        by_id[run.run_id] = run.path
    runs.sort(key=lambda r: natural_key(r.run_id))
    return runs


# --------------------------------------------------------------------------
# Request text and line diffs
# --------------------------------------------------------------------------


def _pretty(value: Any) -> list[str]:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).splitlines()


def request_text(request: dict[str, Any]) -> str:
    """Render a request as lines so that a change inside a long prompt is one line, not one blob.

    Message content strings are split on newlines; every other value is pretty-printed JSON.
    """
    lines: list[str] = []
    for key in sorted(request):
        value = request[key]
        if key == "messages" and isinstance(value, list):
            for index, message in enumerate(value):
                if not isinstance(message, dict):
                    lines.append(f"[messages.{index}]")
                    lines.extend(_pretty(message))
                    continue
                lines.append(f"[messages.{index} role={message.get('role')}]")
                for mkey in sorted(message):
                    if mkey == "role":
                        continue
                    mvalue = message[mkey]
                    if mkey == "content" and isinstance(mvalue, str):
                        lines.extend(mvalue.split("\n"))
                    elif mkey == "content" and isinstance(mvalue, list):
                        for pindex, part in enumerate(mvalue):
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                lines.append(f"[messages.{index}.content.{pindex} type={part.get('type')}]")
                                lines.extend(part["text"].split("\n"))
                            else:
                                lines.append(f"[messages.{index}.content.{pindex}]")
                                lines.extend(_pretty(part))
                    else:
                        lines.append(f"[messages.{index}.{mkey}]")
                        lines.extend(_pretty(mvalue))
        else:
            pretty = _pretty(value)
            lines.append(f"{key}: {pretty[0]}")
            lines.extend(pretty[1:])
    return "\n".join(lines)


def count_differing_lines(a_lines: list[str], b_lines: list[str]) -> int:
    """Lines changed, removed or inserted. A one-line substitution counts as 1."""
    total = 0
    matcher = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        total += max(i2 - i1, j2 - j1)
    return total


def limited_diff(
    a_lines: list[str], b_lines: list[str], a_name: str, b_name: str, max_lines: int, context: int
) -> tuple[list[str], bool]:
    """Unified diff truncated after `max_lines` added/removed lines. Returns (lines, truncated)."""
    out: list[str] = []
    changed = 0
    truncated = False
    for line in difflib.unified_diff(a_lines, b_lines, a_name, b_name, n=context, lineterm=""):
        is_change = line[:1] in "+-" and not line.startswith(("+++", "---"))
        if is_change:
            if changed >= max_lines:
                truncated = True
                break
            changed += 1
        out.append(line)
    # Drop a trailing hunk header or context lines left after truncation.
    while truncated and out and not (out[-1][:1] in "+-" and not out[-1].startswith(("+++", "---"))):
        out.pop()
    return out, truncated


def similarity(a: Call, b: Call) -> float:
    a_lines, b_lines = a.text.split("\n"), b.text.split("\n")
    matcher = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    if matcher.real_quick_ratio() < MIN_PAIR_SIMILARITY or matcher.quick_ratio() < MIN_PAIR_SIMILARITY:
        return matcher.quick_ratio()
    return matcher.ratio()


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def stage_order(runs: list[Run], explicit: list[str] | None) -> list[str]:
    """Stages ordered by the first call sequence at which they appear in any run.

    Stages that have artifacts but no calls follow, sorted by name. An explicit
    --stage-order list takes precedence; stages it omits keep the default order after it.
    """
    first_seen: dict[str, int] = {}
    for run in runs:
        for call in run.calls:
            if call.stage not in first_seen or call.sequence < first_seen[call.stage]:
                first_seen[call.stage] = call.sequence
    with_calls = sorted(first_seen, key=lambda s: (first_seen[s], s))
    artifact_only = sorted({s for run in runs for s in run.stages} - set(first_seen))
    default = with_calls + artifact_only
    if not explicit:
        return default
    head = [s for s in explicit if s]
    return head + [s for s in default if s not in head]


def analyze_per_run(runs: list[Run], order: list[str]) -> list[dict[str, Any]]:
    out = []
    for run in runs:
        counts = run.calls_per_stage()
        out.append(
            {
                "run_id": run.run_id,
                "path": run.path,
                "calls": len(run.calls),
                "calls_per_stage": {s: counts[s] for s in order if s in counts},
                "stage_artifacts": sorted(run.stages),
            }
        )
    return out


def _repeat_stats(runs: list[Run], raw: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    occurrences: dict[str, list[Call]] = {}
    for run in runs:
        for call in run.calls:
            key = call.raw_request_hash if raw else call.request_hash
            occurrences.setdefault(key, []).append(call)
    repeated = 0
    changed: list[dict[str, Any]] = []
    in_all = 0
    single = 0
    for key in sorted(occurrences):
        calls = occurrences[key]
        run_ids = {c.run_id for c in calls}
        if len(run_ids) < 2:
            single += 1
            continue
        repeated += 1
        if len(run_ids) == len(runs):
            in_all += 1
        responses = {(c.raw_response_hash if raw else c.response_hash) for c in calls}
        if len(responses) > 1:
            changed.append(
                {
                    "request_hash": key,
                    "stage": calls[0].stage,
                    "distinct_responses": len(responses),
                    "occurrences": [
                        {
                            "run_id": c.run_id,
                            "call_id": c.call_id,
                            "response_hash": c.raw_response_hash if raw else c.response_hash,
                        }
                        for c in sorted(calls, key=lambda c: (natural_key(c.run_id), c.sequence))
                    ],
                }
            )
    stats = {
        "distinct_requests": len(occurrences),
        "repeated_across_runs": repeated,
        "repeated_in_all_runs": in_all,
        "repeated_with_changed_output": len(changed),
        "seen_in_one_run_only": single,
    }
    return stats, changed


def analyze_repeats(runs: list[Run], normalizer: Normalizer) -> dict[str, Any]:
    stats, changed = _repeat_stats(runs, raw=False)
    result: dict[str, Any] = dict(stats)
    result["changed"] = changed
    if normalizer.active:
        raw_stats, _ = _repeat_stats(runs, raw=True)
        result["without_normalization"] = raw_stats
    else:
        result["without_normalization"] = None
    return result


def _pair_stage(base: list[Call], other: list[Call]) -> tuple[list[tuple[Call, Call, str, float]], list[Call], list[Call]]:
    """Pair calls of one stage between the baseline run and another run.

    1. Calls with equal request hashes are matched and dropped (not near-identical).
    2. Remaining calls at the same position within the stage are paired when the pair is at
       least MIN_PAIR_SIMILARITY and neither call has a closer match elsewhere.
    3. Anything left is paired greedily by highest similarity.
    Returns (pairs, baseline_unpaired, other_unpaired). Each pair is (base, other, method, ratio).
    """
    base_pos = {id(c): i for i, c in enumerate(base)}
    other_pos = {id(c): i for i, c in enumerate(other)}

    remaining_other: dict[str, list[Call]] = {}
    for call in other:
        remaining_other.setdefault(call.request_hash, []).append(call)
    base_left: list[Call] = []
    for call in base:
        bucket = remaining_other.get(call.request_hash)
        if bucket:
            bucket.pop(0)
        else:
            base_left.append(call)
    other_left = [c for bucket in remaining_other.values() for c in bucket]
    other_left.sort(key=lambda c: other_pos[id(c)])

    ratios = {(id(b), id(o)): similarity(b, o) for b in base_left for o in other_left}
    best_for_base = {id(b): max((ratios[(id(b), id(o))] for o in other_left), default=0.0) for b in base_left}
    best_for_other = {id(o): max((ratios[(id(b), id(o))] for b in base_left), default=0.0) for o in other_left}

    # Same position within the stage, accepted only when no other candidate is a closer match
    # for either call, so that reordered concurrent calls are not mispaired.
    pairs: list[tuple[Call, Call, str, float]] = []
    other_by_pos = {other_pos[id(c)]: c for c in other_left}
    still_base: list[Call] = []
    used: set[int] = set()
    for call in base_left:
        candidate = other_by_pos.get(base_pos[id(call)])
        if candidate is not None:
            ratio = ratios[(id(call), id(candidate))]
            if (
                ratio >= MIN_PAIR_SIMILARITY
                and ratio >= best_for_base[id(call)]
                and ratio >= best_for_other[id(candidate)]
            ):
                pairs.append((call, candidate, "position", ratio))
                used.add(id(candidate))
                continue
        still_base.append(call)
    still_other = [c for c in other_left if id(c) not in used]

    scored: list[tuple[float, int, int, int, Call, Call]] = []
    for b in still_base:
        for o in still_other:
            ratio = ratios[(id(b), id(o))]
            if ratio >= MIN_PAIR_SIMILARITY:
                bp, op = base_pos[id(b)], other_pos[id(o)]
                scored.append((-ratio, abs(bp - op), bp, op, b, o))
    scored.sort(key=lambda t: t[:4])
    taken_b: set[int] = set()
    taken_o: set[int] = set()
    for neg_ratio, _, _, _, b, o in scored:
        if id(b) in taken_b or id(o) in taken_o:
            continue
        taken_b.add(id(b))
        taken_o.add(id(o))
        pairs.append((b, o, "similarity", -neg_ratio))
    unpaired_base = [c for c in still_base if id(c) not in taken_b]
    unpaired_other = [c for c in still_other if id(c) not in taken_o]
    pairs.sort(key=lambda p: (p[0].sequence, p[1].sequence))
    return pairs, unpaired_base, unpaired_other


def analyze_near_identical(
    runs: list[Run], order: list[str], max_diff_lines: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = runs[0]
    near: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    for run in runs[1:]:
        for stage in order:
            base_calls = [c for c in baseline.calls if c.stage == stage]
            run_calls = [c for c in run.calls if c.stage == stage]
            if not base_calls and not run_calls:
                continue
            pairs, left_base, left_run = _pair_stage(base_calls, run_calls)
            for b, o, method, ratio in pairs:
                a_lines, b_lines = b.text.split("\n"), o.text.split("\n")
                diff, truncated = limited_diff(
                    a_lines, b_lines, b.call_id, o.call_id, max_diff_lines, context=0
                )
                near.append(
                    {
                        "stage": stage,
                        "baseline_run": baseline.run_id,
                        "baseline_call_id": b.call_id,
                        "baseline_sequence": b.sequence,
                        "run_id": run.run_id,
                        "call_id": o.call_id,
                        "sequence": o.sequence,
                        "matched_by": method,
                        "similarity": round(ratio, 4),
                        "differing_lines": count_differing_lines(a_lines, b_lines),
                        "baseline_request_chars": len(canonical_json(b.request)),
                        "request_chars": len(canonical_json(o.request)),
                        "response_identical": b.response_hash == o.response_hash,
                        "diff": diff,
                        "diff_truncated": truncated,
                    }
                )
            if left_base or left_run:
                unpaired.append(
                    {
                        "stage": stage,
                        "baseline_run": baseline.run_id,
                        "run_id": run.run_id,
                        "baseline_only": [c.call_id for c in left_base],
                        "run_only": [c.call_id for c in left_run],
                    }
                )
    return near, unpaired


def analyze_stages(runs: list[Run], order: list[str], max_diff_lines: int) -> dict[str, Any]:
    artifact_stages = [s for s in order if any(s in run.stages for run in runs)]
    hashes = {s: {run.run_id: run.stage_hashes.get(s) for run in runs} for s in artifact_stages}
    first: str | None = None
    diff: dict[str, Any] | None = None
    for stage in artifact_stages:
        per_run = [hashes[stage][run.run_id] for run in runs]
        if len(set(per_run)) > 1:
            first = stage
            run_a = runs[0]
            run_b = next(r for r in runs[1:] if r.stage_hashes.get(stage) != run_a.stage_hashes.get(stage))
            a_lines = _pretty(run_a.stages[stage]) if stage in run_a.stages else []
            b_lines = _pretty(run_b.stages[stage]) if stage in run_b.stages else []
            lines, truncated = limited_diff(
                a_lines,
                b_lines,
                f"{run_a.run_id}/stages/{stage}.json",
                f"{run_b.run_id}/stages/{stage}.json",
                max_diff_lines,
                context=2,
            )
            diff = {
                "stage": stage,
                "run_a": run_a.run_id,
                "run_b": run_b.run_id,
                "differing_lines": count_differing_lines(a_lines, b_lines),
                "lines": lines,
                "truncated": truncated,
            }
            break
    return {"order": artifact_stages, "hashes": hashes, "first_divergent": first, "diff": diff}


def analyze_final(runs: list[Run], final: str | None) -> dict[str, Any] | None:
    if final is None:
        return None
    if not any(final in run.stages for run in runs):
        available = sorted({s for run in runs for s in run.stages})
        raise InputError(
            f"--final {final!r}: no run has stages/{final}.json "
            f"(stages found: {', '.join(available) if available else 'none'})"
        )
    hashes = {run.run_id: run.stage_hashes.get(final) for run in runs}
    values = list(hashes.values())
    return {
        "stage": final,
        "hashes": hashes,
        "distinct": len(set(values)),
        "all_agree": None not in values and len(set(values)) == 1,
    }


def analyze(
    run_dirs: list[str],
    *,
    final: str | None = None,
    normalize: list[str] | None = None,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    stage_order_override: list[str] | None = None,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise InputError("need at least two run directories")
    if max_diff_lines < 0:
        raise InputError("--max-diff-lines must be zero or positive")
    normalizer = Normalizer(normalize or [])
    runs = load_runs(run_dirs, normalizer)
    order = stage_order(runs, stage_order_override)
    near, unpaired = analyze_near_identical(runs, order, max_diff_lines)
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_run": runs[0].run_id,
        "normalize_patterns": normalizer.patterns,
        "placeholder": PLACEHOLDER,
        "stage_order": order,
        "runs": analyze_per_run(runs, order),
        "repeated_requests": analyze_repeats(runs, normalizer),
        "near_identical": near,
        "unpaired": unpaired,
        "stages": analyze_stages(runs, order, max_diff_lines),
        "final": analyze_final(runs, final),
        "max_diff_lines": max_diff_lines,
    }


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------


def short(value: str | None) -> str:
    return "missing" if value is None else value[:SHORT_HASH]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _fence(lines: list[str], lang: str = "diff") -> list[str]:
    fence = "```"
    while any(fence in line for line in lines):
        fence += "`"
    return [fence + lang, *lines, fence]


def render_markdown(result: dict[str, Any]) -> str:
    runs = result["runs"]
    run_ids = [r["run_id"] for r in runs]
    order = result["stage_order"]
    rep = result["repeated_requests"]
    near = result["near_identical"]
    stages = result["stages"]
    final = result["final"]
    out: list[str] = ["# Run comparison", ""]

    out.append(f"Runs: {', '.join(run_ids)}. Baseline for pairwise comparisons: {result['baseline_run']}.")
    if result["normalize_patterns"]:
        pats = ", ".join(f"`{p}`" for p in result["normalize_patterns"])
        out.append(
            f"Normalisation: matches of {pats} are replaced with `{result['placeholder']}` in requests, "
            "responses and stage artifacts before hashing."
        )
    else:
        out.append("Normalisation: none. Hashes are over the captured bytes.")
    out.append("")

    out += ["## Summary", ""]
    out.append(
        f"- Identical requests repeated across runs: {rep['repeated_across_runs']}; "
        f"with more than one distinct response: {rep['repeated_with_changed_output']}."
    )
    out.append(f"- Near-identical request pairs (same stage, request text differs): {len(near)}.")
    if stages["order"]:
        first = stages["first_divergent"]
        out.append(f"- First stage whose artifact differs: {first if first else 'none'}.")
    else:
        out.append("- Stage artifacts: none found.")
    if final is not None:
        agree = "yes" if final["all_agree"] else "no"
        out.append(
            f"- Final stage `{final['stage']}`: all runs agree: {agree} "
            f"({final['distinct']} distinct hash{'es' if final['distinct'] != 1 else ''})."
        )
    out.append("")

    # 1. Per run
    out += ["## 1. Calls per run", ""]
    header = ["run", "calls", *order]
    rows = [
        [r["run_id"], str(r["calls"]), *(str(r["calls_per_stage"].get(s, 0)) for s in order)] for r in runs
    ]
    out += _table(header, rows)
    out.append("")
    out.append("Stage columns are in pipeline order, taken from the first call sequence of each stage.")
    out.append("")

    # 2. Repeated requests
    out += ["## 2. Identical requests repeated across runs", ""]
    out += _table(
        ["measure", "count"],
        [
            ["distinct request hashes", str(rep["distinct_requests"])],
            ["seen in more than one run", str(rep["repeated_across_runs"])],
            ["seen in every run", str(rep["repeated_in_all_runs"])],
            ["seen in more than one run, more than one distinct response", str(rep["repeated_with_changed_output"])],
            ["seen in one run only", str(rep["seen_in_one_run_only"])],
        ],
    )
    out.append("")
    raw = rep["without_normalization"]
    if raw is not None:
        out.append(
            f"Without normalisation: {raw['repeated_across_runs']} repeated across runs, "
            f"{raw['repeated_with_changed_output']} with more than one distinct response, "
            f"{raw['seen_in_one_run_only']} seen in one run only."
        )
        out.append("")
    if rep["changed"]:
        out.append("Repeated requests whose response changed:")
        out.append("")
        rows = []
        for item in rep["changed"]:
            occ = ", ".join(f"{o['call_id']}={short(o['response_hash'])}" for o in item["occurrences"])
            rows.append([short(item["request_hash"]), item["stage"], str(item["distinct_responses"]), occ])
        out += _table(["request", "stage", "distinct responses", "call=response"], rows)
        out.append("")

    # 3. Near-identical
    out += ["## 3. Near-identical requests", ""]
    out.append(
        f"Each run is compared with {result['baseline_run']}. Within a stage, requests with equal hashes are "
        "matched first; remaining requests are paired by position within the stage, then by line similarity "
        f"(minimum ratio {MIN_PAIR_SIMILARITY}). Differing lines count changed, removed and inserted lines of "
        "the rendered request text."
    )
    out.append("")
    if not near:
        out.append("No near-identical pairs.")
        out.append("")
    else:
        rows = []
        for item in near:
            rows.append(
                [
                    item["stage"],
                    item["baseline_call_id"],
                    item["call_id"],
                    item["matched_by"],
                    str(item["differing_lines"]),
                    f"{item['baseline_request_chars']:,} / {item['request_chars']:,}",
                    "yes" if item["response_identical"] else "no",
                ]
            )
        out += _table(
            ["stage", "baseline call", "call", "paired by", "differing lines", "request chars", "same response"],
            rows,
        )
        out.append("")
        for item in near:
            out.append(
                f"### {item['stage']}: {item['baseline_call_id']} vs {item['call_id']} "
                f"({item['differing_lines']} differing line{'s' if item['differing_lines'] != 1 else ''})"
            )
            out.append("")
            out += _fence(item["diff"])
            if item["diff_truncated"]:
                out.append("")
                out.append(f"Diff truncated after {result['max_diff_lines']} changed lines.")
            out.append("")
    if result["unpaired"]:
        out.append("Requests with no counterpart in the other run (no equal hash, no pair above the similarity minimum):")
        out.append("")
        rows = [
            [
                u["stage"],
                u["run_id"],
                ", ".join(u["baseline_only"]) or "-",
                ", ".join(u["run_only"]) or "-",
            ]
            for u in result["unpaired"]
        ]
        out += _table(["stage", "run", f"only in {result['baseline_run']}", "only in run"], rows)
        out.append("")

    # 4. Stages
    out += ["## 4. Stage artifacts", ""]
    if not stages["order"]:
        out.append("No `stages/*.json` files found.")
        out.append("")
    else:
        rows = []
        for stage in stages["order"]:
            per_run = stages["hashes"][stage]
            distinct = len(set(per_run.values()))
            rows.append([stage, *(short(per_run[r]) for r in run_ids), "yes" if distinct == 1 else "no"])
        out += _table(["stage", *run_ids, "all equal"], rows)
        out.append("")
        if stages["first_divergent"] is None:
            out.append("All stage artifacts are identical across runs.")
            out.append("")
        else:
            d = stages["diff"]
            out.append(
                f"First divergent stage: `{d['stage']}`. Diff of `{d['stage']}.json` between "
                f"{d['run_a']} and {d['run_b']} ({d['differing_lines']} differing lines of pretty-printed JSON):"
            )
            out.append("")
            out += _fence(d["lines"])
            if d["truncated"]:
                out.append("")
                out.append(f"Diff truncated after {result['max_diff_lines']} changed lines.")
            out.append("")

    # 5. Final
    out += ["## 5. Committed result", ""]
    if final is None:
        out.append("Not requested. Pass `--final STAGE` to compare one stage artifact as the committed result.")
        out.append("")
    else:
        rows = [[r, short(final["hashes"][r])] for r in run_ids]
        out += _table(["run", f"{final['stage']}.json"], rows)
        out.append("")
        if final["all_agree"]:
            out.append(f"All {len(run_ids)} runs committed the same `{final['stage']}` artifact.")
        else:
            out.append(
                f"Runs disagree: {final['distinct']} distinct `{final['stage']}` hashes across {len(run_ids)} runs."
            )
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_runs.py",
        description="Compare captured model calls and stage artifacts across repeated pipeline runs.",
    )
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR", help="run directory with calls.jsonl")
    parser.add_argument("--out", metavar="PATH", help="write the markdown report here instead of stdout")
    parser.add_argument("--json", metavar="PATH", dest="json_out", help="also write machine-readable results")
    parser.add_argument("--final", metavar="STAGE", help="stage whose artifact is the committed result")
    parser.add_argument(
        "--normalize-regex",
        metavar="PATTERN",
        action="append",
        default=[],
        help=f"replace matches with {PLACEHOLDER} before hashing (repeatable)",
    )
    parser.add_argument(
        "--max-diff-lines",
        metavar="N",
        type=int,
        default=DEFAULT_MAX_DIFF_LINES,
        help=f"changed lines shown per diff (default {DEFAULT_MAX_DIFF_LINES})",
    )
    parser.add_argument(
        "--stage-order",
        metavar="STAGES",
        help="comma-separated stage order; overrides the order taken from call sequence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    override = [s.strip() for s in args.stage_order.split(",")] if args.stage_order else None
    try:
        result = analyze(
            args.run_dirs,
            final=args.final,
            normalize=args.normalize_regex,
            max_diff_lines=args.max_diff_lines,
            stage_order_override=override,
        )
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = render_markdown(result)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
