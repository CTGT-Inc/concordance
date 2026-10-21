# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.25"]
# ///
"""Three-stage toy ingestion pipeline: parse, interpret, commit.

Runs synthetic_doc.md through an OpenAI-compatible /v1/chat/completions
endpoint and writes every model call plus each stage's artifact in the
Concordance call-capture schema, so compare_calls/compare_runs.py can read it.

    python run_pipeline.py --ids random  --runs 3 --out out --run-prefix random
    python run_pipeline.py --ids aliased --runs 3 --out out --run-prefix aliased
    python run_pipeline.py --validate out
    python run_pipeline.py --summary out/random_1 out/random_2 out/random_3

--ids random   section and table ids are sha256(f"{upload_uuid}:{key}")[:16]
               with a fresh upload_uuid per run, printed into the prompts.
--ids aliased  ids are section_1, section_2, ... and table_1, table_2; the
               mapping to document keys stays outside the prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DOC = HERE / "synthetic_doc.md"
DOC_TITLE = "Tallowmere Public Library District, Branch Operations Manual"

SYSTEM_PROMPT = (
    "You extract operational rules from one section of an internal operations "
    "manual. Reply with one JSON object and nothing else."
)

USER_INSTRUCTIONS = """\
Extract every statement in the section text above as an item.

Use kind "rule" for a statement that requires, permits, prohibits or limits an
action by staff or patrons, or that sets a charge, a time limit or an
escalation. Use kind "background" for definitions, statements of purpose,
descriptions and cross-references that do not themselves direct anyone.

For each item, "cites" lists the ids, taken only from the ids printed above
(section ids and table ids), of the sections or tables the item refers to or
depends on. Use an empty list when there are none.

Reply with a single JSON object of this form:
{"section": "<section id>", "rules": [{"text": "<the statement, in one sentence>", "kind": "rule", "cites": ["<id>"]}]}"""


# ---------------------------------------------------------------------------
# canonical JSON and hashing (same canonical form as the contract)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# stage 1: parse (local, no model)

HEADING_RE = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.?\s+(.*\S)\s*$")
TABLE_CAPTION_RE = re.compile(r"^\*\*Table\s+(\d+)\.\s*(.*?)\*\*\s*$")
TABLE_REF_RE = re.compile(r"\bTable\s+(\d+)\b")


def parse_document(text: str) -> tuple[list[dict], list[dict]]:
    """Split into sections (## and ### numbered headings) and tables.

    Returns (sections, tables). Each section: key, heading, level, body lines.
    Each table: key, number, caption, markdown, defined_in (section key).
    """
    sections: list[dict] = []
    tables: list[dict] = []
    current: dict | None = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADING_RE.match(line)
        if m:
            current = {
                "key": m.group(2),
                "heading": f"{m.group(2)} {m.group(3)}",
                "level": len(m.group(1)),
                "body": [],
            }
            sections.append(current)
            i += 1
            continue
        cap = TABLE_CAPTION_RE.match(line)
        if cap and current is not None:
            # caption, optional blank lines, then contiguous | rows
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j].rstrip())
                j += 1
            number = int(cap.group(1))
            tables.append(
                {
                    "key": f"T{number}",
                    "number": number,
                    "caption": cap.group(2).strip(),
                    "markdown": "\n".join(rows),
                    "defined_in": current["key"],
                }
            )
            current["body"].append({"table": f"T{number}"})
            i = j
            continue
        if current is not None:
            current["body"].append(line)
        i += 1
    for s in sections:
        # trim leading and trailing blank lines
        body = s["body"]
        while body and body[0] == "":
            body.pop(0)
        while body and body[-1] == "":
            body.pop()
    return sections, tables


def assign_ids(sections: list[dict], tables: list[dict], mode: str, upload_uuid: str) -> dict[str, str]:
    """Return key -> id for every section key and table key."""
    ids: dict[str, str] = {}
    if mode == "random":
        for s in sections:
            ids[s["key"]] = hashlib.sha256(f"{upload_uuid}:{s['key']}".encode()).hexdigest()[:16]
        for t in tables:
            ids[t["key"]] = hashlib.sha256(f"{upload_uuid}:table:{t['number']}".encode()).hexdigest()[:16]
    elif mode == "aliased":
        for n, s in enumerate(sections, 1):
            ids[s["key"]] = f"section_{n}"
        for n, t in enumerate(tables, 1):
            ids[t["key"]] = f"table_{n}"
    else:
        raise ValueError(mode)
    return ids


def section_text(section: dict, ids: dict[str, str]) -> str:
    out = []
    for item in section["body"]:
        if isinstance(item, dict):
            out.append(f"[table {ids[item['table']]}]")
        else:
            out.append(item)
    return "\n".join(out)


def referenced_tables(section: dict, tables: list[dict]) -> list[str]:
    keys = set()
    for item in section["body"]:
        if isinstance(item, dict):
            keys.add(item["table"])
        else:
            for m in TABLE_REF_RE.finditer(item):
                keys.add(f"T{m.group(1)}")
    known = {t["key"] for t in tables}
    return sorted(k for k in keys if k in known)


def stage_parse(doc_text: str, mode: str, upload_uuid: str, doc_name: str):
    sections, tables = parse_document(doc_text)
    ids = assign_ids(sections, tables, mode, upload_uuid)
    for s in sections:
        s["id"] = ids[s["key"]]
        s["text"] = section_text(s, ids)
        s["tables"] = referenced_tables(s, tables)
    for t in tables:
        t["id"] = ids[t["key"]]
    artifact = {
        "document": doc_name,
        "document_sha256": sha256_text(doc_text),
        "id_mode": mode,
        "sections": [
            {
                "id": s["id"],
                "heading": s["heading"],
                "level": s["level"],
                "tables": [ids[k] for k in s["tables"]],
                # hash of the section's source text, before ids are substituted
                "source_sha256": sha256_text(canonical(s["body"])),
            }
            for s in sections
        ],
        "tables": [
            {"id": t["id"], "caption": t["caption"], "defined_in": ids[t["defined_in"]]}
            for t in tables
        ],
    }
    return sections, tables, ids, artifact


# ---------------------------------------------------------------------------
# stage 2: interpret (one model call per section)


def build_messages(idx: int, sections: list[dict], tables_by_key: dict[str, dict], neighbours: int = 2) -> list[dict]:
    s = sections[idx]
    lo, hi = max(0, idx - neighbours), min(len(sections), idx + neighbours + 1)
    neigh = [f"- {sections[j]['id']}: {sections[j]['heading']}" for j in range(lo, hi) if j != idx]
    parts = [
        f"Document: {DOC_TITLE}",
        f"Section id: {s['id']}",
        f"Section heading: {s['heading']}",
        "",
        "Neighbouring sections:",
        *neigh,
    ]
    if s["tables"]:
        parts += ["", "Referenced tables:"]
        for k in s["tables"]:
            t = tables_by_key[k]
            parts += [f"Table id {t['id']} (Table {t['number']}): {t['caption']}", t["markdown"], ""]
        if parts[-1] == "":
            parts.pop()
    parts += ["", "Section text:", "<<<", s["text"], ">>>", "", USER_INSTRUCTIONS]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(content: str):
    """Return (obj, error). Strips think blocks and code fences, then falls back
    to the outermost {...} span."""
    text = THINK_RE.sub("", content or "").strip()
    candidates = [text]
    candidates += [m.group(1).strip() for m in FENCE_RE.finditer(text)]
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    err = "empty response"
    for c in candidates:
        if not c:
            continue
        try:
            obj = json.loads(c)
        except json.JSONDecodeError as e:
            err = f"JSONDecodeError: {e.msg} at {e.pos}"
            continue
        if isinstance(obj, dict):
            return obj, None
        err = f"top-level JSON is {type(obj).__name__}, expected object"
    return None, err


def normalise_items(obj: dict) -> tuple[list[dict], str | None]:
    items = obj.get("rules")
    if items is None:
        items = obj.get("items")
    if not isinstance(items, list):
        return [], "no 'rules' list in reply"
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = it.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        kind = str(it.get("kind", "")).strip().lower()
        cites = it.get("cites") or []
        if isinstance(cites, str):
            cites = [cites]
        out.append(
            {
                "text": text.strip(),
                "kind": kind if kind in ("rule", "background") else (kind or "unspecified"),
                "cites": [str(c) for c in cites if isinstance(c, (str, int))],
            }
        )
    return out, None


async def chat(client, args, request: dict) -> dict:
    body = {
        "model": request["model"],
        "messages": request["messages"],
        "temperature": request["temperature"],
        **request["extra"],
    }
    last_err = None
    for attempt in range(args.retries + 1):
        try:
            r = await client.post("/chat/completions", json=body)
            if r.status_code >= 500 or r.status_code == 429:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            elif r.status_code >= 400:
                return {"content": "", "finish_reason": "error", "error": f"HTTP {r.status_code}: {r.text[:500]}"}
            else:
                data = r.json()
                choice = data["choices"][0]
                content = (choice.get("message") or {}).get("content") or ""
                resp = {"content": content, "finish_reason": choice.get("finish_reason")}
                if data.get("usage"):
                    resp["usage"] = data["usage"]
                return resp
        except Exception as e:  # network errors, bad JSON from server
            last_err = f"{type(e).__name__}: {e}"
        if attempt < args.retries:
            await asyncio.sleep(1.5 * (attempt + 1))
    return {"content": "", "finish_reason": "error", "error": last_err}


async def stage_interpret(client, args, run_id: str, model: str, sections, tables, seq_start: int):
    tables_by_key = {t["key"]: t for t in tables}
    extra: dict = {"max_tokens": args.max_tokens}
    if args.seed is not None:
        extra["seed"] = args.seed
    if args.chat_template_kwargs:
        extra["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(idx: int):
        seq = seq_start + idx
        request = {
            "model": model,
            "messages": build_messages(idx, sections, tables_by_key),
            "temperature": args.temperature,
            "extra": extra,
        }
        async with sem:
            response = await chat(client, args, request)
        return {
            "run_id": run_id,
            "call_id": f"{run_id}-{seq:04d}",
            "sequence": seq,
            "stage": "interpret",
            "request": request,
            "response": response,
        }

    # tasks are created in section order; sequence is the section order
    records = await asyncio.gather(*(one(i) for i in range(len(sections))))
    results = []
    for s, rec in zip(sections, records):
        entry = {"id": s["id"], "heading": s["heading"], "sequence": rec["sequence"], "parse_error": False}
        if rec["response"].get("finish_reason") == "error":
            entry.update(parse_error=True, error=rec["response"].get("error"), rules=[], raw="")
        else:
            obj, err = extract_json(rec["response"]["content"])
            items, err2 = normalise_items(obj) if obj is not None else ([], None)
            if err or err2:
                entry.update(parse_error=True, error=err or err2, raw=rec["response"]["content"])
            entry["rules"] = items
        results.append(entry)
    return records, {"sections": results}


# ---------------------------------------------------------------------------
# stage 3: commit (local merge)


def norm_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    return t.rstrip(" .;:")


def section_sort_key(key: str):
    return tuple(int(p) for p in key.split("."))


def stage_commit(interpret: dict, sections, tables, ids: dict[str, str]) -> dict:
    id_to_label = {ids[s["key"]]: f"section {s['key']}" for s in sections}
    id_to_label.update({ids[t["key"]]: f"Table {t['number']}" for t in tables})
    key_of = {s["id"]: s["key"] for s in sections}
    merged: dict[str, dict] = {}
    background = 0
    errors = []
    for entry in interpret["sections"]:
        skey = key_of[entry["id"]]
        if entry.get("parse_error"):
            errors.append(f"section {skey}")
        for item in entry["rules"]:
            if item["kind"] != "rule":
                background += 1
                continue
            cites = sorted({id_to_label.get(c, "unresolved") for c in item["cites"]})
            n = norm_text(item["text"])
            if n in merged:
                merged[n]["cites"] = sorted(set(merged[n]["cites"]) | set(cites))
                continue
            merged[n] = {"text": n, "section": skey, "cites": cites}
    rules = sorted(merged.values(), key=lambda r: (section_sort_key(r["section"]), r["text"]))
    return {
        "rule_count": len(rules),
        "background_count": background,
        "sections_with_errors": errors,
        "rules": rules,
    }


# ---------------------------------------------------------------------------
# run driver


async def resolve_model(client, args) -> str:
    if args.model:
        return args.model
    try:
        r = await client.get("/models")
        r.raise_for_status()
        return r.json()["data"][0]["id"]
    except Exception as e:
        sys.exit(f"--model not given and GET {args.base_url}/models failed: {e}")


async def run_once(client, args, run_id: str, model: str, doc_text: str) -> dict:
    run_dir = Path(args.out) / run_id
    (run_dir / "stages").mkdir(parents=True, exist_ok=True)
    upload_uuid = str(uuid.uuid4())
    t0 = time.monotonic()

    sections, tables, ids, parse_artifact = stage_parse(doc_text, args.ids, upload_uuid, Path(args.doc).name)
    write_json(run_dir / "stages" / "parse.json", parse_artifact)

    records, interpret_artifact = await stage_interpret(client, args, run_id, model, sections, tables, seq_start=1)
    with open(run_dir / "calls.jsonl", "w", encoding="utf-8") as f:
        for rec in sorted(records, key=lambda r: r["sequence"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_json(run_dir / "stages" / "interpret.json", interpret_artifact)

    commit_artifact = stage_commit(interpret_artifact, sections, tables, ids)
    write_json(run_dir / "stages" / "commit.json", commit_artifact)

    # kept outside stages/ and outside every prompt
    write_json(
        run_dir / "run_meta.json",
        {
            "run_id": run_id,
            "id_mode": args.ids,
            "upload_uuid": upload_uuid,
            "id_map": {v: k for k, v in ids.items()},
            "model": model,
            "base_url": args.base_url,
            "elapsed_s": round(time.monotonic() - t0, 2),
        },
    )
    commit_hash = sha256_text(canonical(commit_artifact))[:8]
    errors = sum(1 for e in interpret_artifact["sections"] if e.get("parse_error"))
    line = (
        f"{run_id}  ids={args.ids}  calls={len(records)}  rules_committed={commit_artifact['rule_count']}"
        f"  commit={commit_hash}"
    )
    if errors:
        line += f"  parse_errors={errors}"
    print(line, flush=True)
    return {"run_id": run_id, "commit": commit_hash}


async def main_run(args) -> None:
    import httpx  # imported here so --validate and --summary work without it

    doc_text = Path(args.doc).read_text(encoding="utf-8")
    headers = {}
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    limits = httpx.Limits(max_connections=max(args.concurrency, 1) * 2)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), headers=headers, timeout=args.timeout, limits=limits
    ) as client:
        model = await resolve_model(client, args)
        for n in range(1, args.runs + 1):
            await run_once(client, args, f"{args.run_prefix}_{n}", model, doc_text)


# ---------------------------------------------------------------------------
# --validate


def find_run_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "calls.jsonl").is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted(d for d in p.iterdir() if (d / "calls.jsonl").is_file())
    return out


def validate_run(run_dir: Path) -> list[str]:
    errs = []
    seen_seq = set()
    prev = None
    run_ids = set()
    with open(run_dir / "calls.jsonl", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            where = f"{run_dir}/calls.jsonl:{ln}"
            if not line.strip():
                errs.append(f"{where}: blank line")
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"{where}: invalid JSON: {e}")
                continue
            if not isinstance(rec, dict):
                errs.append(f"{where}: record is not an object")
                continue
            for k in ("run_id", "sequence", "stage", "request", "response"):
                if k not in rec:
                    errs.append(f"{where}: missing required key '{k}'")
            if not isinstance(rec.get("run_id"), str):
                errs.append(f"{where}: run_id must be a string")
            run_ids.add(rec.get("run_id"))
            seq = rec.get("sequence")
            if not isinstance(seq, int) or isinstance(seq, bool):
                errs.append(f"{where}: sequence must be an integer")
            else:
                if seq in seen_seq:
                    errs.append(f"{where}: duplicate sequence {seq}")
                if prev is not None and seq < prev:
                    errs.append(f"{where}: sequence {seq} out of order after {prev}")
                seen_seq.add(seq)
                prev = seq
            if "call_id" in rec and not isinstance(rec["call_id"], str):
                errs.append(f"{where}: call_id must be a string")
            if not isinstance(rec.get("stage"), str):
                errs.append(f"{where}: stage must be a string")
            req = rec.get("request")
            if not isinstance(req, dict):
                errs.append(f"{where}: request must be an object")
            else:
                msgs = req.get("messages")
                if not isinstance(req.get("model"), str):
                    errs.append(f"{where}: request.model must be a string")
                if not isinstance(msgs, list) or not all(
                    isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                    for m in msgs
                ):
                    errs.append(f"{where}: request.messages must be a list of {{role, content}} strings")
                if "temperature" in req and not isinstance(req["temperature"], (int, float)):
                    errs.append(f"{where}: request.temperature must be a number")
                if "extra" in req and not isinstance(req["extra"], dict):
                    errs.append(f"{where}: request.extra must be an object")
                try:
                    canonical(req)
                except (TypeError, ValueError) as e:
                    errs.append(f"{where}: request is not canonical-JSON serialisable: {e}")
            resp = rec.get("response")
            if not isinstance(resp, dict) or not isinstance(resp.get("content"), str):
                errs.append(f"{where}: response.content must be a string")
            elif resp.get("finish_reason") == "error":
                errs.append(f"{where}: call failed: {resp.get('error')}")
    if len(run_ids) > 1:
        errs.append(f"{run_dir}/calls.jsonl: more than one run_id: {sorted(map(str, run_ids))}")
    stages = run_dir / "stages"
    for name in ("parse", "interpret", "commit"):
        p = stages / f"{name}.json"
        if not p.is_file():
            errs.append(f"{p}: missing")
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errs.append(f"{p}: invalid JSON: {e}")
    return errs


def main_validate(paths: list[str]) -> int:
    runs = find_run_dirs(paths)
    if not runs:
        print(f"no run directories (containing calls.jsonl) under {paths}", file=sys.stderr)
        return 2
    bad = 0
    for r in runs:
        errs = validate_run(r)
        n = sum(1 for _ in open(r / "calls.jsonl", encoding="utf-8"))
        if errs:
            bad += 1
            print(f"FAIL {r}  ({len(errs)} problems)")
            for e in errs[:20]:
                print(f"  {e}")
        else:
            print(f"ok   {r}  calls={n}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# --summary


def main_summary(paths: list[str], patterns: list[str]) -> int:
    runs = find_run_dirs(paths)
    regs = [re.compile(p) for p in patterns]

    def norm(s: str) -> str:
        for r in regs:
            s = r.sub("<ID>", s)
        return s

    req_runs: dict[str, set] = {}
    req_resps: dict[str, set] = {}
    commits = {}
    for r in runs:
        for line in open(r / "calls.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            rh = sha256_text(norm(canonical(rec["request"])))
            req_runs.setdefault(rh, set()).add(r.name)
            req_resps.setdefault(rh, set()).add(sha256_text(norm(rec["response"]["content"])))
        c = json.loads((r / "stages" / "commit.json").read_text(encoding="utf-8"))
        commits[r.name] = (sha256_text(canonical(c))[:8], c.get("rule_count"))
    repeated = [h for h, s in req_runs.items() if len(s) > 1]
    changed = [h for h in repeated if len(req_resps[h]) > 1]
    label = " (normalised)" if regs else ""
    print(f"runs: {len(runs)}{label}")
    for name, (h, n) in commits.items():
        print(f"  {name}: commit {h}, {n} rules")
    print(f"distinct commit hashes: {len({h for h, _ in commits.values()})}")
    print(f"requests repeated across runs: {len(repeated)}; of those with more than one response: {len(changed)}")
    return 0


# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1", help="OpenAI-compatible base URL (default %(default)s)")
    p.add_argument("--model", default=None, help="model name; default: first id from GET /models")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None, help="passed through as the request's 'seed' field (default: not sent)")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--chat-template-kwargs",
        default='{"enable_thinking": false}',
        help="JSON sent as chat_template_kwargs (default turns Qwen3 thinking off); pass '' to omit",
    )
    p.add_argument("--ids", choices=["random", "aliased"], default="random")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--out", default="out")
    p.add_argument("--run-prefix", default=None, help="run ids are <prefix>_1..N (default: the --ids value)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300.0, help="per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=2, help="retries on connection errors, 429 and 5xx")
    p.add_argument("--doc", default=str(DEFAULT_DOC))
    p.add_argument("--validate", nargs="+", metavar="DIR", help="validate run dirs (or dirs of runs) and exit")
    p.add_argument("--summary", nargs="+", metavar="RUN_DIR", help="print key numbers for a set of runs and exit")
    p.add_argument("--normalize-regex", action="append", default=[], help="with --summary: mask matches before hashing")
    args = p.parse_args(argv)
    if args.run_prefix is None:
        args.run_prefix = args.ids
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.validate:
        return main_validate(args.validate)
    if args.summary:
        return main_summary(args.summary, args.normalize_regex)
    asyncio.run(main_run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
