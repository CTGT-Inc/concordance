"""Deterministic stand-in for an OpenAI-compatible chat endpoint. Stdlib only.

    python tools/mock_server.py --port 30001
    python run_pipeline.py --base-url http://127.0.0.1:30001/v1 ...

The reply is a pure function of the request body: the body is hashed
(SHA-256 of its canonical JSON) and the hash picks formatting choices and,
for section 9.4 only, whether the section's items come back as "rule" or
"background". Identical requests therefore get identical replies, and
requests that differ only in a printed id can get different ones. That is
the behaviour of a batch-invariant server, not of a language model; the
extraction itself is a sentence splitter with a keyword test.

Endpoints: GET /v1/models, POST /v1/chat/completions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "mock-deterministic"
FLIP_SECTION = "9.4"

RULE_WORDS = re.compile(
    r"\b(must|may|cannot|shall|requires?|required|charged|limited|deleted|kept|treated|"
    r"passed|reported|declined|applies|accrue|stop|lifts|released|refunded|extends?)\b",
    re.IGNORECASE,
)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_prompt(user: str) -> dict:
    sec_id = re.search(r"^Section id: (\S+)", user, re.M)
    heading = re.search(r"^Section heading: (.+)$", user, re.M)
    body = re.search(r"^<<<\n(.*?)\n>>>$", user, re.M | re.S)
    neighbours = re.findall(r"^- (\S+): (\d+(?:\.\d+)?) ", user, re.M)
    tables = re.findall(r"^Table id (\S+) \(Table (\d+)\)", user, re.M)
    return {
        "id": sec_id.group(1) if sec_id else "unknown",
        "heading": heading.group(1) if heading else "",
        "body": body.group(1) if body else "",
        "neighbours": {num: nid for nid, num in neighbours},
        "tables": {num: tid for tid, num in tables},
    }


def sentences(body: str) -> list[str]:
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("[table ")]
    out = []
    for p in paras:
        text = " ".join(p.split())
        out += [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", text) if s.strip()]
    return out


def extract(prompt: dict, digest: bytes) -> dict:
    items = []
    for s in sentences(prompt["body"]):
        cites = []
        for num in re.findall(r"\bTable (\d+)\b", s):
            if num in prompt["tables"]:
                cites.append(prompt["tables"][num])
        for num in re.findall(r"\bsection (\d+(?:\.\d+)?)\b", s):
            if num in prompt["neighbours"]:
                cites.append(prompt["neighbours"][num])
        kind = "rule" if RULE_WORDS.search(s) else "background"
        items.append({"text": s, "kind": kind, "cites": sorted(set(cites))})
    if prompt["heading"].startswith(FLIP_SECTION + " ") and digest[1] % 2 == 0:
        for it in items:
            it["kind"] = "background"
    return {"section": prompt["id"], "rules": items}


def render(obj: dict, digest: bytes) -> str:
    text = json.dumps(obj, indent=2 if digest[2] % 2 else None, ensure_ascii=False)
    if digest[0] % 2:
        text = "```json\n" + text + "\n```"
    return text


class Handler(BaseHTTPRequestHandler):
    server_version = "concordance-mock/1"

    def log_message(self, fmt, *args):  # quiet
        if self.server.verbose:
            super().log_message(fmt, *args)

    def send_json(self, code: int, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            body = json.loads(raw)
            user = next(m["content"] for m in reversed(body["messages"]) if m.get("role") == "user")
        except Exception as e:
            self.send_json(400, {"error": f"bad request: {e}"})
            return
        digest = hashlib.sha256(canonical(body).encode()).digest()
        content = render(extract(parse_prompt(user), digest), digest)
        self.send_json(
            200,
            {
                "id": "chatcmpl-" + digest.hex()[:24],
                "object": "chat.completion",
                "model": body.get("model", MODEL_ID),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(content) // 4},
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.verbose = args.verbose
    print(f"mock server on http://{args.host}:{args.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
