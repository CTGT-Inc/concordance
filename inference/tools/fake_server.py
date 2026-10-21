"""A fake OpenAI-compatible server for exercising the benchmark without a GPU.

Serves the endpoints the harness talks to, and can be switched into the specific failure
modes the harness is supposed to catch. Output shape (--shape) and fault injection
(--diverge, --ignore-seed, --drop-index) are independent, so a run can combine, say,
inline <think> tags with a divergence at a chosen character index.

It also counts client disconnects, which is how we check that cancellation actually
reaches the server rather than leaving it generating into a batch slot.

    python3 tools/fake_server.py --port 30000 --diverge --diverge-at 120
"""

from __future__ import annotations

import argparse
import json
import random
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN_WIDTH = 6  # every generated token is exactly this many characters
MODEL_ID = "fake/determinism-test"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class State:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0
        self.started = 0
        self.completed = 0
        self.aborted = 0
        self.injected = 0
        self.nonce = 0
        self.request_number = 0
        # Bumped by /abort_request; a zombie watches it to know it has been killed.
        self.abort_generation = 0
        self.abort_calls = 0
        self.aborted_rids: set[str] = set()
        self.live_rids: set[str] = set()
        self.targeted_aborts = 0
        self.abort_ignored = 0

    def enter(self) -> int:
        with self.lock:
            self.inflight += 1
            self.started += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            return self.inflight

    def leave(self, *, aborted: bool) -> None:
        with self.lock:
            self.inflight -= 1
            if aborted:
                self.aborted += 1
            else:
                self.completed += 1

    def next_request_number(self) -> int:
        with self.lock:
            self.request_number += 1
            return self.request_number

    def next_nonce(self) -> int:
        with self.lock:
            self.nonce += 1
            return self.nonce

    def note_injection(self) -> None:
        with self.lock:
            self.injected += 1

    def abort_everything(self) -> None:
        with self.lock:
            self.abort_generation += 1
            self.abort_calls += 1

    def generation(self) -> int:
        with self.lock:
            return self.abort_generation

    def register(self, rid: str) -> None:
        with self.lock:
            self.live_rids.add(rid)

    def forget(self, rid: str | None) -> None:
        # Mirrors the tokenizer manager dropping its per-request state the moment the
        # client disconnects.
        if rid is None:
            return
        with self.lock:
            self.live_rids.discard(rid)

    def note_ignored_abort(self) -> None:
        with self.lock:
            self.abort_calls += 1
            self.abort_ignored += 1

    def abort_one(self, rid: str) -> None:
        with self.lock:
            self.abort_calls += 1
            if rid not in self.live_rids:
                # SGLang returns early when the rid is no longer in rid_to_state, so an
                # abort that arrives after the disconnect never reaches the scheduler.
                self.abort_ignored += 1
                return
            self.aborted_rids.add(rid)
            self.targeted_aborts += 1

    def is_aborted(self, rid: str | None) -> bool:
        if rid is None:
            return False
        with self.lock:
            # SGLang matches with req.rid.startswith(recv_req.rid), so mirror that.
            return any(rid.startswith(prefix) for prefix in self.aborted_rids)

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "started": self.started,
                "completed": self.completed,
                "aborted": self.aborted,
                "injected": self.injected,
                "max_inflight": self.max_inflight,
                "inflight": self.inflight,
                "abort_calls": self.abort_calls,
                "targeted_aborts": self.targeted_aborts,
                "abort_ignored": self.abort_ignored,
            }


STATE: State


def make_tokens(prompt: str, seed: object, count: int) -> list[str]:
    """Fixed-width tokens so a character index maps cleanly onto a token index."""
    rng = random.Random(f"{prompt}|{seed}")
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [
        "".join(rng.choice(letters) for _ in range(TOKEN_WIDTH - 1)) + " "
        for _ in range(count)
    ]


def inject_at_char(tokens: list[str], target: int) -> None:
    """Alter exactly one character so the first difference lands on `target`."""
    offset = 0
    for i, token in enumerate(tokens):
        if offset + len(token) > target:
            pos = target - offset
            replacement = "Z" if token[pos] != "Z" else "Q"
            tokens[i] = token[:pos] + replacement + token[pos + 1 :]
            STATE.note_injection()
            return
        offset += len(token)


def build_response(
    prompt: str, seed: object, max_tokens: int | None, concurrent: bool
) -> list[str]:
    args = STATE.args

    key: object = seed
    if args.ignore_seed:
        key = f"{seed}|{STATE.next_nonce()}"

    tokens = make_tokens(prompt, key, args.tokens)

    if args.shape == "inline-think":
        half = len(tokens) // 2
        tokens = [THINK_OPEN] + tokens[:half] + [THINK_CLOSE] + tokens[half:]

    # Truncation happens after shaping so that a request capped at the control's token
    # count reproduces the control exactly.
    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    should_inject = args.diverge and concurrent
    if args.flaky_every:
        # Every Nth response differs mid-text, so a batch-size-1 control repeat
        # disagrees with a long matching prefix rather than from character 0.
        should_inject = should_inject or STATE.next_request_number() % args.flaky_every == 0
    if should_inject:
        inject_at_char(tokens, args.diverge_at)
    return tokens


def split_channels(tokens: list[str]) -> list[tuple[str, str]]:
    """Return (channel, text) per token."""
    if STATE.args.shape == "separate-reasoning":
        half = len(tokens) // 2
        return [("reasoning", t) for t in tokens[:half]] + [
            ("content", t) for t in tokens[half:]
        ]
    return [("content", t) for t in tokens]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *fmt_args: object) -> None:
        if STATE.args.verbose:
            super().log_message(fmt, *fmt_args)

    # ------------------------------------------------------------- helpers

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        """Mirror a server launched with an API key: everything needs the bearer token."""
        required = STATE.args.require_key
        if not required:
            return True
        if self.headers.get("Authorization") == f"Bearer {required}":
            return True
        self._send_json({"error": "Unauthorized"}, status=401)
        return False

    def _client_gone(self, timeout: float) -> bool:
        """Wait, but notice a disconnect as it happens rather than on the next write.

        The real server drops its per-request state the moment the connection closes,
        which is what makes an abort sent afterwards a no-op. Detecting the close only on
        the next write would hide that ordering entirely.
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], timeout)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    # ----------------------------------------------------------------- GET

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        backend = STATE.args.backend
        if not self._authorized():
            return

        if path == "/get_server_info":
            if backend != "sglang":
                self._send_json({"error": "not found"}, status=404)
                return
            self._send_json(
                {
                    "version": "0.4.fake",
                    "server_args": {
                        "model_path": MODEL_ID,
                        "disable_radix_cache": not STATE.args.prefix_cache,
                        "speculative_algorithm": STATE.args.speculative,
                        "attention_backend": "fake",
                        "enable_deterministic_inference": False,
                        "max_running_requests": 256,
                    },
                }
            )
        elif path == "/version":
            if backend != "vllm":
                self._send_json({"error": "not found"}, status=404)
                return
            self._send_json({"version": "0.9.fake"})
        elif path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [{"id": MODEL_ID, "object": "model", "owned_by": "fake"}],
                }
            )
        elif path == "/metrics":
            snap = STATE.snapshot()
            body = (
                "# HELP sglang:num_running_reqs Running requests\n"
                "# TYPE sglang:num_running_reqs gauge\n"
                f'sglang:num_running_reqs{{model_name="{MODEL_ID}"}} {snap["inflight"]}.0\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/stats":
            self._send_json(STATE.snapshot())
        else:
            self._send_json({"error": "not found"}, status=404)

    # ---------------------------------------------------------------- POST

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if not self._authorized():
            return
        if path in {"/flush_cache", "/reset_prefix_cache"}:
            self._read_json()
            self._send_json({"ok": True})
            return
        if path == "/abort_request":
            body = self._read_json()
            if STATE.args.ignore_rid_abort:
                # A server that accepts /abort_request but does nothing with the rid.
                STATE.note_ignored_abort()
                self._send_json({"ok": True})
                return
            if body.get("abort_all"):
                STATE.abort_everything()
            elif body.get("rid"):
                STATE.abort_one(str(body["rid"]))
            self._send_json({"ok": True})
            return
        if path != "/v1/chat/completions":
            self._send_json({"error": "not found"}, status=404)
            return

        body = self._read_json()
        messages = body.get("messages") or []
        prompt = messages[-1].get("content", "") if messages else ""
        seed = body.get("seed")
        max_tokens = body.get("max_tokens")
        stream = bool(body.get("stream"))
        rid = body.get("rid")
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        inflight = STATE.enter()
        if rid:
            STATE.register(str(rid))
        aborted = False
        try:
            tokens = build_response(prompt, seed, max_tokens, concurrent=inflight > 1)
            truncated = max_tokens is not None and len(tokens) >= max_tokens
            if stream:
                aborted = self._stream(tokens, want_usage, rid, truncated)
            else:
                self._whole(tokens, truncated)
        finally:
            STATE.leave(aborted=aborted)

    def _whole(self, tokens: list[str], truncated: bool = False) -> None:
        pairs = split_channels(tokens)
        reasoning = "".join(t for c, t in pairs if c == "reasoning")
        content = "".join(t for c, t in pairs if c == "content")
        message: dict[str, object] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        self._send_json(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "length" if truncated else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": len(tokens),
                    "total_tokens": len(tokens) + 1,
                },
            }
        )

    def _stream(
        self,
        tokens: list[str],
        want_usage: bool,
        rid: str | None = None,
        truncated: bool = False,
    ) -> bool:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        pairs = split_channels(tokens)
        # Dropping a chunk affects the streaming path only; the non-streamed response
        # stays complete, which is the shape the integrity check exists to catch.
        drop = STATE.args.drop_index

        try:
            for i, (channel, text) in enumerate(pairs):
                if STATE.is_aborted(rid):
                    # An honoured abort stops generation, as the real server does.
                    break
                if i == drop:
                    continue
                key = "content" if channel == "content" else "reasoning_content"
                self._write_event(
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [
                            {"index": 0, "delta": {key: text}, "finish_reason": None}
                        ],
                    }
                )
                if STATE.args.token_delay and self._client_gone(
                    STATE.args.token_delay
                ):
                    raise BrokenPipeError

            self._write_event(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length" if truncated else "stop",
                        }
                    ],
                }
            )
            if want_usage:
                self._write_event(
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": len(tokens),
                            "total_tokens": len(tokens) + 1,
                        },
                    }
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-stream. This is what a real abort looks like.
            STATE.forget(rid)
            # Disconnecting does not necessarily stop the server: SGLang has been seen
            # decoding on for a while afterwards, holding its batch slot. --zombie-seconds
            # reproduces that so the harness can be checked against it.
            if STATE.args.zombie_seconds:
                gen = STATE.generation()
                deadline = time.time() + STATE.args.zombie_seconds
                while (
                    time.time() < deadline
                    and STATE.generation() == gen
                    and not STATE.is_aborted(rid)
                ):
                    time.sleep(0.02)
            return True
        return False

    def _write_event(self, payload: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
        self.wfile.flush()


def main() -> int:
    global STATE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--backend", choices=["sglang", "vllm"], default="sglang")
    p.add_argument(
        "--shape",
        choices=["content", "inline-think", "separate-reasoning"],
        default="content",
        help="how thinking text is presented, if at all",
    )
    p.add_argument("--tokens", type=int, default=40, help="tokens in an uncapped response")
    p.add_argument(
        "--diverge",
        action="store_true",
        help="alter one character once more than one request is in flight",
    )
    p.add_argument("--diverge-at", type=int, default=120, help="character index to alter")
    p.add_argument(
        "--ignore-seed",
        action="store_true",
        help="return different text for identical requests, as an unseeded server would",
    )
    p.add_argument(
        "--drop-index",
        type=int,
        default=-1,
        help="omit this chunk from the stream only (-1 disables)",
    )
    p.add_argument(
        "--flaky-every",
        type=int,
        default=0,
        help="alter every Nth response at --diverge-at, even at batch size 1",
    )
    p.add_argument("--token-delay", type=float, default=0.005)
    p.add_argument(
        "--zombie-seconds",
        type=float,
        default=0.0,
        help="keep holding a batch slot for this long after the client disconnects",
    )
    p.add_argument("--prefix-cache", action="store_true", help="advertise caching as enabled")
    p.add_argument("--speculative", default=None)
    p.add_argument("--require-key", default=None, help="reject requests without this bearer token")
    p.add_argument(
        "--ignore-rid-abort",
        action="store_true",
        help="accept /abort_request but never act on the rid",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    STATE = State(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(
        f"fake {args.backend} on http://{args.host}:{args.port} shape={args.shape} "
        f"tokens={args.tokens} diverge={args.diverge} ignore_seed={args.ignore_seed}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
