# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "litellm>=1.63",
#   "httpx>=0.27",
# ]
# ///
"""Benchmark CLI script

Measures how deterministic one LLM serving configuration is.

A control run at batch size 1 establishes the reference output for each prompt. The same
prompts are then replayed under concurrency and compared character by character against
their control; the index of the first differing character is the score. A larger index
means the configuration held its output together for longer. Identical output scores the
full length of the control.

One invocation benchmarks one server configuration. Reconfigure and relaunch the server
between runs, and give each run a --name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# litellm reads its log level at import time and is extremely chatty by default.
os.environ.setdefault("LITELLM_LOG", "ERROR")

import httpx
import litellm

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
PROMPT_SET_VERSION = "1"


class BenchmarkAbort(RuntimeError):
    """A gate failed and the run cannot produce a meaningful number."""


# Baked in so that results stay comparable across configurations. Each one is chosen to
# produce a long answer: the metric has no resolution if the control is 40 characters.
DEFAULT_PROMPTS: list[dict[str, str]] = [
    {
        "id": "math-chain",
        "prompt": (
            "A freight train leaves station A at 06:00 travelling at 72 km/h. A second "
            "train leaves station B, 500 km away, at 07:30 travelling toward A at 88 km/h. "
            "A bird starts at the front of the first train at 07:30, flies at 120 km/h to "
            "the second train, turns around instantly, and continues bouncing between them "
            "until they meet. Work out, step by step: when and where the trains meet, the "
            "total distance the bird flies, and how many complete legs the bird makes "
            "before the gap closes below one kilometre. Show every intermediate "
            "calculation and state your assumptions."
        ),
    },
    {
        "id": "code-lru",
        "prompt": (
            "Write a thread-safe LRU cache in Python with a per-entry TTL, a maximum "
            "memory budget in bytes rather than a maximum item count, and metrics for hit "
            "rate and eviction cause. Include the full implementation, docstrings, and a "
            "test suite covering concurrent access, TTL expiry, and eviction under memory "
            "pressure. Then explain the trade-offs you made and how the design would need "
            "to change for a multi-process deployment."
        ),
    },
    {
        "id": "explain-consensus",
        "prompt": (
            "Explain how Raft achieves consensus, in enough depth that a reader could "
            "implement it. Cover leader election, log replication, the commit rule, "
            "membership changes, and log compaction. For each mechanism, describe the "
            "failure it exists to prevent and give a concrete example of what goes wrong "
            "without it. Finish with the specific ways Raft differs from Paxos and why "
            "those differences were made."
        ),
    },
    {
        "id": "debug-trace",
        "prompt": (
            "A Python service using asyncio leaks file descriptors under load. It rises by "
            "roughly one descriptor per request, only in production, only when a "
            "downstream call times out. Restarting fixes it for a few hours. Walk through "
            "how you would diagnose this from first principles: what you would measure, in "
            "what order, what each result would rule in or out, and the three most likely "
            "root causes with the code pattern that produces each. Include the exact "
            "commands you would run."
        ),
    },
    {
        "id": "sql-optimize",
        "prompt": (
            "Given a Postgres table `events(id bigserial, user_id bigint, kind text, "
            "payload jsonb, created_at timestamptz)` with 4 billion rows, write and explain "
            "the query plan for finding, per user, the first event of each kind in the last "
            "30 days. Discuss index choices, partitioning strategy, why the obvious "
            "window-function approach is slow at this scale, and at least two rewrites with "
            "their trade-offs. Include DDL for the indexes you recommend."
        ),
    },
    {
        "id": "logic-puzzle",
        "prompt": (
            "Five researchers each ran one experiment on a different day of the working "
            "week, using a different instrument, and reported a different result. Given "
            "that the spectrometer was used the day after the run that reported a null "
            "result; the Tuesday run used neither the microscope nor the centrifuge; the "
            "researcher who reported an anomaly ran on Friday; the calorimeter was used at "
            "some point before the microscope; and the null result was not reported on "
            "Monday, determine the full assignment. Show the complete deduction, stating "
            "which constraint you applied at each step and what it eliminated."
        ),
    },
    {
        "id": "design-ratelimit",
        "prompt": (
            "Design a distributed rate limiter for an API gateway handling 500k requests "
            "per second across 40 regions, supporting per-key limits at second, minute, and "
            "day granularity, with hard guarantees against over-admission on the day "
            "window. Cover the algorithm, the storage layer, how you handle clock skew and "
            "network partitions, what you sacrifice under CAP, and the failure modes at "
            "each layer. Give pseudocode for the hot path and an estimate of its latency "
            "budget."
        ),
    },
    {
        "id": "long-form-writing",
        "prompt": (
            "Write a detailed technical retrospective of a hypothetical four-hour outage "
            "caused by a schema migration that acquired an unexpected lock. Include the "
            "timeline with timestamps, the escalation path, what each responder saw and "
            "concluded, the three false leads and why each was plausible, the eventual "
            "root cause, the immediate mitigation, and eight follow-up action items with "
            "owners and rationale. Write it as an internal document, not a summary."
        ),
    },
]


REDACTED = "<REDACTED>"
SECRET_FLAGS = frozenset({"--api-key", "--admin-key"})
SECRET_CONFIG_KEYS = frozenset({"api_key", "admin_key"})
# Matched as substrings against field names. Deliberately specific: a bare "token"
# would hit tokenizer_path, max_total_tokens, speculative_num_draft_tokens and most of
# the rest of an inference server's arguments, redacting the configuration this report
# exists to record.
SECRET_NAME_PATTERNS = (
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "password",
    "passwd",
    "credential",
    "auth_token",
    "access_token",
    "refresh_token",
    "bearer",
    "private_key",
)
# Field names that are secrets on their own but too short to match as substrings.
SECRET_EXACT_NAMES = frozenset({"key", "token", "auth", "authorization"})
# Below this length a value is a mode or a backend name, not a credential.
MIN_SECRET_LENGTH = 8


def _is_secret(value: Any) -> bool:
    # "EMPTY" is the placeholder for a server with no auth, not a credential.
    return isinstance(value, str) and value not in ("", "EMPTY")


def _is_secret_flag(token: str) -> bool:
    # argparse accepts any unambiguous prefix, so --api-k reaches --api-key and would
    # otherwise sail past an exact-match check straight into the report.
    if not token.startswith("--") or len(token) < 4:
        return False
    return any(flag.startswith(token) for flag in SECRET_FLAGS)


def scrub_url(value: str) -> str:
    """Strip credentials from a URL, which --api-base is free to carry."""
    try:
        parsed = httpx.URL(value)
    except Exception:  # noqa: BLE001 - not a URL, nothing to scrub
        return value
    if not parsed.userinfo:
        return value
    return str(parsed.copy_with(userinfo=b""))


def redact_argv(argv: list[str]) -> str:
    """The command line is recorded in every report, and it carries the key."""
    out: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            out.append(REDACTED)
            hide_next = False
            continue
        if _is_secret_flag(token):
            out.append(token)
            hide_next = True
            continue
        flag, sep, _value = token.partition("=")
        if sep and _is_secret_flag(flag):
            out.append(f"{flag}={REDACTED}")
            continue
        out.append(scrub_url(token) if "://" in token else token)
    return " ".join(out)


def redact_blob(value: Any) -> Any:
    """Redact anything secret-looking in a structure the server handed us.

    The server-info blob is echoed verbatim into both reports, and a server launched
    with its own --api-key reports that key back among its arguments -- so redacting
    only our own config would leave the credential in the file regardless.
    """
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if _looks_secret(str(key)) and _is_secret(inner)
                else redact_blob(inner)
            )
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [redact_blob(item) for item in value]
    if isinstance(value, str) and "://" in value:
        return scrub_url(value)
    return value


def _looks_secret(name: str) -> bool:
    lowered = name.lower()
    if lowered in SECRET_EXACT_NAMES:
        return True
    return any(pattern in lowered for pattern in SECRET_NAME_PATTERNS)


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: REDACTED if key in SECRET_CONFIG_KEYS and _is_secret(value) else value
        for key, value in config.items()
    }


def _auth_headers(args: argparse.Namespace) -> dict[str, str]:
    """Bearer credentials for the requests litellm does not handle.

    Every request the harness makes is authenticated. Completions get the key through
    litellm, which sets the header itself from `api_key`; these cover the other
    transport, carrying the server-info, model, cache, metrics and abort endpoints.
    """
    key = getattr(args, "admin_key", None) or args.api_key
    if not key or key == "EMPTY":
        return {}
    return {"Authorization": f"Bearer {key}"}


def divergence_window(
    control: str,
    other: str,
    index: int,
    context: int = 160,
    other_label: str = "repeat",
) -> list[str]:
    """Two aligned excerpts around the first differing character, which is bracketed.

    Newlines are escaped so the two lines stay on one row each and line up.
    """
    start = max(0, index - context)

    def render(text: str) -> str:
        before = text[start:index]
        at = text[index] if index < len(text) else "<end of output>"
        after = text[index + 1 : index + 1 + context]
        body = f"{before}[{at}]{after}"
        return body.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")

    lead = "..." if start > 0 else ""
    width = max(len("control"), len(other_label))
    return [
        f"  {'control':<{width}}: {lead}{render(control)}",
        f"  {other_label:<{width}}: {lead}{render(other)}",
    ]


def unstable_control_dump(control: Control) -> str:
    """Everything needed to judge an unstable control by eye, not by index.

    Only the repeat's tail is shown. Everything before the divergence index is
    byte-identical to the control by definition, so printing the repeat in full would
    mostly duplicate what is directly above it.
    """
    index = control.first_divergence or 0
    other = control.divergent_text or ""
    tail = other[index:]
    rule = "=" * 78
    tail_header = (
        f"--- repeat, from the divergence onward ({len(tail)} of {len(other)} chars; "
        f"the first {index} matched) ---"
    )
    return "\n".join(
        [
            "",
            rule,
            f"CONTROL NOT REPRODUCIBLE: {control.prompt_id}",
            rule,
            f"first difference at character {index} of {len(control.text)}",
            "",
            f"--- original control response ({len(control.text)} chars) ---",
            control.text,
            "",
            f"--- at character {index} ---",
            *divergence_window(control.text, other, index),
            "",
            tail_header,
            tail if tail else "<the repeat ended here>",
            "",
        ]
    )


def divergence_dump(
    prompt_id: str, control_text: str, response: str, index: int
) -> str:
    """Show what a measured request produced where it parted from the control.

    The control itself is not repeated here -- it is in the control section above and in
    the JSON -- only the point of disagreement and what the response did from there.
    """
    tail = response[index:]
    rule = "-" * 78
    header = (
        f"--- response, from the divergence onward ({len(tail)} of {len(response)} "
        f"chars; the first {index} matched) ---"
    )
    return "\n".join(
        [
            "",
            rule,
            f"{prompt_id}: first difference at character {index} of {len(control_text)}",
            rule,
            *divergence_window(control_text, response, index, other_label="response"),
            "",
            header,
            tail if tail else "<the response ended here>",
            "",
        ]
    )


def _get(obj: Any, key: str) -> Any:
    """Read a field from a litellm chunk, which may be a pydantic object or a dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_deltas(chunk: Any) -> list[tuple[str, str]]:
    """Pull text out of one streamed chunk as (channel, text) pairs, reasoning first.

    litellm normalizes thinking to `reasoning_content`, but several providers surface it
    as `reasoning`, inside `thinking_blocks`, or only in `provider_specific_fields`, and
    litellm has dropped chunks in each of those shapes. Check all of them so the compared
    character stream is not silently missing text.
    """
    choices = _get(chunk, "choices")
    if not choices:
        return []
    delta = _get(choices[0], "delta")
    if delta is None:
        return []

    out: list[tuple[str, str]] = []

    reasoning = _get(delta, "reasoning_content") or _get(delta, "reasoning")
    if not reasoning:
        psf = _get(delta, "provider_specific_fields") or {}
        if isinstance(psf, dict):
            reasoning = psf.get("reasoning_content") or psf.get("reasoning")
    if not reasoning:
        blocks = _get(delta, "thinking_blocks")
        if blocks:
            parts = [
                _get(b, "thinking") or _get(b, "text") or ""
                for b in blocks
                if b is not None
            ]
            reasoning = "".join(parts) or None
    if reasoning:
        out.append(("reasoning", str(reasoning)))

    content = _get(delta, "content")
    if content:
        out.append(("content", str(content)))
    return out


def message_text(message: Any) -> tuple[str, int]:
    """Concatenate a non-streamed message the same way the stream is concatenated."""
    reasoning = _get(message, "reasoning_content") or _get(message, "reasoning") or ""
    if not reasoning:
        psf = _get(message, "provider_specific_fields") or {}
        if isinstance(psf, dict):
            reasoning = psf.get("reasoning_content") or psf.get("reasoning") or ""
    content = _get(message, "content") or ""
    return str(reasoning) + str(content), len(str(reasoning))


@dataclass
class StreamOutcome:
    text: str = ""
    reasoning_len: int = 0
    chunk_count: int = 0
    completion_tokens: int | None = None
    finish_reason: str | None = None
    # Set only when a control was supplied to compare against.
    divergence_index: int | None = None
    divergence_chunk: int | None = None
    identical: bool = False
    aborted: bool = False
    error: str | None = None


@dataclass
class Control:
    prompt_id: str
    prompt: str
    text: str = ""
    reasoning_len: int = 0
    completion_tokens: int | None = None
    channel_mode: str = "unknown"
    think_boundary: int | None = None
    deterministic: bool = False
    first_divergence: int | None = None
    repeats: int = 0
    # True when the control stopped at a token limit rather than finishing on its own.
    truncated: bool = False
    # The repeat that failed to reproduce, kept in full so the disagreement can be read
    # rather than inferred from an index.
    divergent_text: str | None = None


@dataclass
class Measurement:
    round_index: int
    prompt_id: str
    divergence_index: int
    control_len: int
    normalized: float
    identical: bool
    phase: str
    chunk_index: int | None
    error: str | None = None
    # What this request actually produced, kept so a divergence can be read rather than
    # inferred from an index. Truncated at the abort point when cancelling early.
    text: str = ""


@dataclass
class ServerFacts:
    backend: str = "unknown"
    model: str | None = None
    version: str | None = None
    server_args: dict[str, Any] = field(default_factory=dict)
    prefix_cache_disabled: bool | None = None
    warnings: list[str] = field(default_factory=list)
    request_body: dict[str, Any] | None = None
    # The server's own view of batch occupancy during the rounds, sampled from /metrics.
    # A client-side count cannot see requests that keep decoding after we disconnect.
    observed_occupancy_max: int | None = None
    observed_occupancy_mean: float | None = None
    occupancy_samples: int = 0
    abort_all_calls: int = 0
    targeted_aborts: int = 0
    abort_honours_rid: bool | None = None
    abort_check_chunks_after: int | None = None
    abort_baseline_chunks: int | None = None


class RecordingTransport(httpx.AsyncHTTPTransport):
    """Captures the first chat-completions body litellm actually puts on the wire.

    This is the evidence that seed, reasoning_effort and extra_body reached the server
    rather than being filtered out somewhere in the client stack. A dropped seed does not
    raise; it just makes every configuration look non-deterministic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captured: dict[str, Any] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.captured is None and "chat/completions" in str(request.url):
            with suppress(Exception):
                self.captured = json.loads(request.content)
        return await super().handle_async_request(request)


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.api_base.rstrip("/").removesuffix("/v1")
        if args.port is not None:
            parsed = httpx.URL(self.root)
            self.root = str(parsed.copy_with(port=args.port)).rstrip("/")
        self.api_base = f"{self.root}/v1"
        self.facts = ServerFacts()
        # Limits must go on the transport: httpx returns a supplied transport as-is and
        # never applies the client-level limits to it, so passing them to AsyncClient
        # alongside transport= leaves the pool at the 100/20 defaults.
        pool = max(args.concurrency * 2, 16)
        self.transport = RecordingTransport(
            retries=0,
            limits=httpx.Limits(
                max_connections=pool, max_keepalive_connections=pool
            ),
        )
        # Every request the harness makes is authenticated, not just the completions.
        # The admin endpoints sit behind the same key, and a 401 there is easy to miss:
        # it looks exactly like an endpoint that is not present.
        self.admin = httpx.AsyncClient(timeout=30.0, headers=_auth_headers(args))
        self.session = httpx.AsyncClient(
            transport=self.transport, timeout=httpx.Timeout(args.timeout)
        )
        self.model: str = args.model or ""
        self.prompts: list[dict[str, str]] = []
        self.controls: dict[str, Control] = {}
        self.measurements: list[Measurement] = []
        self.extra_body: dict[str, Any] = json.loads(args.extra_body) if args.extra_body else {}
        # Seeded so the filler load is the same shape from one run to the next.
        self.filler_rng = random.Random(args.seed if args.seed is not None else 0)
        self.metrics_available = True
        self.occupancy: list[int] = []
        self.idle_waits = 0
        self.aborts_issued = 0
        self.abort_all_failed = False
        self.targeted_aborts = 0
        self.targeted_abort_failed = False

    # ---------------------------------------------------------------- setup

    def configure_litellm(self) -> None:
        # Unsupported parameters must raise, never vanish silently.
        litellm.drop_params = False
        litellm.telemetry = False
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        # Route litellm through our client so the transport can record the request body
        # and so the pool is large enough for the configured concurrency.
        litellm.aclient_session = self.session

    async def close(self) -> None:
        await self.admin.aclose()
        await self.session.aclose()

    async def _get_json(self, path: str) -> dict[str, Any] | None:
        try:
            resp = await self.admin.get(f"{self.root}{path}")
        except httpx.HTTPError:
            return None
        self._check_auth(path, resp.status_code)
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else {"value": data}

    async def detect_backend(self) -> None:
        info = await self._get_json("/get_server_info")
        if info is not None:
            self.facts.backend = "sglang"
            args_blob = info.get("server_args") if isinstance(info.get("server_args"), dict) else info
            # Redacted here rather than at render time: a server launched with its
            # own key reports it back among these arguments.
            self.facts.server_args = redact_blob(args_blob)
            self.facts.version = str(info.get("version") or "")
            disabled = self._find_key(args_blob, "disable_radix_cache")
            if disabled is not None:
                self.facts.prefix_cache_disabled = bool(disabled)
            return

        version = await self._get_json("/version")
        if version is not None:
            self.facts.backend = "vllm"
            self.facts.version = str(version.get("version") or version.get("value") or "")
            self.facts.warnings.append(
                "vLLM exposes no server-args endpoint, so prefix caching could not be "
                "verified over HTTP. vLLM V1 enables it by default; the server must be "
                "launched with --no-enable-prefix-caching or these numbers are confounded."
            )
            return

        self.facts.warnings.append(
            "Backend could not be identified (no /get_server_info, no /version). Server "
            "configuration is not recorded in this report."
        )

    @staticmethod
    def _find_key(blob: Any, key: str) -> Any:
        """Locate a server arg that may sit at the top level or one dict down."""
        if not isinstance(blob, dict):
            return None
        if key in blob:
            return blob[key]
        for value in blob.values():
            if isinstance(value, dict) and key in value:
                return value[key]
        return None

    async def resolve_model(self) -> None:
        if self.model:
            return
        data = await self._get_json("/v1/models")
        entries = (data or {}).get("data") or []
        if entries and isinstance(entries[0], dict) and entries[0].get("id"):
            self.model = str(entries[0]["id"])
            return
        raise BenchmarkAbort(
            "Could not resolve a model id from /v1/models. Pass --model explicitly."
        )

    def _check_auth(self, path: str, status: int) -> None:
        """Stop on a rejected key rather than degrading into a much weaker run.

        A 401 is indistinguishable from a missing endpoint at the call site, so without
        this the backend simply fails to be identified and the run continues with no
        prefix-cache check, no occupancy verification and no targeted abort.
        """
        if status not in (401, 403):
            return
        source = "--admin-key" if self.args.admin_key else "--api-key"
        raise BenchmarkAbort(
            f"The server rejected the credentials on {path} (HTTP {status}). The key "
            f"from {source} is used for every request, including the server-info, "
            "model, cache, metrics and abort endpoints. Pass the key the server was "
            "launched with, or --admin-key if its admin endpoints use a different one."
        )

    def _note_abort_failure(self, detail: str) -> None:
        if self.abort_all_failed:
            return
        self.abort_all_failed = True
        self.facts.warnings.append(
            f"--abort-all was requested but /abort_request did not succeed ({detail}). "
            "Leftover requests are not being cleared, so the harness is relying on "
            "waiting for the server to report itself idle instead."
        )

    def targeted_abort_enabled(self) -> bool:
        # rid is an SGLang extension to the chat request; other backends would ignore it
        # or reject the body outright.
        return self.args.targeted_abort and self.facts.backend == "sglang"

    async def abort_rid(self, rid: str) -> None:
        """Stop one specific request, leaving the rest of the batch running.

        Closing the connection is not a reliable stop signal, and a request that keeps
        decoding holds a batch slot we believe we released. Naming the request means the
        cancellation can happen mid-round without touching live measurements.
        """
        try:
            resp = await self.admin.post(
                f"{self.root}/abort_request",
                json={"rid": rid, "abort_all": False},
            )
        except httpx.HTTPError as exc:
            self._note_targeted_failure(f"{type(exc).__name__}: {exc}")
            return
        if resp.status_code != 200:
            self._note_targeted_failure(f"HTTP {resp.status_code}")
            return
        self.targeted_aborts += 1

    def _note_targeted_failure(self, detail: str) -> None:
        if self.targeted_abort_failed:
            return
        self.targeted_abort_failed = True
        self.facts.warnings.append(
            f"Per-request abort did not succeed ({detail}). Cancelled requests are being "
            "left to stop on their own, so they may hold batch slots and inflate the "
            "batch beyond the requested concurrency. --no-early-cancel avoids this at "
            "the cost of generating every response in full."
        )

    async def abort_all(self) -> None:
        """Clear every request on the server.

        Disconnecting is not a reliable stop signal, so between phases the server is told
        outright to drop everything. This aborts requests belonging to anyone, so it is
        opt-in and only appropriate on a machine dedicated to the benchmark. Never called
        while our own measured requests are in flight.
        """
        if not self.args.abort_all:
            return
        try:
            resp = await self.admin.post(
                f"{self.root}/abort_request",
                json={"rid": "", "abort_all": True},
            )
        except httpx.HTTPError as exc:
            self._note_abort_failure(f"{type(exc).__name__}: {exc}")
            return
        if resp.status_code != 200:
            self._note_abort_failure(f"HTTP {resp.status_code}")
            return
        self.aborts_issued += 1

    async def clear_server(self) -> None:
        """Bring the server to a known-empty state before a measurement phase."""
        await self.abort_all()
        await self.wait_until_idle()

    async def running_requests(self) -> int | None:
        """How many requests the server currently has running, or None if unavailable.

        SGLang publishes this on /metrics when launched with --enable-metrics. It is the
        only way to see the server's own view of batch occupancy: a client-side count is
        blind to requests that keep decoding after we disconnect.
        """
        if not self.metrics_available:
            return None
        try:
            resp = await self.admin.get(f"{self.root}/metrics")
        except httpx.HTTPError:
            self.metrics_available = False
            return None
        if resp.status_code != 200:
            self.metrics_available = False
            if resp.status_code in (401, 403):
                # Not fatal on its own: the run can proceed without occupancy
                # verification, it just cannot confirm the batch was the size requested.
                self.facts.warnings.append(
                    f"The server rejected the credentials on /metrics "
                    f"(HTTP {resp.status_code}), so batch occupancy could not be checked."
                )
            return None
        for line in resp.text.splitlines():
            if line.startswith("#") or "num_running_reqs" not in line:
                continue
            try:
                return int(float(line.rsplit(maxsplit=1)[-1]))
            except ValueError:
                continue
        self.metrics_available = False
        return None

    async def wait_until_idle(self) -> None:
        """Block until the server reports nothing running, before a batch-size-1 request.

        Disconnecting does not reliably stop generation, so without this a leftover from
        the previous request would share the batch with the next one and quietly turn a
        batch-size-1 control into a batch of two.
        """
        if not self.args.wait_idle:
            return
        deadline = time.monotonic() + self.args.idle_timeout
        while time.monotonic() < deadline:
            running = await self.running_requests()
            if running is None:
                # No metrics endpoint; fall back to a fixed settle so an abort at least
                # has a chance to take effect.
                await asyncio.sleep(self.args.settle_seconds)
                return
            if running == 0:
                return
            self.idle_waits += 1
            await asyncio.sleep(0.25)
        self.facts.warnings.append(
            "Timed out waiting for the server to go idle before a batch-size-1 request; "
            "some control runs may have shared the batch with a leftover request."
        )

    async def sample_occupancy(self, stop: asyncio.Event) -> None:
        """Record the server's actual batch occupancy while a round runs.

        Readings taken before the first request has reached the server say nothing about
        the batch and would drag the mean toward zero, so sampling starts once the server
        reports something running.
        """
        started = False
        while not stop.is_set():
            running = await self.running_requests()
            if running is None:
                return
            if not started:
                if running == 0:
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(stop.wait(), timeout=0.05)
                    continue
                started = True
            self.occupancy.append(running)
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=0.25)

    async def flush_cache(self) -> None:
        if not self.args.flush_cache:
            return
        path = "/flush_cache" if self.facts.backend == "sglang" else "/reset_prefix_cache"
        try:
            resp = await self.admin.post(f"{self.root}{path}")
        except httpx.HTTPError:
            return
        # A rejected key here would leave cached prefixes in place between phases, which
        # is one of the confounds the run exists to remove, so it is worth stopping for.
        self._check_auth(path, resp.status_code)

    # ------------------------------------------------------------ requests

    def _request_kwargs(
        self, prompt: str, *, max_tokens: int | None, rid: str | None = None
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.args.system_prompt:
            messages.append({"role": "system", "content": self.args.system_prompt})
        messages.append({"role": "user", "content": prompt})

        extra = dict(self.extra_body)
        if self.args.separate_reasoning and self.facts.backend == "sglang":
            extra.setdefault("separate_reasoning", True)
        if self.args.reasoning_effort:
            # Sent as an extra body field rather than a top-level parameter: litellm
            # validates reasoning_effort against its own model registry, which knows
            # nothing about a self-hosted model and rejects it outright. The server
            # reads it from the request body either way.
            extra.setdefault("reasoning_effort", self.args.reasoning_effort)
        if rid is not None:
            # Naming the request up front is what makes a targeted abort possible; the
            # response does not otherwise expose the id the server aborts by.
            extra["rid"] = rid

        kwargs: dict[str, Any] = {
            "model": f"{self.args.model_prefix}{self.model}",
            "messages": messages,
            "api_base": self.api_base,
            "api_key": self.args.api_key,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "timeout": self.args.timeout,
            "num_retries": 0,
        }
        if self.args.seed is not None:
            kwargs["seed"] = self.args.seed
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    async def stream_once(
        self,
        prompt: str,
        *,
        max_tokens: int | None,
        control: Control | None = None,
        want_usage: bool = False,
        stop_event: asyncio.Event | None = None,
        early_cancel: bool | None = None,
    ) -> StreamOutcome:
        """Stream one completion, optionally comparing against a control as it arrives."""
        cancel_early = self.args.early_cancel if early_cancel is None else early_cancel
        out = StreamOutcome()
        # Full-length and unique: the server matches abort by rid prefix, so ids that
        # share a prefix would be cancelled together.
        rid = uuid.uuid4().hex if self.targeted_abort_enabled() else None
        kwargs = self._request_kwargs(prompt, max_tokens=max_tokens, rid=rid)
        if want_usage:
            kwargs["stream_options"] = {"include_usage": True}

        matched = 0
        target = control.text if control is not None else None
        resp = None
        try:
            resp = await litellm.acompletion(stream=True, **kwargs)
            async for chunk in resp:
                if stop_event is not None and stop_event.is_set():
                    out.aborted = True
                    break

                usage = _get(chunk, "usage")
                if usage is not None:
                    tokens = _get(usage, "completion_tokens")
                    if tokens is not None:
                        out.completion_tokens = int(tokens)

                choices = _get(chunk, "choices")
                if choices:
                    reason = _get(choices[0], "finish_reason")
                    if reason:
                        out.finish_reason = str(reason)

                pieces = extract_deltas(chunk)
                if not pieces:
                    continue
                out.chunk_count += 1

                for channel, text in pieces:
                    if channel == "reasoning":
                        out.reasoning_len += len(text)
                    out.text += text

                    # Once the first difference is found it is the answer; keep
                    # accumulating text (--no-early-cancel) but stop comparing, or the
                    # index would be overwritten by later chunks.
                    if target is None or out.divergence_index is not None:
                        continue
                    for ch in text:
                        if matched >= len(target):
                            # Ran past the end of the control: they differ here.
                            out.divergence_index = len(target)
                            out.divergence_chunk = out.chunk_count
                            break
                        if ch != target[matched]:
                            out.divergence_index = matched
                            out.divergence_chunk = out.chunk_count
                            break
                        matched += 1

                if out.divergence_index is not None and cancel_early:
                    out.aborted = True
                    break
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            out.error = f"{type(exc).__name__}: {exc}"
        finally:
            # Order matters. The server looks the abort up in its table of live requests,
            # and disconnecting removes the entry, so an abort sent after the connection
            # closes finds nothing and silently does nothing while the request carries on
            # decoding. Name the request first, then drop the connection.
            if out.aborted and rid is not None:
                await self.abort_rid(rid)
            # Breaking out of the iterator does not close the upstream stream on its own;
            # litellm leaves it suspended until garbage collection.
            if resp is not None:
                aclose = getattr(resp, "aclose", None)
                if aclose is not None:
                    with suppress(Exception):
                        await aclose()

        if target is not None and out.divergence_index is None:
            if len(out.text) == len(target):
                out.divergence_index = len(target)
                out.identical = True
            else:
                # Stopped early; the outputs differ at the point the shorter one ended.
                out.divergence_index = min(len(out.text), len(target))
        return out

    async def complete_once(self, prompt: str, *, max_tokens: int | None) -> tuple[str, int]:
        kwargs = self._request_kwargs(prompt, max_tokens=max_tokens)
        resp = await litellm.acompletion(stream=False, **kwargs)
        choices = _get(resp, "choices") or []
        if not choices:
            return "", 0
        return message_text(_get(choices[0], "message"))

    # -------------------------------------------------------------- phases

    async def warmup(self) -> None:
        if not self.args.warmup:
            return
        print("Warming up...", flush=True)
        await self.stream_once(self.prompts[0]["prompt"], max_tokens=16)

    async def check_abort_support(self) -> None:
        """Prove that naming a request actually stops it, instead of assuming.

        The server only honours an abort while it still holds the request, and it may not
        use the rid we supplied at all. Either way the failure is silent: cancelled
        requests keep decoding and every later measurement is taken in a batch larger
        than configured. One short request settles it.
        """
        if not self.targeted_abort_enabled():
            return
        print("Checking that requests can be stopped by rid...", flush=True)

        prompt = self.prompts[0]["prompt"]
        probe_cap = max(self.args.abort_check_chunks * 3, 30)

        # How far this prompt runs when nothing interrupts it. Without this baseline a
        # response that merely ends on its own is indistinguishable from one that was
        # stopped, and the check would report success having tested nothing.
        baseline = 0
        try:
            base_resp = await litellm.acompletion(
                stream=True, **self._request_kwargs(prompt, max_tokens=probe_cap)
            )
            try:
                async for chunk in base_resp:
                    if extract_deltas(chunk):
                        baseline += 1
            finally:
                with suppress(Exception):
                    await base_resp.aclose()
        except Exception as exc:  # noqa: BLE001 - reported as a warning, not fatal
            self.facts.warnings.append(
                f"Abort support could not be checked: {type(exc).__name__}: {exc}"
            )
            return

        if baseline <= self.args.abort_check_chunks + 5:
            self.facts.warnings.append(
                f"Abort support could not be checked: this prompt only produces "
                f"{baseline} chunks, too few to tell a stopped response from one that "
                "ended on its own. Whether cancelled requests actually stop is unverified."
            )
            print(f"  inconclusive: response is only {baseline} chunks", flush=True)
            return

        rid = uuid.uuid4().hex
        kwargs = self._request_kwargs(prompt, max_tokens=probe_cap, rid=rid)
        before = after = 0
        sent = False
        served_id: str | None = None

        try:
            resp = await litellm.acompletion(stream=True, **kwargs)
        except Exception as exc:  # noqa: BLE001 - reported as a warning, not fatal
            self.facts.warnings.append(
                f"Abort support could not be checked: {type(exc).__name__}: {exc}"
            )
            return
        try:
            async for chunk in resp:
                if served_id is None:
                    served_id = str(_get(chunk, "id") or "")
                if not extract_deltas(chunk):
                    continue
                if not sent:
                    before += 1
                    if before >= 5:
                        # Sent while the connection is open, which is the only time the
                        # server will act on it.
                        await self.abort_rid(rid)
                        sent = True
                    continue
                after += 1
                if after > self.args.abort_check_chunks:
                    break
        except Exception as exc:  # noqa: BLE001 - reported as a warning, not fatal
            self.facts.warnings.append(
                f"Abort support could not be checked: {type(exc).__name__}: {exc}"
            )
            return
        finally:
            with suppress(Exception):
                await resp.aclose()

        self.facts.abort_baseline_chunks = baseline
        if not sent:
            # The response finished before the probe got far enough to abort anything, so
            # nothing was tested. Saying "yes" here would be worse than saying nothing.
            self.facts.warnings.append(
                "Abort support could not be checked: the probe response ended after "
                f"{before} chunks, before an abort was issued. Whether cancelled "
                "requests actually stop is unverified."
            )
            print("  inconclusive: response ended before the abort was sent", flush=True)
            return

        self.facts.abort_check_chunks_after = after
        # Meaningful only because the baseline says the response had further to run.
        self.facts.abort_honours_rid = after <= self.args.abort_check_chunks
        if self.facts.abort_honours_rid:
            print(
                f"  stopped after {after} further chunks "
                f"(would have run {baseline - before})",
                flush=True,
            )
            return

        echoed = bool(served_id and rid in served_id)
        self.facts.warnings.append(
            f"Requests cannot be stopped by rid: after aborting, the response kept "
            f"streaming for more than {self.args.abort_check_chunks} chunks. Cancelled "
            "requests will hold batch slots, so the measured batch will be larger than "
            "the requested concurrency. Use --no-early-cancel to let requests finish "
            "instead, at the cost of generating every response in full."
            + (
                ""
                if echoed
                else " The id in the response does not contain the rid that was sent, "
                "so the server may be ignoring the supplied rid entirely."
            )
        )
        print(f"  WARNING: still streaming after {after} chunks; abort by rid is not working", flush=True)

    async def stream_integrity_check(self) -> None:
        """Prove the streamed characters equal the non-streamed message.

        The metric is a character index into the model's output. If the client drops or
        rewrites a chunk, every index after it is wrong and nothing looks broken.

        Two identical non-streamed requests run first. Without a server that reproduces
        its own output there is no stable reference, and a streamed-vs-whole mismatch
        would say nothing about the client — so that case defers to the control phase,
        which is where server non-determinism gets diagnosed properly.
        """
        if not self.args.stream_integrity_check:
            return
        print("Checking stream integrity...", flush=True)
        prompt = self.prompts[0]["prompt"]
        cap = 256

        try:
            first, _ = await self.complete_once(prompt, max_tokens=cap)
            second, _ = await self.complete_once(prompt, max_tokens=cap)
        except Exception as exc:
            raise BenchmarkAbort(
                f"Stream integrity check could not run: {type(exc).__name__}: {exc}"
            ) from exc

        if first != second:
            note = (
                "Stream integrity could not be verified: two identical non-streamed "
                "requests returned different text, so there is no stable reference to "
                "compare the stream against. This points at the server, not the client."
            )
            self.facts.warnings.append(note)
            print("  server is not reproducible here; deferring to the control phase", flush=True)
            return

        streamed = await self.stream_once(prompt, max_tokens=cap)
        if streamed.error:
            raise BenchmarkAbort(f"Stream integrity check failed to run: {streamed.error}")

        if streamed.text != first:
            # litellm parses <think> tags out of a non-streamed message but passes them
            # through verbatim when streaming, so the two paths differ by exactly the
            # delimiters. Every measurement compares a streamed response against a
            # streamed control, so this does not touch the result.
            if streamed.text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "") == first:
                note = (
                    "The client strips <think> delimiters from non-streamed responses but "
                    "not from streamed ones, so stream integrity could only be confirmed "
                    "up to those delimiters. Measurements are streamed-against-streamed "
                    "and unaffected. Recorded because a run against a server with a "
                    "reasoning parser is not comparable to one without."
                )
                self.facts.warnings.append(note)
                print("  stream matches apart from <think> delimiters", flush=True)
                return

            index = next(
                (i for i, (a, b) in enumerate(zip(streamed.text, first)) if a != b),
                min(len(streamed.text), len(first)),
            )
            raise BenchmarkAbort(
                "Stream integrity check failed: the streamed text does not match the "
                f"non-streamed response (first difference at character {index}; streamed "
                f"{len(streamed.text)} chars, non-streamed {len(first)} chars).\n"
                "The server reproduced this prompt identically across two non-streamed "
                "requests, so text is being dropped or rewritten on the streaming path. "
                "That would silently shift every divergence index. Re-run with "
                "--no-stream-integrity-check only if you have confirmed this is benign."
            )
        print(f"  stream matches non-streamed response ({len(first)} chars)", flush=True)

    async def build_controls(self) -> None:
        repeats = self.args.control_repeats
        print(
            f"Control phase: {len(self.prompts)} prompts x {repeats} repeats at batch size 1",
            flush=True,
        )
        failures: list[Control] = []

        for spec in self.prompts:
            await self.clear_server()
            await self.flush_cache()
            control = Control(prompt_id=spec["id"], prompt=spec["prompt"])
            first = await self.stream_once(
                spec["prompt"],
                max_tokens=self.args.control_max_tokens,
                want_usage=True,
            )
            if first.error:
                raise BenchmarkAbort(f"Control run for {spec['id']} failed: {first.error}")
            if not first.text:
                raise BenchmarkAbort(f"Control run for {spec['id']} returned no text.")

            control.text = first.text
            control.reasoning_len = first.reasoning_len
            tokens = first.completion_tokens or first.chunk_count
            cap = self.args.control_max_tokens
            # Not every server reports finish_reason, so also treat a control that used
            # its whole allowance as truncated.
            control.truncated = first.finish_reason == "length" or (
                cap is not None and tokens >= cap
            )
            # Every later request is capped at this. Too tight and the response is cut
            # short, which is then indistinguishable from a divergence -- and the count
            # can be a client-side estimate when the server sends no usage chunk, so it
            # is not trustworthy to the token. A generous cap costs nothing, since a
            # response is abandoned as soon as it passes the control's length anyway.
            # A truncated control is the exception: matching its limit exactly is the
            # only way a repeat can reproduce it.
            control.completion_tokens = tokens if control.truncated else tokens * 2 + 32
            control.channel_mode = self._channel_mode(first)
            control.think_boundary = self._think_boundary(first, control.channel_mode)
            control.deterministic = True
            control.repeats = repeats

            for _ in range(repeats - 1):
                await self.clear_server()
                await self.flush_cache()
                # Control repeats never cancel early, and are capped at the control's own
                # token count. An aborted request is not necessarily stopped on the
                # server: it can keep decoding for a while after the client disconnects,
                # and that leftover occupies a batch slot. In a phase whose entire premise
                # is batch size 1, a leftover from the previous repeat would put the next
                # one in a batch of two, which is exactly the variable under test. The
                # divergence index is recorded either way; only the early exit is given up.
                again = await self.stream_once(
                    spec["prompt"],
                    max_tokens=control.completion_tokens,
                    control=control,
                    early_cancel=False,
                )
                if again.error:
                    raise BenchmarkAbort(f"Control repeat for {spec['id']} failed: {again.error}")
                if not again.identical:
                    control.deterministic = False
                    control.first_divergence = again.divergence_index
                    control.divergent_text = again.text
                    failures.append(control)
                    break

            self.controls[spec["id"]] = control
            state = "stable" if control.deterministic else "UNSTABLE"
            print(
                f"  {spec['id']}: {len(control.text)} chars, "
                f"{control.completion_tokens} tokens, {control.channel_mode}, {state}",
                flush=True,
            )

        if failures:
            self._report_unstable_control(failures)

    def _report_unstable_control(self, failures: list[Control]) -> None:
        for control in failures:
            print(unstable_control_dump(control), flush=True)

        body = self.transport.captured or {}
        seed_sent = "seed" in body
        detail = "\n".join(
            f"  {c.prompt_id}: repeats diverge at character {c.first_divergence} "
            f"of {len(c.text)}"
            for c in failures
        )
        seed_note = (
            f"The request body sent to the server did include seed={body.get('seed')!r}, "
            "so the client is not at fault: this server is not honoring per-request seeds, "
            "or it is non-deterministic even at batch size 1."
            if seed_sent
            else "The captured request body did not contain a seed, so the client may have "
            "filtered it before it reached the server."
        )
        message = (
            "Control runs at batch size 1 are not reproducible:\n"
            f"{detail}\n\n"
            f"Sampling was temperature={self.args.temperature}, seed={self.args.seed}. "
            f"{seed_note}\n\n"
            "Concurrency results would be meaningless against an unstable control. Either "
            "re-run with --temperature 0 to take the sampler out of the picture, or pass "
            "--allow-nondeterministic-control to proceed anyway."
        )
        if not self.args.allow_nondeterministic_control:
            raise BenchmarkAbort(message)
        print(f"\nWARNING: {message}\n", flush=True)
        self.facts.warnings.append("Control was not deterministic; results are indicative only.")

    @staticmethod
    def _channel_mode(outcome: StreamOutcome) -> str:
        if outcome.reasoning_len > 0:
            return "separate-reasoning"
        if THINK_CLOSE in outcome.text:
            return "inline-think"
        return "content-only"

    @staticmethod
    def _think_boundary(outcome: StreamOutcome, mode: str) -> int | None:
        if mode == "separate-reasoning":
            return outcome.reasoning_len
        if mode == "inline-think":
            found = outcome.text.find(THINK_CLOSE)
            return found + len(THINK_CLOSE) if found >= 0 else None
        return None

    def _phase(self, control: Control, index: int) -> str:
        if control.think_boundary is None:
            return "answer"
        return "reasoning" if index < control.think_boundary else "answer"

    async def run_round(self, round_index: int) -> list[Measurement]:
        # Clear anything left over from the previous round's filler before starting, so
        # the batch begins at the configured concurrency rather than above it.
        await self.clear_server()
        await self.flush_cache()
        order = [
            self.prompts[i % len(self.prompts)]["id"]
            for i in range(self.args.requests_per_round)
        ]
        if self.args.batch_order == "shuffle":
            # A permutation of the balanced list, not a resample: every prompt keeps the
            # same number of measurements, and only the order changes -- which is what
            # decides who shares a batch with whom. Seeded from --seed and the round
            # index, so the sequence repeats exactly under every configuration and
            # comparisons between them stay paired.
            random.Random(f"{self.args.seed}|{round_index}").shuffle(order)
        queue: asyncio.Queue[str] = asyncio.Queue()
        for prompt_id in order:
            queue.put_nowait(prompt_id)

        results: list[Measurement] = []
        remaining = len(order)
        done = asyncio.Event()
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal remaining
            while True:
                try:
                    prompt_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    # No measured work left. Keep the batch at --concurrency with filler
                    # until the round ends, then exit.
                    if done.is_set() or not self.args.sustain_load:
                        return
                    await self.filler_request(done)
                    continue

                try:
                    control = self.controls[prompt_id]
                    outcome = await self.stream_once(
                        control.prompt,
                        max_tokens=control.completion_tokens,
                        control=control,
                    )
                    index = (
                        outcome.divergence_index
                        if outcome.divergence_index is not None
                        else 0
                    )
                    measurement = Measurement(
                        round_index=round_index,
                        prompt_id=prompt_id,
                        divergence_index=index,
                        control_len=len(control.text),
                        normalized=index / len(control.text) if control.text else 0.0,
                        identical=outcome.identical,
                        phase=self._phase(control, index),
                        chunk_index=outcome.divergence_chunk,
                        error=outcome.error,
                        text=outcome.text if self.args.store_responses else "",
                    )
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    measurement = Measurement(
                        round_index=round_index,
                        prompt_id=prompt_id,
                        divergence_index=0,
                        control_len=0,
                        normalized=0.0,
                        identical=False,
                        phase="unknown",
                        chunk_index=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                async with lock:
                    results.append(measurement)
                    remaining -= 1
                    if remaining == 0:
                        done.set()

        # Workers retire on their own once the round is done, which lets in-flight filler
        # streams close through the same aclose() path as everything else. Cancelling them
        # instead would abandon the litellm coroutine and leak the upstream connection.
        workers = [asyncio.create_task(worker()) for _ in range(self.args.concurrency)]
        sampler = asyncio.create_task(self.sample_occupancy(done))
        _, pending = await asyncio.wait(workers, timeout=self.args.timeout + 30)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        done.set()
        await asyncio.gather(sampler, return_exceptions=True)
        return results

    async def filler_request(self, done: asyncio.Event) -> None:
        """Unmeasured load so the batch stays at --concurrency while stragglers finish."""
        spec = self.filler_rng.choice(self.prompts)
        control = self.controls.get(spec["id"])
        cap = control.completion_tokens if control else 512
        await self.stream_once(spec["prompt"], max_tokens=cap, stop_event=done)

    async def recheck_control(self) -> list[str]:
        if not self.args.recheck_control:
            return []
        print("Re-checking control...", flush=True)
        drift: list[str] = []
        for spec in self.prompts:
            # Same batch-size-1 premise as the control phase, so the same rules apply.
            await self.clear_server()
            await self.flush_cache()
            control = self.controls[spec["id"]]
            outcome = await self.stream_once(
                spec["prompt"],
                max_tokens=control.completion_tokens,
                control=control,
                early_cancel=False,
            )
            if not outcome.identical:
                drift.append(
                    f"{spec['id']} drifted at character {outcome.divergence_index} "
                    f"of {len(control.text)}"
                )
        if drift:
            print("  WARNING: control drifted during the run:", flush=True)
            for line in drift:
                print(f"    {line}", flush=True)
        else:
            print("  control still reproduces", flush=True)
        return drift


# ------------------------------------------------------------------ stats


BOOTSTRAP_RESAMPLES = 2000


def _percentile_bounds(
    stats: list[float], alpha: float
) -> tuple[float, float]:
    stats.sort()
    n = len(stats)
    return (stats[int((alpha / 2) * n)], stats[min(int((1 - alpha / 2) * n), n - 1)])


def bootstrap_median_ci(
    values: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval for a median."""
    if not values:
        return (0.0, 0.0)
    if len(set(values)) == 1:
        return (values[0], values[0])
    rng = random.Random(20240117)
    n = len(values)
    draws = [statistics.median(rng.choices(values, k=n)) for _ in range(resamples)]
    return _percentile_bounds(draws, alpha)


def fixed_prompt_median_ci(
    by_prompt: dict[str, list[float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Interval for the median of per-prompt medians, with the prompt set held fixed.

    The same prompts run under every configuration, so variation between prompts is a
    constant rather than sampling error, and folding it into the interval would only make
    configurations harder to tell apart. Resampling happens inside each prompt, which
    leaves run-to-run variation as the sole source of uncertainty -- the quantity that
    actually differs between configurations.
    """
    prompts = sorted(by_prompt)
    if not prompts:
        return (0.0, 0.0)
    if all(len(set(by_prompt[p])) == 1 for p in prompts):
        point = statistics.median([by_prompt[p][0] for p in prompts])
        return (point, point)
    rng = random.Random(20240117)
    draws: list[float] = []
    for _ in range(resamples):
        medians = [
            statistics.median(rng.choices(by_prompt[p], k=len(by_prompt[p])))
            for p in prompts
        ]
        draws.append(statistics.median(medians))
    return _percentile_bounds(draws, alpha)


def per_prompt_stats(measurements: list[Measurement]) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[Measurement]] = {}
    for m in measurements:
        if m.error is None:
            by_prompt.setdefault(m.prompt_id, []).append(m)

    rows = []
    for prompt_id, items in sorted(by_prompt.items()):
        indices = [float(i.divergence_index) for i in items]
        normalized = [i.normalized for i in items]
        lo, hi = bootstrap_median_ci(indices)
        nlo, nhi = bootstrap_median_ci(normalized)
        rows.append(
            {
                "prompt_id": prompt_id,
                "n": len(items),
                "control_len": items[0].control_len,
                "median": statistics.median(indices),
                "ci95": [lo, hi],
                "min": min(indices),
                "max": max(indices),
                "median_normalized": statistics.median(normalized),
                "ci95_normalized": [nlo, nhi],
                "identical_rate": sum(1 for i in items if i.identical) / len(items),
            }
        )
    return rows


def summarize(
    measurements: list[Measurement], rounds: int, per_prompt: list[dict[str, Any]]
) -> dict[str, Any]:
    ok = [m for m in measurements if m.error is None]

    indices_by_prompt: dict[str, list[float]] = {}
    normalized_by_prompt: dict[str, list[float]] = {}
    for m in ok:
        indices_by_prompt.setdefault(m.prompt_id, []).append(float(m.divergence_index))
        normalized_by_prompt.setdefault(m.prompt_id, []).append(m.normalized)

    prompt_medians = [row["median"] for row in per_prompt]
    prompt_medians_norm = [row["median_normalized"] for row in per_prompt]
    lo, hi = fixed_prompt_median_ci(indices_by_prompt)
    nlo, nhi = fixed_prompt_median_ci(normalized_by_prompt)

    # Kept as a diagnostic only. A round median mixes prompts whose control lengths
    # differ, so it is not the basis for any interval.
    per_round: list[float] = []
    for r in range(rounds):
        vals = [float(m.divergence_index) for m in ok if m.round_index == r]
        if vals:
            per_round.append(statistics.median(vals))

    identical = sum(1 for m in ok if m.identical)
    phases = [m.phase for m in ok if not m.identical]

    return {
        "measurements": len(measurements),
        "errors": len(measurements) - len(ok),
        "prompts": len(per_prompt),
        "median_of_prompt_medians": (
            statistics.median(prompt_medians) if prompt_medians else 0.0
        ),
        "ci95": [lo, hi],
        "median_normalized": (
            statistics.median(prompt_medians_norm) if prompt_medians_norm else 0.0
        ),
        "ci95_normalized": [nlo, nhi],
        "identical_rate": identical / len(ok) if ok else 0.0,
        "identical_count": identical,
        "diverged_in_reasoning": sum(1 for p in phases if p == "reasoning"),
        "diverged_in_answer": sum(1 for p in phases if p == "answer"),
        "rounds_completed": len(per_round),
        "round_medians": per_round,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def _first_divergence_per_prompt(
    measurements: list[Measurement],
) -> list[Measurement]:
    seen: set[str] = set()
    out: list[Measurement] = []
    for m in measurements:
        if m.error or m.identical or m.prompt_id in seen or not m.text:
            continue
        seen.add(m.prompt_id)
        out.append(m)
    return sorted(out, key=lambda m: m.prompt_id)


# ----------------------------------------------------------------- output


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    facts = payload["server"]
    cfg = payload["config"]
    lines: list[str] = []

    lines.append(f"# Determinism benchmark — {payload['name']}")
    lines.append("")
    lines.append(f"*{payload['timestamp']}*")
    lines.append("")
    if payload.get("notes"):
        lines.append(f"> {payload['notes']}")
        lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Identical to control | **{s['identical_rate']:.1%}** ({s['identical_count']}/{s['measurements'] - s['errors']}) |")
    lines.append(f"| Median divergence index | **{s['median_of_prompt_medians']:.0f}** chars |")
    lines.append(f"| 95% CI | {s['ci95'][0]:.0f} – {s['ci95'][1]:.0f} |")
    lines.append(
        f"| Median normalized position | {s['median_normalized']:.3f} of control "
        f"({s['ci95_normalized'][0]:.3f} – {s['ci95_normalized'][1]:.3f}) |"
    )
    lines.append(f"| Rounds completed | {s['rounds_completed']} / {cfg['rounds']} |")
    lines.append(f"| Concurrency | {cfg['concurrency']} |")
    lines.append(f"| Batch order | {cfg.get('batch_order')} |")
    lines.append(f"| Failed requests | {s['errors']} |")
    lines.append("")
    lines.append(
        f"Divergence landed in reasoning {s['diverged_in_reasoning']} times and in the "
        f"answer {s['diverged_in_answer']} times."
    )
    lines.append("")

    if facts.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in facts["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Server")
    lines.append("")
    lines.append(f"- Backend: `{facts['backend']}`")
    lines.append(f"- Model: `{facts.get('model')}`")
    lines.append(f"- Version: `{facts.get('version')}`")
    lines.append(f"- Prefix cache disabled: `{facts.get('prefix_cache_disabled')}`")
    if facts.get("occupancy_samples"):
        lines.append(
            f"- Batch occupancy observed on the server: max "
            f"`{facts.get('observed_occupancy_max')}`, mean "
            f"`{facts.get('observed_occupancy_mean')}` "
            f"over {facts['occupancy_samples']} samples "
            f"(requested concurrency {cfg['concurrency']})"
        )
    if facts.get("abort_honours_rid") is not None:
        verdict = "yes" if facts["abort_honours_rid"] else "NO"
        lines.append(
            f"- Requests can be stopped by rid: **{verdict}** "
            f"(kept streaming {facts.get('abort_check_chunks_after')} chunks after abort)"
        )
    if facts.get("targeted_aborts"):
        lines.append(
            f"- Cancelled requests stopped by rid: {facts['targeted_aborts']}"
        )
    if facts.get("abort_all_calls"):
        lines.append(
            f"- Server cleared between phases via `/abort_request`: "
            f"{facts['abort_all_calls']} times"
        )
    lines.append("")
    if facts.get("server_args"):
        lines.append("<details><summary>Server arguments</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(facts["server_args"], indent=2, sort_keys=True, default=str))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    if facts.get("request_body"):
        lines.append("<details><summary>Request body as sent to the server</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(facts["request_body"], indent=2, sort_keys=True, default=str))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## Control")
    lines.append("")
    lines.append("| Prompt | Chars | Tokens | Channels | Stable |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for c in payload["controls"]:
        stable = "yes" if c["deterministic"] else f"NO (diverged at {c['first_divergence']})"
        lines.append(
            f"| `{c['prompt_id']}` | {len(c['text'])} | {c['completion_tokens']} | "
            f"{c['channel_mode']} | {stable} |"
        )
    lines.append("")

    unstable = [c for c in payload["controls"] if not c["deterministic"]]
    for c in unstable:
        index = c["first_divergence"] or 0
        other = c["divergent_text"] or ""
        lines.append(f"### Control not reproducible: `{c['prompt_id']}`")
        lines.append("")
        lines.append(
            f"First difference at character {index} of {len(c['text'])}."
        )
        lines.append("")
        lines.append("```")
        lines.extend(divergence_window(c["text"], other, index))
        lines.append("```")
        lines.append("")
        lines.append(f"<details><summary>Original control response ({len(c['text'])} chars)</summary>")
        lines.append("")
        lines.append("```")
        lines.append(c["text"])
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        tail = other[index:]
        lines.append(
            f"<details><summary>Repeat, from the divergence onward "
            f"({len(tail)} of {len(other)} chars; the first {index} matched)</summary>"
        )
        lines.append("")
        lines.append("```")
        lines.append(tail if tail else "<the repeat ended here>")
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if payload.get("control_drift"):
        lines.append("Control drifted during the run:")
        lines.append("")
        for line in payload["control_drift"]:
            lines.append(f"- {line}")
        lines.append("")

    lines.append("## Per prompt")
    lines.append("")
    lines.append(
        "Each interval is a percentile bootstrap over that prompt's own measurements."
    )
    lines.append("")
    lines.append(
        "| Prompt | n | Control chars | Median | 95% CI | Min | Max | Normalized | Identical |"
    )
    lines.append("| --- | ---: | ---: | ---: | :-: | ---: | ---: | ---: | ---: |")
    for row in payload["per_prompt"]:
        lo, hi = row["ci95"]
        lines.append(
            f"| `{row['prompt_id']}` | {row['n']} | {row['control_len']} | "
            f"{row['median']:.0f} | {lo:.0f} – {hi:.0f} | {row['min']:.0f} | "
            f"{row['max']:.0f} | {row['median_normalized']:.3f} | "
            f"{row['identical_rate']:.0%} |"
        )
    lines.append("")

    examples = payload.get("divergence_examples") or []
    if examples:
        lines.append("## Divergences")
        lines.append("")
        lines.append(
            "One example per prompt. Every measured response is in the JSON sidecar."
        )
        lines.append("")
        for ex in examples:
            lines.append(
                f"### `{ex['prompt_id']}` — character {ex['index']} of "
                f"{ex['control_len']} ({ex['phase']})"
            )
            lines.append("")
            lines.append("```")
            lines.extend(
                divergence_window(
                    ex["control_text"], ex["text"], ex["index"], other_label="response"
                )
            )
            lines.append("```")
            lines.append("")
            tail = ex["text"][ex["index"] :]
            lines.append(
                f"<details><summary>Response from the divergence onward "
                f"({len(tail)} of {len(ex['text'])} chars)</summary>"
            )
            lines.append("")
            lines.append("```")
            lines.append(tail if tail else "<the response ended here>")
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    rounds = s.get("round_medians") or []
    if rounds:
        lines.append("## Round medians")
        lines.append("")
        lines.append(
            "Diagnostic only. A round median mixes prompts with different control "
            "lengths, so no interval is derived from it."
        )
        lines.append("")
        lines.append("```")
        lines.append(", ".join(f"{v:.0f}" for v in rounds))
        lines.append("```")
        lines.append("")
        spread = statistics.stdev(rounds) if len(rounds) > 1 else 0.0
        lines.append(
            f"min {min(rounds):.0f}, max {max(rounds):.0f}, stdev {spread:.1f}"
        )
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append("```")
    lines.append(payload["argv"])
    lines.append("```")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2, sort_keys=True, default=str))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def print_console_summary(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    print()
    print("=" * 68)
    print(f"  {payload['name']}")
    print("=" * 68)
    print(f"  identical to control     {s['identical_rate']:>10.1%}  ({s['identical_count']}/{s['measurements'] - s['errors']})")
    print(f"  median divergence index  {s['median_of_prompt_medians']:>10.0f}  chars")
    print(f"  95% CI                   {s['ci95'][0]:>10.0f} – {s['ci95'][1]:.0f}")
    print(
        f"  normalized position      {s['median_normalized']:>10.3f}  of control"
        f"  ({s['ci95_normalized'][0]:.3f} – {s['ci95_normalized'][1]:.3f})"
    )
    print(f"  rounds                   {s['rounds_completed']:>10d}")
    print(f"  failures                 {s['errors']:>10d}")
    print("-" * 78)
    print(
        f"  {'prompt':<20}{'n':>4}{'median':>9}{'95% CI':>17}"
        f"{'norm':>8}{'ident':>7}"
    )
    for row in payload["per_prompt"]:
        lo, hi = row["ci95"]
        ci = f"{lo:.0f} - {hi:.0f}"
        print(
            f"  {row['prompt_id']:<20}{row['n']:>4}{row['median']:>9.0f}{ci:>17}"
            f"{row['median_normalized']:>8.3f}{row['identical_rate']:>7.0%}"
        )
    print("=" * 78)


# ------------------------------------------------------------------- cli


def load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompts_file:
        path = Path(args.prompts_file)
        raw = path.read_text(encoding="utf-8")
        prompts: list[dict[str, str]] = []
        if path.suffix in {".jsonl", ".ndjson"}:
            for i, line in enumerate(raw.splitlines()):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompts.append(
                    {"id": str(obj.get("id") or f"p{i}"), "prompt": str(obj["prompt"])}
                )
        else:
            for i, line in enumerate(raw.splitlines()):
                line = line.strip()
                if line:
                    prompts.append({"id": f"p{i}", "prompt": line})
        if not prompts:
            raise BenchmarkAbort(f"No prompts found in {path}")
    else:
        prompts = [dict(p) for p in DEFAULT_PROMPTS]

    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    return prompts


def validate_plan(args: argparse.Namespace, prompts: list[dict[str, str]]) -> None:
    """Reject settings that would quietly produce a number meaning something else."""
    ids = [p["id"] for p in prompts]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise BenchmarkAbort(
            f"Prompt ids must be unique; these repeat: {', '.join(duplicates)}. "
            "Controls are held per id, so duplicates would collapse into one and some "
            "prompts would be scored against another prompt's control."
        )

    if args.requests_per_round < 1:
        raise BenchmarkAbort(
            f"--requests-per-round must be at least 1 (got {args.requests_per_round}); "
            "a round with no measured requests never ends on its own."
        )

    if args.requests_per_round < len(prompts):
        raise BenchmarkAbort(
            f"--requests-per-round ({args.requests_per_round}) is below the prompt count "
            f"({len(prompts)}), so only the first {args.requests_per_round} prompts would "
            "ever be measured while controls are still generated for all of them. Raise "
            "it to a multiple of the prompt count, or lower --num-prompts."
        )

    if args.requests_per_round % len(prompts):
        raise BenchmarkAbort(
            f"--requests-per-round ({args.requests_per_round}) is not a multiple of the "
            f"prompt count ({len(prompts)}), so some prompts would be measured more often "
            "than others and would carry more weight in the summary."
        )

    if args.control_repeats < 2:
        raise BenchmarkAbort(
            f"--control-repeats must be at least 2 (got {args.control_repeats}): the "
            "first run establishes the reference and the rest verify it, so a value of 1 "
            "checks nothing while still reporting the control as stable."
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", required=True, help="label for this run, used in the report")
    p.add_argument("--notes", default="", help="free text recorded in the report")
    p.add_argument("--out-dir", default="reports", help="where reports are written")

    p.add_argument(
        "--api-base", "--base-url", dest="api_base", default="http://127.0.0.1:30000"
    )
    p.add_argument("--port", type=int, default=None, help="override the port in --api-base")
    p.add_argument("--model", default=None, help="model id (default: first from /v1/models)")
    p.add_argument("--model-prefix", default="openai/", help="litellm provider prefix")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument(
        "--admin-key",
        default=os.environ.get("SGLANG_ADMIN_KEY"),
        help="key for admin endpoints such as /abort_request (defaults to --api-key)",
    )
    p.add_argument(
        "--abort-all",
        action="store_true",
        help=(
            "tell the server to drop every request between phases, instead of waiting "
            "for it to go idle. Aborts requests belonging to anyone, so use only on a "
            "machine dedicated to this benchmark"
        ),
    )
    p.add_argument("--backend", choices=["auto", "sglang", "vllm"], default="auto")
    p.add_argument("--timeout", type=float, default=900.0)

    p.add_argument("--concurrency", "-c", type=int, default=4)
    p.add_argument(
        "--requests-per-round",
        type=int,
        default=None,
        help="measured requests per round (default: one per prompt)",
    )
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument(
        "--batch-order",
        choices=["shuffle", "roundrobin"],
        default="shuffle",
        help=(
            "how prompts are ordered within a round. shuffle (default) permutes them "
            "per round, so batch composition varies while each prompt keeps the same "
            "number of measurements; roundrobin uses a fixed order, giving every round "
            "the same composition. Either way the sequence is identical across runs "
            "with the same --seed, so configurations stay comparable"
        ),
    )

    p.add_argument("--prompts-file", default=None)
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--system-prompt", default=None)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--control-max-tokens",
        type=int,
        default=None,
        help=(
            "cap the reference generation (default: uncapped). Every other request is "
            "capped at the control's own token count, so that a response matching the "
            "control ends where the control ended"
        ),
    )
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--extra-body", default=None, help="JSON merged into the request body")

    p.add_argument("--control-repeats", type=int, default=3)
    p.add_argument(
        "--abort-check-chunks",
        type=int,
        default=30,
        help=(
            "how many further chunks a request may stream after being aborted before "
            "abort-by-rid is judged not to work"
        ),
    )
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=60.0,
        help="how long to wait for the server to go idle before a batch-size-1 request",
    )
    p.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="fixed pause before a batch-size-1 request when /metrics is unavailable",
    )

    for name, default, helptext in [
        ("warmup", True, "send a throwaway request first"),
        ("sustain-load", True, "hold the batch at --concurrency with filler requests"),
        ("early-cancel", True, "abort a request as soon as it diverges"),
        ("flush-cache", True, "flush the server cache between phases"),
        ("recheck-control", True, "re-verify the control after the run"),
        ("separate-reasoning", True, "ask SGLang to split reasoning from content"),
        ("stream-integrity-check", True, "verify streamed text matches non-streamed"),
        ("wait-idle", True, "wait for the server to go idle before each control request"),
        ("targeted-abort", True, "abort each cancelled request by rid (SGLang only)"),
        ("store-responses", True, "keep every measured response in the JSON report"),
        ("show-divergences", True, "print one divergence per prompt after the rounds"),
    ]:
        dest = name.replace("-", "_")
        p.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=helptext)
        p.add_argument(f"--no-{name}", dest=dest, action="store_false")

    p.add_argument(
        "--allow-nondeterministic-control",
        action="store_true",
        help="continue even if the batch-size-1 control is not reproducible",
    )
    p.add_argument(
        "--allow-radix-cache",
        action="store_true",
        help="continue even if the server has prefix caching enabled",
    )
    return p


async def run(args: argparse.Namespace) -> int:
    runner = Runner(args)
    runner.configure_litellm()
    started = time.time()

    try:
        return await _run_inner(args, runner, started)
    except BenchmarkAbort as abort:
        print(f"\n{abort}", file=sys.stderr, flush=True)
        return 1
    finally:
        await runner.close()


async def _run_inner(
    args: argparse.Namespace, runner: Runner, started: float
) -> int:
    runner.prompts = load_prompts(args)
    if args.requests_per_round is None:
        args.requests_per_round = len(runner.prompts)
    validate_plan(args, runner.prompts)

    # Probe regardless of --backend so the server details still land in the report; an
    # explicit choice then overrides what the probe concluded.
    await runner.detect_backend()
    if args.backend != "auto":
        runner.facts.backend = args.backend

    await runner.resolve_model()
    runner.facts.model = runner.model
    print(f"Backend {runner.facts.backend}, model {runner.model}, base {runner.api_base}")

    if runner.facts.prefix_cache_disabled is False and not args.allow_radix_cache:
        raise BenchmarkAbort(
            "The server has prefix (radix) caching enabled. Cached prefixes change "
            "results between the first and later identical requests, which confounds "
            "this measurement. Relaunch with caching disabled, or pass "
            "--allow-radix-cache to proceed anyway."
        )
    for warning in runner.facts.warnings:
        print(f"WARNING: {warning}", flush=True)

    await runner.warmup()
    await runner.check_abort_support()
    await runner.stream_integrity_check()

    # Capture the body from a control request rather than from the warmup: same
    # parameters, but without the warmup's artificial token cap, so the report shows what
    # a real measurement actually sends.
    runner.transport.captured = None
    await runner.build_controls()
    runner.facts.request_body = redact_blob(runner.transport.captured)

    print(
        f"\n{args.rounds} rounds x {args.requests_per_round} requests "
        f"at concurrency {args.concurrency}",
        flush=True,
    )
    for r in range(args.rounds):
        results = await runner.run_round(r)
        runner.measurements.extend(results)
        ok = [m for m in results if m.error is None]
        median = statistics.median([m.divergence_index for m in ok]) if ok else 0
        ident = sum(1 for m in ok if m.identical)
        print(
            f"  round {r + 1:>3}/{args.rounds}: median {median:>8.0f}  "
            f"identical {ident}/{len(ok)}",
            flush=True,
        )

    if runner.occupancy:
        runner.facts.observed_occupancy_max = max(runner.occupancy)
        runner.facts.observed_occupancy_mean = round(
            statistics.mean(runner.occupancy), 2
        )
        runner.facts.occupancy_samples = len(runner.occupancy)
        if runner.facts.observed_occupancy_max < args.concurrency:
            # Under-filling is as much a problem as over-filling: the responses were not
            # generated in the batch the run claims to have measured.
            runner.facts.warnings.append(
                f"The server never reported more than "
                f"{runner.facts.observed_occupancy_max} requests running while "
                f"concurrency was set to {args.concurrency}, so the batch never reached "
                "the requested size and these measurements were taken under lighter load "
                "than intended. Check that the client is not the bottleneck and that "
                "--requests-per-round is large enough to keep the batch full."
            )
        elif runner.facts.observed_occupancy_max > args.concurrency:
            mean = runner.facts.observed_occupancy_mean or 0.0
            # A brief overshoot at round boundaries is expected as filler winds down.
            # A mean above the requested concurrency means cancelled requests are still
            # holding slots for a sustained period, which changes what was measured.
            severity = (
                "Mean occupancy is also above the requested concurrency, so cancelled "
                "requests are holding batch slots for a sustained period and every "
                "measurement was taken in a larger batch than configured."
                if mean > args.concurrency
                else "Mean occupancy is within the requested concurrency, so this looks "
                "like a brief overshoot as filler requests wind down between rounds "
                "rather than sustained contamination."
            )
            runner.facts.warnings.append(
                f"The server reported up to {runner.facts.observed_occupancy_max} "
                f"requests running (mean {mean}) while concurrency was set to "
                f"{args.concurrency}. {severity} Re-run with --no-early-cancel for a "
                "faithful batch, at the cost of generating every response in full."
            )
    elif runner.metrics_available is False:
        runner.facts.warnings.append(
            "Batch occupancy could not be verified: the server exposes no /metrics "
            "endpoint. Launch SGLang with --enable-metrics so the harness can confirm "
            "the batch was the size it asked for."
        )
    else:
        # Metrics were reachable but no reading ever caught a request in flight, so
        # nothing is known about the batch these numbers were measured in.
        runner.facts.warnings.append(
            "Batch occupancy could not be verified: no sample caught a request running, "
            "so the size of the batch these measurements were taken in is unknown. "
            "Rounds may be finishing faster than the sampling interval."
        )

    if args.show_divergences:
        # One example per prompt is enough to tell a flipped token from a different
        # answer; every response is in the JSON for anything more detailed.
        seen: set[str] = set()
        for m in runner.measurements:
            if m.error or m.identical or m.prompt_id in seen or not m.text:
                continue
            seen.add(m.prompt_id)
            control = runner.controls[m.prompt_id]
            print(
                divergence_dump(
                    m.prompt_id, control.text, m.text, m.divergence_index
                ),
                flush=True,
            )

    drift = await runner.recheck_control()
    runner.facts.abort_all_calls = runner.aborts_issued
    runner.facts.targeted_aborts = runner.targeted_aborts

    per_prompt = per_prompt_stats(runner.measurements)
    summary = summarize(runner.measurements, args.rounds, per_prompt)
    payload = {
        "name": args.name,
        "notes": args.notes,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 1),
        "argv": redact_argv(sys.argv),
        "prompt_set_version": PROMPT_SET_VERSION,
        "config": redact_config(vars(args)),
        "server": asdict(runner.facts),
        "controls": [asdict(c) for c in runner.controls.values()],
        "control_drift": drift,
        "summary": summary,
        "per_prompt": per_prompt,
        "divergence_examples": [
            {
                "prompt_id": m.prompt_id,
                "index": m.divergence_index,
                "control_len": m.control_len,
                "phase": m.phase,
                "control_text": runner.controls[m.prompt_id].text,
                "text": m.text,
            }
            for m in _first_divergence_per_prompt(runner.measurements)
        ],
        "measurements": [asdict(m) for m in runner.measurements],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in args.name)
    md_path = out_dir / f"{slug}_{stamp}.md"
    json_path = out_dir / f"{slug}_{stamp}.json"
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    print_console_summary(payload)
    print(f"\nreport {md_path}")
    print(f"data   {json_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
