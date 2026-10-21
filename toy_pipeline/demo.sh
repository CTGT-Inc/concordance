#!/usr/bin/env bash
# Toy pipeline demo: 3 runs with random ids, 3 runs with aliased ids, then
# compare_runs.py on each set. Reports go to out/REPORT_*.md.
#
#   ./demo.sh                 against BASE_URL (default http://127.0.0.1:30000/v1)
#   ./demo.sh --mock          start tools/mock_server.py and run against it
#   MODEL=Qwen/Qwen3-8B ./demo.sh --concurrency 8   extra args go to run_pipeline.py
set -euo pipefail
cd "$(dirname "$0")"

USE_MOCK=0
if [[ "${1:-}" == "--mock" ]]; then
  USE_MOCK=1
  shift
fi

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
fi

if "$PY" -c 'import httpx' 2>/dev/null; then
  PIPE=("$PY" run_pipeline.py)
elif command -v uv >/dev/null 2>&1; then
  PIPE=(uv run --quiet run_pipeline.py)
else
  echo "run_pipeline.py needs httpx: pip install httpx (or install uv)" >&2
  exit 1
fi

COMPARE=../compare_calls/compare_runs.py
ID_REGEX='\b[0-9a-f]{16}\b'
OUT=out
BASE_URL="${BASE_URL:-http://127.0.0.1:30000/v1}"

MOCK_PID=""
cleanup() { [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ $USE_MOCK == 1 ]]; then
  MOCK_PORT="${MOCK_PORT:-30001}"
  "$PY" tools/mock_server.py --port "$MOCK_PORT" >/dev/null 2>&1 &
  MOCK_PID=$!
  BASE_URL="http://127.0.0.1:${MOCK_PORT}/v1"
  for _ in $(seq 1 50); do
    "$PY" -c "import urllib.request,sys; urllib.request.urlopen('$BASE_URL/models', timeout=1)" 2>/dev/null && break
    sleep 0.1
  done
  echo "mock server: $BASE_URL"
fi

MODEL_ARGS=()
[[ -n "${MODEL:-}" ]] && MODEL_ARGS=(--model "$MODEL")

rm -rf "$OUT"
mkdir -p "$OUT"
start=$(date +%s)

echo "== pipeline runs ($BASE_URL)"
"${PIPE[@]}" --base-url "$BASE_URL" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} --ids random  --runs 3 --out "$OUT" --run-prefix random  "$@"
"${PIPE[@]}" --base-url "$BASE_URL" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} --ids aliased --runs 3 --out "$OUT" --run-prefix aliased "$@"

echo
echo "== schema check"
"$PY" run_pipeline.py --validate "$OUT"

RANDOM_RUNS=("$OUT"/random_1 "$OUT"/random_2 "$OUT"/random_3)
ALIASED_RUNS=("$OUT"/aliased_1 "$OUT"/aliased_2 "$OUT"/aliased_3)
CMP_ARGS=(--final commit --stage-order parse,interpret,commit)

echo
if [[ -f "$COMPARE" ]]; then
  echo "== compare_runs.py"
  "$PY" "$COMPARE" "${RANDOM_RUNS[@]}"  "${CMP_ARGS[@]}" --out "$OUT/REPORT_random.md"  --json "$OUT/REPORT_random.json"
  "$PY" "$COMPARE" "${ALIASED_RUNS[@]}" "${CMP_ARGS[@]}" --out "$OUT/REPORT_aliased.md" --json "$OUT/REPORT_aliased.json"
  "$PY" "$COMPARE" "${RANDOM_RUNS[@]}"  "${CMP_ARGS[@]}" --normalize-regex "$ID_REGEX" \
    --out "$OUT/REPORT_random_normalised.md" --json "$OUT/REPORT_random_normalised.json"
  echo "wrote $OUT/REPORT_random.md, $OUT/REPORT_aliased.md, $OUT/REPORT_random_normalised.md"
else
  echo "== $COMPARE not found; skipping the markdown reports"
fi

echo
echo "== key numbers: random ids"
"$PY" run_pipeline.py --summary "${RANDOM_RUNS[@]}"
echo
echo "== key numbers: random ids, ids masked with $ID_REGEX"
"$PY" run_pipeline.py --summary "${RANDOM_RUNS[@]}" --normalize-regex "$ID_REGEX"
echo
echo "== key numbers: aliased ids"
"$PY" run_pipeline.py --summary "${ALIASED_RUNS[@]}"
echo
echo "elapsed: $(( $(date +%s) - start ))s"
