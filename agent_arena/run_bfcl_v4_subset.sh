#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 LOG_FILE PID_FILE <bfcl_v4_subset_runner args...>" >&2
  exit 2
fi

LOG_FILE="$1"
PID_FILE="$2"
shift 2

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
echo "$$" > "$PID_FILE"

exec >"$LOG_FILE" 2>&1
echo "BFCL subset runner PID=$$"
echo "Started: $(date -Is)"
echo "Command: .venv-qai/bin/python agent_arena/bfcl_v4_subset_runner.py $*"
exec .venv-qai/bin/python agent_arena/bfcl_v4_subset_runner.py "$@"
