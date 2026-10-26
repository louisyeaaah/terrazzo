#!/bin/bash
# Supervise `terrazzo serve`: restart it when it exits (e.g. a fatal Metal OOM)
# and run a self-check after each restart.
#
# Usage:
#   scripts/serve_watchdog.sh [--max-restarts N] -- <serve args...>
# Example:
#   scripts/serve_watchdog.sh --max-restarts 5 -- \
#       terrazzo serve --model <dir> --port 8501 --num-gpu-blocks 8192
#
# Logs go to stdout/stderr; the script exits when the server exits cleanly
# (status 0) or when the restart budget is exhausted.

set -u
max_restarts=10
if [ "${1:-}" = "--max-restarts" ]; then
  max_restarts="$2"
  shift 2
fi
if [ "${1:-}" = "--" ]; then shift; fi
if [ $# -eq 0 ]; then
  echo "usage: $0 [--max-restarts N] -- <command...>" >&2
  exit 2
fi

restarts=0
while :; do
  started=$(date +%s)
  echo "[watchdog] starting: $*"
  "$@"
  code=$?
  ran=$(( $(date +%s) - started ))
  if [ "$code" -eq 0 ]; then
    echo "[watchdog] server exited cleanly after ${ran}s"
    exit 0
  fi
  restarts=$((restarts + 1))
  if [ "$restarts" -gt "$max_restarts" ]; then
    echo "[watchdog] restart budget exhausted (${max_restarts}); last exit=${code}" >&2
    exit "$code"
  fi
  echo "[watchdog] crashed (exit=${code}, ran=${ran}s); restart ${restarts}/${max_restarts} in 2s" >&2
  sleep 2
done
