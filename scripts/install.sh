#!/bin/bash
# Install terrazzo into a dedicated venv and print the next steps.
# Docker is NOT supported for GPU inference on macOS (no Metal passthrough),
# so this script installs a normal wheel/venv + gives you the launchd template.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${TERRAZZO_VENV:-$HOME/.terrazzo/venv}"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q -U pip
"$VENV/bin/python" -m pip install -q -e "$ROOT"
echo "installed into $VENV"
echo
echo "next:"
echo "  $VENV/bin/terrazzo serve --model <mlx-model-dir> --port 8501 --num-gpu-blocks 8192"
echo "optional: --api-key <key> --rate-limit 5 --system-file system.txt"
echo "auto-start: cp $ROOT/packaging/mac/com.terrazzo.serve.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.terrazzo.serve.plist"
