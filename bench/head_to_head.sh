#!/bin/bash
# Refreshed head-to-head on one machine, one service at a time:
#   terrazzo (current default)  vs  terrazzo (pre-registered prefix)
#   vs  Ollama qwen3:4b (llama.cpp, GGUF Q4_K_M)
# Protocol: random front marker (no cross-target prefix reuse), warm model,
# api_bench cases (cold / warm-same / warm-diff / short decode / 4-way concurrency).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY="${TERRAZZO_STUDY_ROOT:-$ROOT/../deepinfer-study}"
PY=$STUDY/.venv/bin/python
MODEL=$(ls -d $STUDY/.hf-cache/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/*/ | head -1)
BENCH=$ROOT/bench/api_bench.py
OUT=/tmp/h2h_results.jsonl
LOG=/tmp/h2h.log
: > "$OUT"; : > "$LOG"

say() { echo "=== $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

kill_servers() {
  pkill -f "terrazzo serve" 2>/dev/null
  pkill -f "mlx_lm.server" 2>/dev/null
  sleep 2
}

make_system() {  # $1 = label, $2 = outfile
  "$PY" - "$1" "$2" <<'PYEOF'
import sys, uuid
label, out = sys.argv[1], sys.argv[2]
marker = uuid.uuid4().hex[:16]
text = (f"[run {label} {marker}] "
        "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
        "以下是背景资料，请在所有回答中遵循：" + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算；" * 20
        + f"（运行标记：{marker}）")
open(out, "w").write(text)
PYEOF
}

wait_health() {  # $1 port
  for _ in $(seq 1 45); do
    curl -s --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

run_terrazzo() {  # $1 label, $2 port, $3 prereg yes|no
  local label=$1 port=$2 prereg=$3
  local sysfile=/tmp/h2h_sys_$label.txt
  make_system "$label" "$sysfile"
  kill_servers
  local extra=""
  [ "$prereg" = yes ] && extra="--system-file $sysfile"
  ( cd "$ROOT" && PYTHONUNBUFFERED=1 nohup "$PY" -m terrazzo serve \
      --model "$MODEL" --port "$port" $extra > "/tmp/h2h_$label.log" 2>&1 & )
  if ! wait_health "$port"; then
    echo "{\"label\":\"$label\",\"error\":\"start failed\"}" >> "$OUT"; return
  fi
  sleep 3
  curl -s --max-time 120 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"热身：请说一句你好。"}],"max_tokens":8}' > /dev/null
  "$PY" "$BENCH" --base "http://127.0.0.1:$port" --label "$label" \
    --model terrazzo --system-file "$sysfile" 2>&1 | tee -a "$LOG" | tail -1 >> "$OUT"
  kill_servers
}

run_ollama() {
  local label=ollama sysfile=/tmp/h2h_sys_ollama.txt
  make_system "$label" "$sysfile"
  # free any resident model, then warm it with a throwaway
  curl -s --max-time 10 http://127.0.0.1:11434/api/generate \
    -d '{"model":"qwen3:4b","prompt":"x","keep_alive":0}' >/dev/null 2>&1
  curl -s --max-time 180 http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"热身：请说一句你好。"}],"max_tokens":8}' >/dev/null 2>&1
  "$PY" "$BENCH" --base "http://127.0.0.1:11434" --label "$label" \
    --model qwen3:4b --system-file "$sysfile" 2>&1 | tee -a "$LOG" | tail -1 >> "$OUT"
}

say "versions: $(ollama --version 2>/dev/null)"
say "mlx model: $MODEL"
run_terrazzo terrazzo-default 8615 no
run_terrazzo terrazzo-prereg 8616 yes
run_ollama
say "done"
