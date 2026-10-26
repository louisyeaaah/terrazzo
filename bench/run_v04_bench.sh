#!/bin/bash
# v0.4 comparison: batch / paged (v0.3) vs pool / pool+metal (v0.4).
# One server at a time; random cold prefix for fair TTFT.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY="${TERRAZZO_STUDY_ROOT:-$ROOT/../deepinfer-study}"
PY=$STUDY/.venv/bin/python
MODEL=$(ls -d $STUDY/.hf-cache/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/*/ | head -1)
BENCH=$ROOT/bench/api_bench.py
OUT=/tmp/bench_v04_results.jsonl
: > $OUT

kill_all() {
  pkill -f "mlx_lm.server" 2>/dev/null
  pkill -f "terrazzo serve" 2>/dev/null
  sleep 2
}

make_system() {  # $1 nonce, $2 outfile
  $PY - "$1" "$2" <<'PYEOF'
import sys, uuid
nonce, out = sys.argv[1], sys.argv[2]
marker = uuid.uuid4().hex[:16]
text = (f"[session {marker}] "
        "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
        "以下是背景资料，请在所有回答中遵循：" + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算；" * 20
        + f"（运行标记：{nonce}）")
open(out, "w").write(text)
PYEOF
}

wait_health() {
  for i in $(seq 1 60); do
    curl -s --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

run_target() {  # label port mode_flags
  local label=$1 port=$2 flags=$3
  local sysfile=/tmp/v04sys_$label.txt
  make_system "$label" "$sysfile"
  kill_all
  cd "$ROOT"
  PYTHONUNBUFFERED=1 nohup "$PY" -m terrazzo serve --model "$MODEL" --port "$port" \
    --system-file "$sysfile" $flags > /tmp/v04_$label.log 2>&1 &
  wait_health "$port" || { echo "{\"label\":\"$label\",\"error\":\"start failed\"}" >> $OUT; return; }
  sleep 3
  curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"热身：请说一句你好。"}],"max_tokens":8}' > /dev/null
  "$PY" "$BENCH" --base "http://127.0.0.1:$port" --label "$label" --model terrazzo \
    --system-file "$sysfile" 2>&1 | tail -1 >> $OUT
}

run_target batch       8201 ""
run_target paged       8202 "--paged"
run_target pool        8203 "--pool"
run_target pool_metal  8204 "--pool --metal"

kill_all
echo DONE
