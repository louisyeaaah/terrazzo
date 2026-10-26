#!/bin/bash
# Clean one-server-at-a-time benchmark of Mac inference servers.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY="${TERRAZZO_STUDY_ROOT:-$ROOT/../deepinfer-study}"
PY=$STUDY/.venv/bin/python
MODEL=$(ls -d $STUDY/.hf-cache/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/*/ | head -1)
BENCH=$ROOT/bench/api_bench.py
OUT=/tmp/bench_results.jsonl
: > $OUT

kill_all() {
  pkill -f "mlx_lm.server" 2>/dev/null
  pkill -f "terrazzo serve" 2>/dev/null
  sleep 2
}

make_system() {  # $1 = nonce, $2 = outfile
  $PY - "$1" "$2" <<'PYEOF'
import sys, uuid
nonce, out = sys.argv[1], sys.argv[2]
# random marker at the *front* so no server can reuse a cached prefix
marker = uuid.uuid4().hex[:16]
text = (f"[session {marker}] "
        "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
        "以下是背景资料，请在所有回答中遵循：" + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算；" * 20
        + f"（运行标记：{nonce}）")
open(out, "w").write(text)
PYEOF
}

wait_health() {  # $1 = port
  for i in $(seq 1 40); do
    if curl -s --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

run_target() {  # $1 label, $2 port, $3 mode(batch|paged|mlx), $4 prereg(yes|no)
  local label=$1 port=$2 mode=$3 prereg=$4
  local sysfile=/tmp/bensys_$label.txt
  make_system "$label" "$sysfile"
  kill_all
  cd "$ROOT"
  case $mode in
    mlx)
      PYTHONUNBUFFERED=1 nohup "$PY" -m mlx_lm.server --model "$MODEL" --port "$port" --host 127.0.0.1 > /tmp/bench_$label.log 2>&1 &
      ;;
    batch)
      local extra=""
      [ "$prereg" = yes ] && extra="--system-file $sysfile"
      PYTHONUNBUFFERED=1 nohup "$PY" -m terrazzo serve --model "$MODEL" --port "$port" $extra > /tmp/bench_$label.log 2>&1 &
      ;;
    paged)
      local extra=""
      [ "$prereg" = yes ] && extra="--system-file $sysfile"
      PYTHONUNBUFFERED=1 nohup "$PY" -m terrazzo serve --model "$MODEL" --port "$port" --paged $extra > /tmp/bench_$label.log 2>&1 &
      ;;
  esac
  wait_health "$port" || { echo "{\"label\":\"$label\",\"error\":\"server failed to start\"}" >> $OUT; return; }
  sleep 3
  # warm the model (not the prompt cache) with a throwaway question
  curl -s --max-time 120 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"热身：请说一句你好。"}],"max_tokens":8}' > /dev/null
  local m=default_model
  [ "$mode" != mlx ] && m=terrazzo
  "$PY" "$BENCH" --base "http://127.0.0.1:$port" --label "$label" --model "$m" --system-file "$sysfile" 2>&1 | tail -1 >> $OUT
}

run_target mlx-lm            8103 mlx   no
run_target terrazzo-batch   8101 batch no
run_target terrazzo-batch-prefix 8104 batch yes
run_target terrazzo-paged   8102 paged no
run_target terrazzo-paged-prefix 8105 paged yes

# Ollama: warm it, then benchmark (fresh nonce -> cold prompt cache, warm model)
kill_all
make_system ollama /tmp/bensys_ollama.txt
curl -s --max-time 180 http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"热身：请说一句你好。"}],"max_tokens":8}' > /dev/null
"$PY" "$BENCH" --base http://127.0.0.1:11434 --label ollama-qwen3-4b --model qwen3:4b --system-file /tmp/bensys_ollama.txt 2>&1 | tail -1 >> $OUT

kill_all
echo "DONE"
