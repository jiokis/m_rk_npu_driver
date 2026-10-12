#!/bin/bash
# 并发吞吐实验：宿主开销能不能与 NPU 工作重叠？
#
# ═══ 为什么这是"最大化 NPU 利用率"的正解 ═══
#
# docs/33 量出：一次推理 26 ms 里，NPU 硬件只忙 9~10 ms（约 36%）。
# 剩下 2/3 是宿主侧开销，而其中**大头是闭源 librknnrt 的内存抖动**
# （每推理数百次 mmap/munmap/openat，见 docs/34），**驱动侧根本救不了**。
#
# 但换个角度：如果宿主开销是"CPU 在算"，而 NPU 是"硬件在算"，
# 那么**把多个推理并发起来，一个的宿主开销就能盖住另一个的 NPU 时间** ——
# NPU 利用率立刻上去了，而且不需要改任何一行代码。
#
# 这就是本实验要验证的：并发 N 路时，总吞吐能到几倍？
#   · 若接近 N 倍 → 单进程时 NPU 大量空转，并发就是答案
#   · 若接近 1 倍 → 有串行瓶颈（比如驱动锁、NPU 排队），并发没用
#
# 用法：
#   sudo scripts/board/bench-concurrency.sh            # 1/2/3/4 路并发各测一轮
#   PROCS="1 2 4" ITERS=30 sudo scripts/board/bench-concurrency.sh
set -uo pipefail

ITERS="${ITERS:-30}"
WARM="${WARM:-5}"
PROCS="${PROCS:-1 2 3 4}"
MODEL="${MODEL:-/opt/rknn-models/yolov5s-640-640.rknn}"
CM="${CM:-7}"          # core_mask: 7 = 三核

PY=""
for c in /usr/bin/python3 "$HOME/rknn-gateway/venv/bin/python3"; do
  [ -x "$c" ] && "$c" -c "import rknnlite" 2>/dev/null && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "✗ 找不到带 rknnlite 的 python" >&2; exit 1; }
[ -e "$MODEL" ] || { echo "✗ 找不到 $MODEL" >&2; exit 1; }

WORKER=$(mktemp /tmp/rknpu-worker-XXXXXX.py)
cat > "$WORKER" <<'PY'
import sys, time, numpy as np
from rknnlite.api import RKNNLite
model, iters, warm, cm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
r = RKNNLite(); r.load_rknn(model); r.init_runtime(core_mask=cm)
x = np.zeros((1, 640, 640, 3), dtype=np.uint8)
for _ in range(warm):
    r.inference(inputs=[x])
t0 = time.perf_counter()
ts = time.time()
for _ in range(iters):
    r.inference(inputs=[x])
te = time.time()
dt = time.perf_counter() - t0
r.release()
# ★ 自报「推理循环」的绝对起止时间 —— 父进程用它算吞吐，
#   这样进程启动 + 模型加载的时间不会被算进去（第一版就栽在这，
#   固定偏移把 1 路的吞吐压低了，虚高了并发的相对收益）
print(f"WINDOW {dt:.4f} {ts:.6f} {te:.6f}", flush=True)
PY

echo "═══ 并发吞吐实验（$MODEL, core_mask=$CM, 每进程 $ITERS 次推理）═══"
echo "  逻辑：一次推理 NPU 只忙约 36%（docs/33），剩下是宿主侧开销。"
echo "        若并发能把宿主开销与 NPU 工作重叠，总吞吐应接近 N 倍。"
echo
printf '  %-6s %-12s %-14s %-14s %-10s\n' "并发" "墙钟(s)" "总吞吐(次/s)" "单进程(次/s)" "相对1路"
echo "  ────────────────────────────────────────────────────────────────"

BASE=""
for N in $PROCS; do
  pids=()
  for i in $(seq 1 "$N"); do
    "$PY" "$WORKER" "$MODEL" "$ITERS" "$WARM" "$CM" > "/tmp/rknpu-w$i.out" 2>/dev/null &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  # 取所有 worker 里最早的开始与最晚的结束 —— 这才是真实的并发工作窗口
  # 只认 WINDOW 开头的行 —— RKNN 自己也会往 stdout 打日志，不加标记会被 awk 吃进去
  wall=$(awk 'BEGIN{s=1e18;e=0;n=0}
              /^WINDOW /{n++; if($3<s)s=$3; if($4>e)e=$4}
              END{if(n>0) printf "%.4f", e-s; else printf "0"}' /tmp/rknpu-w*.out)
  total=$((ITERS * N))
  tput=$(echo "scale=2; $total / $wall" | bc)
  if [ -z "$BASE" ]; then BASE="$tput"; rel="1.00×"; else rel=$(echo "scale=2; $tput / $BASE" | bc)"×"; fi
  per=$(echo "scale=2; $wall / $ITERS" | bc)
  printf '  %-6s %-12.2f %-14s %-14s %-10s\n' "$N 路" "$wall" "$tput" "$per" "$rel"
  rm -f /tmp/rknpu-w*.out
done

echo
echo "  解读："
echo "   · 「相对1路」≈ N  → 宿主开销能与 NPU 重叠，并发直接提升 NPU 利用率"
echo "   · 「相对1路」≈ 1  → 存在串行瓶颈，并发无效（要查驱动锁/NPU 排队）"
echo "   · 介于两者之间      → 部分重叠，取收益最大的那一路数"
rm -f "$WORKER"
