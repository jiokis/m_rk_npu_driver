#!/bin/bash
# DDR 频率影响：严格 A/B 实测
#
# ═══ 为什么不能像第一版那样测 ═══
# 第一版是「先钉 performance 测一轮，再 echo ondemand 测一轮」，得出 30.11 vs 27.89 ms，
# 既看不出提升、又和 verify 那轮的 37.55 ms 对不上 —— 方法本身有问题：
#   1) `echo ondemand > governor` **不保证频率真的降下来**。前一步刚把 min/max 顶到 2400，
#      切回 ondemand 后它未必会掉回 534 —— 所谓"对照组"可能还在 2400 上跑。
#   2) 顺序测量把温度/缓存/后台负载的漂移混进了组间差异。
#   3) 单轮样本太少，而这块板子的波动很大（docs/25 记载过同一实验得到 25.5~55.0 ms）。
#
# ═══ 这次的规矩（沿用 docs/25 定下的）═══
#   · 频率靠 min_freq=max_freq **硬钉**，不靠 governor 猜；每一步都读回确认
#   · A/B **交错**进行（A B A B …），把漂移摊平到两组
#   · 多轮 + 每组多次推理，报中位数与极值，并把每轮原始值打出来
set -uo pipefail

LO=534000000
HI=2400000000
ROUNDS="${ROUNDS:-8}"
ITERS="${ITERS:-30}"
PY=""
for c in /usr/bin/python3 "$HOME/rknn-gateway/venv/bin/python3"; do
  [ -x "$c" ] && "$c" -c "import rknnlite" 2>/dev/null && { PY="$c"; break; }
done
MODEL=/opt/rknn-models/yolov5s-640-640.rknn
DMC=/sys/class/devfreq/dmc
[ -n "$PY" ] || { echo "✗ 找不到带 rknnlite 的 python" >&2; exit 1; }
[ -e "$MODEL" ] || { echo "✗ 找不到 $MODEL" >&2; exit 1; }
[ -d "$DMC" ] || { echo "✗ 没有 $DMC" >&2; exit 1; }

setfreq() { # setfreq <频率>；硬钉 min=max，并读回确认
  echo "$1" > "$DMC/min_freq" 2>/dev/null
  echo "$1" > "$DMC/max_freq" 2>/dev/null
  echo performance > "$DMC/governor" 2>/dev/null
  local got; got=$(cat "$DMC/cur_freq")
  [ "$got" = "$1" ] && return 0 || { echo "  ⚠ 想钉 $1，实际 $got" >&2; return 1; }
}

bench() { # bench <标签>；输出该次的中位 ms
  "$PY" - "$MODEL" "$ITERS" 2>/dev/null <<'PY'
import sys, time, numpy as np
from rknnlite.api import RKNNLite
model, iters = sys.argv[1], int(sys.argv[2])
r = RKNNLite(); r.load_rknn(model); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
x = np.zeros((1,640,640,3), dtype=np.uint8)
for _ in range(10): r.inference(inputs=[x])
ts=[]
for _ in range(iters):
    t=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t)*1000)
ts.sort()
print(f"{ts[len(ts)//2]:.2f} {ts[0]:.2f} {ts[-1]:.2f}")
r.release()
PY
}

echo "═══ DDR 频率影响：交错 A/B 实测 ═══"
echo "  低档 $((LO/1000000)) MHz  vs  高档 $((HI/1000000)) MHz   （$ROUNDS 轮交错，每轮 $ITERS 次推理）"
echo "  NPU 固定: $(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null) Hz"
echo

LO_S=(); HI_S=()
for r in $(seq 1 "$ROUNDS"); do
  setfreq "$LO" >/dev/null 2>&1
  L=$(bench lo); LO_S+=("${L%% *}")
  printf '  第 %d 轮  DDR %4d MHz: 中位 %6s ms   [实际频率 %s]\n' "$r" "$((LO/1000000))" "${L%% *}" "$(cat $DMC/cur_freq)"

  setfreq "$HI" >/dev/null 2>&1
  H=$(bench hi); HI_S+=("${H%% *}")
  printf '  第 %d 轮  DDR %4d MHz: 中位 %6s ms   [实际频率 %s]\n' "$r" "$((HI/1000000))" "${H%% *}" "$(cat $DMC/cur_freq)"
done

med() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{ if(NR%2) print a[(NR+1)/2]; else printf "%.2f", (a[NR/2]+a[NR/2+1])/2 }'; }
LO_M=$(med "${LO_S[@]}"); HI_M=$(med "${HI_S[@]}")
echo
echo "── 汇总 ──"
printf '  DDR %4d MHz: 中位 %s ms   各轮: %s\n' "$((LO/1000000))" "$LO_M" "$(printf '%s ' "${LO_S[@]}")"
printf '  DDR %4d MHz: 中位 %s ms   各轮: %s\n' "$((HI/1000000))" "$HI_M" "$(printf '%s ' "${HI_S[@]}")"
awk -v l="$LO_M" -v h="$HI_M" 'BEGIN{printf "  → 提升: %.2f×  （%.2f ms → %.2f ms，差 %.2f ms）\n", l/h, l, h, l-h}'
echo
echo "  恢复：钉回 $((HI/1000000)) MHz（performance）"
setfreq "$HI" >/dev/null 2>&1
echo "  当前: $(cat $DMC/cur_freq) Hz / $(cat $DMC/governor)"
