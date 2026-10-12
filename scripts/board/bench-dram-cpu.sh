#!/bin/bash
# 纯 CPU 侧 DRAM 带宽基准 —— 给 NPU 的"带宽墙"提供一个独立参照系
#
# ═══ 为什么必须有这个 ═══
# 之前一直用"DDR 钉频带来 1.59×"来论证"DRAM 带宽是瓶颈"。但那个论证有个漏洞：
# **它只证明"降频会变慢"，不证明"NPU 把带宽用满了"。**
# 这两件事完全不同：
#   · 带宽饱和 → 加带宽有救
#   · 延迟/并行度受限 → 加带宽没用，得提高访存并行度或减少往返
#
# 所以需要一个独立于 NPU 的"这台机器到底能跑多少带宽"的基准。
# 纯 CPU memcpy 就够了，零风险、不需要碰 NPU、结果可比。
#
# ═══ 用法 ═══
#   sudo scripts/board/bench-dram-cpu.sh              # 在两个 DMC 频率点上各测一次
#   sudo scripts/board/bench-dram-cpu.sh --freq 2400000000   # 只在指定频率测
#   sudo scripts/board/bench-dram-cpu.sh --restore    # 测完恢复 performance@max
#
# ⚠ 脚本会临时改 DMC 频率，跑完会恢复（除非 --freq 指定了固定值）
set -uo pipefail

[ "$(id -u)" = "0" ] || { echo "请用 root 运行（sudo）" >&2; exit 1; }
DMC=/sys/class/devfreq/dmc
[ -d "$DMC" ] || { echo "✗ 没有 $DMC" >&2; exit 1; }

LO=534000000
HI=2400000000
SIZE_MB="${SIZE_MB:-64}"
REPS="${REPS:-4}"
FIXED=""
RESTORE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --freq) FIXED="$2"; RESTORE=0; shift 2 ;;
    --restore) FIXED=""; RESTORE=1; shift ;;
    *) echo "未知参数 $1" >&2; exit 1 ;;
  esac
done

setfreq() {
  echo "$1" > "$DMC/min_freq" 2>/dev/null
  echo "$1" > "$DMC/max_freq" 2>/dev/null
  echo performance > "$DMC/governor" 2>/dev/null
  local got; got=$(cat "$DMC/cur_freq")
  [ "$got" = "$1" ] || echo "  ⚠ 想钉 $1，实际 $got" >&2
}

bench() {
  python3 - "$SIZE_MB" "$REPS" <<'PY'
import array, sys, time
size_mb = int(sys.argv[1]); reps = int(sys.argv[2])
N = size_mb * 1024 * 1024
a = array.array("b", bytes(N))
b = array.array("b", bytes(N))
b[:] = a                      # 预热，把页真正分配好
best = 1e9
for _ in range(3):
    t = time.perf_counter()
    for _ in range(reps):
        b[:] = a
    dt = time.perf_counter() - t
    best = min(best, dt)
# memcpy：读 N 写 N，共 2N 字节
print(f"{reps*2*N/best/1e9:.2f} {best*1000:.0f}")
PY
}

echo "═══ CPU 侧 DRAM 带宽基准（${SIZE_MB}MB × ${REPS} 次 memcpy，取 3 轮最快）═══"
echo "  说明：读 N + 写 N = 2N 字节。这是这台机器**实际可达**的 DRAM 吞吐。"
echo

printrow() { printf '  DDR %5s MHz  %8s GB/s   (%.0f ms)\n' "$1" "$2" "$3"; }

if [ -n "$FIXED" ]; then
  setfreq "$FIXED"
  r=$(bench)
  printrow "$((FIXED/1000000))" "${r%% *}" "${r##* }"
else
  declare -A R
  for f in $LO $HI; do
    setfreq "$f"
    r=$(bench)
    R[$f]="${r%% *}"
    printrow "$((f/1000000))" "${r%% *}" "${r##* }"
  done
  awk -v a="${R[$LO]}" -v b="${R[$HI]}" 'BEGIN{
    printf "\n  频率比 %.2f×（%d→%d MHz），带宽比 %.2f×\n", 2400/534, 534, 2400, b/a }'
  echo
  echo "  ★ 把这个带宽比和 NPU 的 1.59× 对照："
  echo "      若 NPU 是纯带宽受限，它应该跟着带宽比一起变；"
  echo "      实测 NPU 只拿到 1.59× ⇒ NPU 没有把带宽用满。"
fi

if [ "$RESTORE" = 1 ]; then
  setfreq "$HI"
  echo
  echo "  已恢复: $(cat $DMC/cur_freq) Hz / $(cat $DMC/governor)"
fi
