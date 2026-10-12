#!/bin/bash
# 等 tarball → 准备构建树 → 构建完整内核（in-tree 集成驱动）
set -u
ROOT=/home/dunaandone/rk-kernel
TB="$ROOT/kernel-src.tar.gz"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "▶ 等待下载…"
if pgrep -f 'curl.*codeload' >/dev/null; then
  last=0; stall=0
  for i in $(seq 1 240); do
    pgrep -f 'curl.*codeload' >/dev/null || break
    sz=$(stat -c %s "$TB" 2>/dev/null || echo 0)
    [ "$sz" -eq "$last" ] && stall=$((stall+1)) || stall=0
    last=$sz
    [ "$stall" -ge 10 ] && { echo "  ⚠️ 300 秒无增长，判定卡住"; break; }
    sleep 30
  done
fi
printf "  大小 %.1f MB\n" "$(echo "$(stat -c %s "$TB" 2>/dev/null || echo 0)/1048576" | bc -l)"

echo "▶ 校验完整性…"
tar -tzf "$TB" >/dev/null 2>&1 || { echo "  ❌ tarball 不完整"; exit 1; }
echo "  ✅ 完整"

echo "▶ 准备构建树…"
bash "$HERE/prepare-kernel.sh" 2>&1 | tail -15 || exit 1

echo "▶ 构建内核（in-tree 驱动）…"
bash "$HERE/build-kernel.sh" 2>&1 | tail -30
