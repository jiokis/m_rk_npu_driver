#!/usr/bin/env bash
# get-gdb.sh — 无需 root 获取可用的 gdb（逆向 librknnc.so 用）
#
# 背景：docs/118 判定「gdb 路径不可行」的原因之一是「本机没装 gdb」。
# 本脚本证明它**可移除**：apt-get download + dpkg-deb -x 不需要 root。
#
# 用法：bash tools/get-gdb.sh [目标目录，默认 /tmp/gdbbin]
# 之后：export LD_LIBRARY_PATH=<dir>/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
#       <dir>/root/usr/bin/gdb --version
set -euo pipefail
DEST="${1:-/tmp/gdbbin}"
mkdir -p "$DEST" && cd "$DEST"
# gdb 本体 + 它缺的 4 个运行库（实测在 Ubuntu 24.04 上缺这些）
PKGS="gdb libbabeltrace1 libdebuginfod1t64 libipt2 libsource-highlight4t64"
for p in $PKGS; do
  apt-get download "$p" >/dev/null 2>&1 && echo "  ✓ $p" || echo "  ✗ $p（可能名字随发行版不同）"
done
for d in *.deb; do dpkg-deb -x "$d" root/ 2>/dev/null; done
export LD_LIBRARY_PATH="$DEST/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
echo "--- 验证 ---"
"$DEST/root/usr/bin/gdb" --version 2>&1 | head -1
echo "用：LD_LIBRARY_PATH=$DEST/root/usr/lib/x86_64-linux-gnu $DEST/root/usr/bin/gdb <目标>"
