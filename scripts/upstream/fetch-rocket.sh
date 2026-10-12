#!/bin/bash
# 拉取上游 rocket 开源实现（内核侧 + Mesa 用户态）用于与我们的逆向结论对撞
#
# ═══ 为什么需要这个 ═══
#
# 我们的逆向是**独立**完成的（615 条真实 regcmd dump + 驱动源码 + NVDLA 对照）。
# 上游有一份开源实现，是绝佳的**第三方交叉验证**，而且它还是「指令流水构建」的
# 开源参考实现（Mesa 的 rkt_regcmd.c 就是把算子编译成 regcmd 序列的完整示例）。
#
# ★ 这条路**完全不碰硬件寄存器**，所以零"把内核打挂"的风险（对比 docs/29 的事故）。
#
# ═══ 网络注意事项（本机实测）═══
#   · raw.githubusercontent.com 在本机被解析到非公网 IP，**取不到**
#   · 内核侧改走 kernel.org 的 gitweb plain 接口（可用）
#   · Mesa 侧走 freedesktop GitLab 的 raw 接口（可用）
#
# 用法：scripts/upstream/fetch-rocket.sh [目标目录]     默认 ~/rocket-upstream
set -uo pipefail

DEST="${1:-$HOME/rocket-upstream}"
K="$DEST/kernel"; M="$DEST/mesa"
mkdir -p "$K" "$M"

KURL="https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/accel/rocket"
MURL="https://gitlab.freedesktop.org/mesa/mesa/-/raw/main/src/gallium/drivers/rocket"

K_FILES="Kconfig Makefile rocket_core.c rocket_core.h rocket_device.c rocket_device.h \
         rocket_drv.c rocket_drv.h rocket_gem.c rocket_gem.h rocket_job.c rocket_job.h \
         rocket_registers.h"

M_FILES="registers.xml rkt_regcmd.c rkt_regcmd.h rkt_task.c rkt_task.h rkt_ml.c rkt_ml.h \
         rkt_device.c rkt_device.h rkt_coefs.c rkt_coefs.h intercept.c decode.py \
         extract_registers.py gen_header.py gen_parser.py"

doc_fetch() { # doc_fetch <url> <outfile> <描述>
  local url="$1" out="$2" desc="$3"
  if [ -s "$out" ]; then echo "  已有 $desc"; return 0; fi
  local code
  code=$(timeout 60 curl -s -o "$out" -w '%{http_code}' "$url" 2>/dev/null)
  if [ "$code" = "200" ] && [ -s "$out" ]; then
    printf '  ✅ %-28s %8d B\n' "$desc" "$(stat -c%s "$out")"
  else
    printf '  ✗ %-28s http=%s\n' "$desc" "$code"; rm -f "$out"; return 1
  fi
}

echo "═══ 拉取上游 rocket 实现 → $DEST ═══"
echo "▶ 内核侧（kernel.org gitweb plain）"
fail=0
for f in $K_FILES; do doc_fetch "$KURL/$f" "$K/$f" "kernel/$f" || fail=1; done
echo "▶ 内核文档"
doc_fetch "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/accel/rocket/index.rst" \
          "$K/index.rst" "Documentation index.rst" || true

echo "▶ Mesa 侧（freedesktop GitLab raw）"
for f in $M_FILES; do doc_fetch "$MURL/$f" "$M/$f" "mesa/$f" || fail=1; done

echo
echo "═══ 这批文件里最关键的几处 ═══"
cat <<'EOF'
  kernel/rocket_registers.h     1566 个 #define —— 完整寄存器与位域（自动生成）
  mesa/registers.xml            寄存器的源头（含 target 枚举、位域、注释）
  mesa/rkt_regcmd.c             ★ 命令流构建器：emit_raw 的编码方式、
                                  段终止序列（含引用 TRM 原文的注释）
  mesa/rkt_task.c               ★ 分块与 bank 分配（calc_input_banks /
                                  calc_weights_banks / available_input_banks）
  mesa/rkt_ml.c                 算子如何被编译成 operation/task
  mesa/intercept.c              上游自己的抓取工具（对比我们的 LD_PRELOAD 探针）
EOF
echo
[ "$fail" = 0 ] && echo "✅ 拉取完成（用 docs/36 的对撞方法做交叉验证）" \
                || echo "⚠ 有文件没拉到（网络不稳，可重跑本脚本续传）"
