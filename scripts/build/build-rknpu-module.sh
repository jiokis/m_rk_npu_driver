#!/bin/bash
# 从 GPL 源码构建 rknpu 驱动模块（交叉编译，aarch64）
#
# 前提：
#   KERNEL_SRC  — 内核源码树（orange-pi-6.1-rk35xx，已 make modules_prepare）
#   TOOLCHAIN   — aarch64-none-linux-gnu- 交叉工具链 bin 目录
#   CONFIG      — 板子的 /boot/config-*
#
# 用法： ./build-rknpu-module.sh [outdir]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KERNEL_SRC="${KERNEL_SRC:-/home/dunaandone/rk-kernel/kernel}"
TOOLCHAIN="${TOOLCHAIN:-/home/dunaandone/rk-kernel/toolchain/arm-gnu-toolchain/bin}"
OUT="${1:-$REPO/artifacts/rknpu-module}"

export PATH="$TOOLCHAIN:$PATH"
export ARCH=arm64
export CROSS_COMPILE=aarch64-none-linux-gnu-

echo "═══ 构建 rknpu 模块 ═══"
echo "  内核源码: $KERNEL_SRC"
echo "  工具链  : $(command -v ${CROSS_COMPILE}gcc || echo '❌ 找不到')"
echo "  输出    : $OUT"
echo

# 1) 准备外部构建目录
SRCDIR="$OUT/src"
mkdir -p "$SRCDIR/include"
cp "$REPO/src/rknpu-6.1"/*.c "$SRCDIR/" 2>/dev/null || \
  cp "/home/dunaandone/orange pi/src/rknpu-6.1"/*.c "$SRCDIR/"
cp "/home/dunaandone/orange pi/src/rknpu-6.1/include"/*.h "$SRCDIR/include/"

cat > "$SRCDIR/Makefile" <<'MK'
obj-m += rknpu.o
rknpu-y := rknpu_drv.o rknpu_reset.o rknpu_job.o rknpu_debugger.o rknpu_iommu.o
rknpu-$(CONFIG_PM_DEVFREQ) += rknpu_devfreq.o
rknpu-$(CONFIG_ROCKCHIP_RKNPU_SRAM) += rknpu_mm.o
rknpu-$(CONFIG_ROCKCHIP_RKNPU_FENCE) += rknpu_fence.o
rknpu-$(CONFIG_ROCKCHIP_RKNPU_DRM_GEM) += rknpu_gem.o
rknpu-$(CONFIG_ROCKCHIP_RKNPU_DMA_HEAP) += rknpu_mem.o
ccflags-y += -I$(src)/include
ccflags-y += -Werror
MK

# 2) 按板子的配置构建（RKNPU 作为模块）
echo "▶ make modules …"
make -C "$KERNEL_SRC" M="$SRCDIR" \
     CONFIG_ROCKCHIP_RKNPU=m \
     CONFIG_ROCKCHIP_RKNPU_DEBUG_FS=y \
     CONFIG_ROCKCHIP_RKNPU_DRM_GEM=y \
     CONFIG_PM_DEVFREQ=y \
     KBUILD_MODPOST_WARN=1 \
     modules 2>&1 | tail -25

echo
echo "▶ 产物："
find "$SRCDIR" -name "*.ko" -exec ls -la {} \; | sed 's/^/  /'
find "$SRCDIR" -name "*.ko" -exec sh -c '
  echo "  ── 模块信息 ──"
  '"$TOOLCHAIN"'/aarch64-none-linux-gnu-objdump -h "$1" 2>/dev/null | head -3
' _ {} \;
echo
echo "✅ 完成。产物在 $SRCDIR/rknpu.ko"
