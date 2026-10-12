#!/bin/bash
# 准备内核构建树（in-tree 集成 rknpu 驱动）
#
# 路线：把驱动改动编进内核本体（CONFIG_ROCKCHIP_RKNPU=y），不做模块替换。
#       理由：板子原内核就是 =y，且在树内核源码与板子在跑的驱动逐字节一致（diff 为空），
#             所以首版构建的运行行为 = 现状，零回归风险；插桩只是显式 diff。
#
# 前提：
#   /home/dunaandone/rk-kernel/kernel-src.tar.gz        内核源码 tarball
#   /home/dunaandone/rk-kernel/build/.config.vendor     板子的原厂内核配置（未经改动）
#   /home/dunaandone/rk-kernel/toolchain/arm-gnu-toolchain
#   scripts/build/host-tools-env.sh                     免 root 的 flex/bison/m4
set -euo pipefail

ROOT=/home/dunaandone/rk-kernel
S="$ROOT/kernel"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=host-tools-env.sh
source "$HERE/host-tools-env.sh"

echo "═══ 准备内核构建树 ═══"

# ── 1) 解压 ────────────────────────────────────────────────
if [ ! -d "$S" ]; then
  echo "▶ 解压内核源码（246 MB，约 1.7 GB 展开）…"
  mkdir -p "$S"
  tar -xzf "$ROOT/kernel-src.tar.gz" -C "$S" --strip-components=1
fi
echo "  版本: $(grep -E '^(VERSION|PATCHLEVEL|SUBLEVEL) =' "$S/Makefile" | awk '{printf "%s.", $3}' | sed 's/\.$//')"
echo "  工具链: $(${CROSS_COMPILE}gcc --version | head -1)"
echo "  主机工具: flex $(flex --version) / bison $(bison --version | head -1 | awk '{print $NF}')"

# ── 2) 用板子的原厂配置 ────────────────────────────────────
echo "▶ 应用板子原厂 .config …"
cp "$ROOT/build/.config.vendor" "$S/.config"

# ── 3) 内核融合路线的关键项自检 ────────────────────────────
rk=$(grep '^CONFIG_ROCKCHIP_RKNPU=' "$S/.config" | cut -d= -f2)
[ "$rk" = "y" ] || { echo "✗ CONFIG_ROCKCHIP_RKNPU=$rk，应为 y（内核融合路线）" >&2; exit 1; }

wifi=$(grep '^CONFIG_RTW89_8852BE=' "$S/.config" | cut -d= -f2)
[ "$wifi" = "m" ] || { echo "✗ CONFIG_RTW89_8852BE=$wifi，应为 m —— 缺了它新内核起不来网卡，板子直接失联" >&2; exit 1; }

echo "  ✅ CONFIG_ROCKCHIP_RKNPU=y    驱动编进内核本体"
echo "  ✅ CONFIG_RTW89_8852BE=m      RTL8852BE WiFi，必须随内核装到新模块目录"

# ── 4) 唯一一处有意偏离原厂配置：关掉 DEBUG_INFO ───────────
# 原因：本机 7 GB 内存 + 2 GB swap，arm64 vmlinux 带 DWARF 链接要吃 4–8 GB，
#       并发编译也显著变慢；而我们要反复迭代插桩构建。
# 影响：仅少了 vmlinux 的调试信息，运行行为零差异；CONFIG_KALLSYMS=y 仍保留，
#       oops/panic 回溯照样有符号名。
# 回退：想还原成原厂配置，删掉下面 set_kv 两行即可（.config.vendor 从未被改动）。
set_kv() { # set_kv SYMBOL VALUE
  local sym="$1" val="$2"
  sed -i "/^${sym}=/d" "$S/.config"
  sed -i "/^# ${sym} is not set$/d" "$S/.config"
  if [ "$val" = "n" ]; then
    printf '# %s is not set\n' "$sym" >> "$S/.config"
  else
    printf '%s=%s\n' "$sym" "$val" >> "$S/.config"
  fi
}
set_kv CONFIG_DEBUG_INFO n
set_kv CONFIG_DEBUG_INFO_NONE y

# ── 5) 生成完整配置 ────────────────────────────────────────
echo "▶ make olddefconfig …"
make -C "$S" olddefconfig >/dev/null

for s in ROCKCHIP_RKNPU ROCKCHIP_RKNPU_DEBUG_FS ROCKCHIP_RKNPU_DRM_GEM RTW89_8852BE DEBUG_INFO; do
  printf '  %-28s %s\n' "$s" "$(grep -E "^(CONFIG_${s}=|# CONFIG_${s} is not set)" "$S/.config" || echo '<未定义>')"
done

echo
echo "✅ 构建树就绪：$S"
ND=$(diff <(grep -E '^CONFIG_' "$ROOT/build/.config.vendor") <(grep -E '^CONFIG_' "$S/.config") | grep -c '^[<>]' || true)
echo "   与厂商配置的差异：$ND 行"
diff <(grep -E '^CONFIG_' "$ROOT/build/.config.vendor") <(grep -E '^CONFIG_' "$S/.config") | grep '^[<>]' | sed 's/^/     /' || true
