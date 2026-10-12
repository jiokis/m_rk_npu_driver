#!/bin/bash
# 构建内核（in-tree 集成 rknpu 驱动）—— 性能优先版
#
# ═══ 为什么这么设计（本机是虚拟机，硬扛全量构建会把机器拖死）═══
#
# 实测环境：AMD Ryzen 9 8945H 的虚拟机，8 物理核 / 16 vCPU、7.58 GB 内存、
#           2 GB swap、**磁盘 ROTA=1（机械盘虚拟盘）**、没有 cpufreq 节点。
# 之前的做法 `make -j14 Image dtbs modules` 直接把整机冻住：并发 gcc 吃满内存后
# 进 swap，叠加机械盘的写回风暴，交互式操作全被饿死。
#
# 关键观察：查板子原厂配置发现 —— 所有**启动关键驱动本来就全是 =y**：
#   MMC_DW_ROCKCHIP=y  MMC_SDHCI_OF_DWCMSHC=y  MMC_BLOCK=y  BLK_DEV_NVME=y
#   EXT4_FS=y  PCIE_ROCKCHIP_HOST=y  PCIE_DW_ROCKCHIP=y  PHY_ROCKCHIP_*=y
#   连无线协议栈 CFG80211=y  MAC80211=y  RFKILL=y 也全是内置
# 整个配置里 792 个 =m，而**跟这块板子启动/联网真正相关的只有 4 个**：
#   rtw89_core / rtw89_pci / rtw89_8852b / rtw89_8852be
#
# 所以默认只编 `Image dtbs`（= 全部内置代码），再单独编 rtw89 那一个目录，
# 而不是让 make 去编 792 个用不上的模块 —— 省掉将近一半工作量，外加省掉
# modules_install / depmod / 打 200 MB 模块包 / 往板子传 / 装模块目录 一整串动作。
#
# 三重性能护栏（保证编译期间机器仍可正常使用）：
#   1) nice -n 15    —— 编译让出 CPU 给前台
#   2) ionice -c 3   —— 编译走 idle I/O 类，机械盘的带宽优先给前台
#   3) -j 默认 8（可调）—— 把并发 gcc 的峰值内存压在 2 GB 出头，不触发 swap 抖动
#
# 用法：
#   scripts/build/build-kernel.sh                       # 默认：Image + dtbs + rtw89 模块
#   JOBS=6 scripts/build/build-kernel.sh                # 更保守
#   BUILD_ALL_MODULES=1 scripts/build/build-kernel.sh   # 老做法：编全部 792 个模块（很慢）
#   TARGETS="Image dtbs" MODDIRS="" scripts/build/build-kernel.sh   # 纯内核，连 rtw89 都不编
set -euo pipefail

ROOT=/home/dunaandone/rk-kernel
S="$ROOT/kernel"
OUT="${1:-$ROOT/out}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=host-tools-env.sh
source "$HERE/host-tools-env.sh"

LV="${LOCALVERSION:--rockchip-rk3588-priv1}"
JOBS="${JOBS:-8}"
NICE="${NICE:-15}"
TARGETS="${TARGETS:-Image dtbs}"
# 实测板子在跑的 10 个模块，反查它们落在下面 7 个目录里（依赖闭包是平的，
# 没有任何模块依赖自己目录之外的模块 —— ip_tables 甚至无依赖，说明 x_tables 是内置的）：
#   pwm_fan       → drivers/hwmon                          （风扇，用户关心）
#   rtw89_8852be  → drivers/net/wireless/realtek/rtw89     （WiFi，唯一启动关键项）
#     rtw89_8852b / rtw89_pci / rtw89_core （同目录，依赖互锁）
#   r8125         → drivers/net/ethernet/realtek/r8125      （2.5G 有线网卡）
#   pgdrv         → drivers/net/ethernet/realtek/realtek_pgtool
#   fuse          → fs/fuse
#   ip_tables     → net/ipv4/netfilter
#   sch_fq_codel  → net/sched
# 另外补一个当前没在用、但迟早要的：
#   overlayfs     → fs/overlayfs    （Docker/容器必需；板子的 apt 配置里有 Docker 痕迹）
# 其余 =m 都是当前用不到的功能。工具链已就绪，缺什么增量编一个目录只要 1 分钟：
#   make -j8 M=<目录> modules
MODDIRS="${MODDIRS:-drivers/net/wireless/realtek/rtw89 drivers/net/ethernet/realtek/r8125 drivers/net/ethernet/realtek/realtek_pgtool drivers/hwmon fs/fuse fs/overlayfs net/sched net/ipv4/netfilter net/netfilter net/netfilter/ipset net/ipv6/netfilter}"
BUILD_ALL_MODULES="${BUILD_ALL_MODULES:-0}"

mkdir -p "$OUT"
RELEASE="$(make -s -C "$S" LOCALVERSION="$LV" kernelrelease)"

cat <<EOF
═══ 构建内核（性能优先）═══
  release:    $RELEASE
  源码:       $S
  编译器:     $(${CROSS_COMPILE}gcc --version | head -1)
  主机工具:   flex $(flex --version) / bison $(bison --version | head -1 | awk '{print $NF}') / m4 $(m4 --version | head -1 | awk '{print $NF}')
  构建目标:   $TARGETS
  模块目标:   $([ "$BUILD_ALL_MODULES" = 1 ] && echo '全部 =m（慢）' || echo "${MODDIRS:-无}")
  并行度:     -j$JOBS   nice=$NICE   ionice=idle
  内存:       $(free -m | awk '/Mem:/{print $2" MB 总 / "$7" MB 可用"}')
  磁盘:       $(df -h "$S" | awk 'NR==2{print $1" "$2" 已用 "$5}')
  配置差异:   $(diff <(grep -E '^CONFIG_' "$ROOT/build/.config.vendor") <(grep -E '^CONFIG_' "$S/.config") | grep -c '^[<>]') 行（相对板子原厂配置）
EOF
echo

cd "$S"
START=$(date +%s)

# ── 1) 内核镜像 + 设备树（只含 =y 的内置代码）──────────────
echo "▶ [1/3] make $TARGETS"
# shellcheck disable=SC2086
nice -n "$NICE" ionice -c 3 make -j"$JOBS" LOCALVERSION="$LV" $TARGETS
T1=$(date +%s); echo "  ⏱ $(( (T1-START)/60 ))m$(( (T1-START)%60 ))s"

# ── 2) 只编板子真正要加载的那几个模块 ──────────────────────
KOS=()
if [ "$BUILD_ALL_MODULES" = 1 ]; then
  echo "▶ [2/3] make modules （全部 =m，慢）"
  nice -n "$NICE" ionice -c 3 make -j"$JOBS" LOCALVERSION="$LV" modules
  mapfile -t KOS < <(find "$S" -name '*.ko')
elif [ -n "$MODDIRS" ]; then
  echo "▶ [2/3] 按目录编模块：$MODDIRS"
  for d in $MODDIRS; do
    echo "  → $d"
    if nice -n "$NICE" ionice -c 3 make -j"$JOBS" LOCALVERSION="$LV" M="$d" modules; then
      while IFS= read -r k; do KOS+=("$k"); done < <(find "$S/$d" -name '*.ko')
    else
      echo "  ✗ $d 编模块失败（先记下，Image 已产出，可稍后单独处理）" >&2
    fi
  done
else
  echo "▶ [2/3] 跳过模块（MODDIRS 为空）"
fi
T2=$(date +%s); echo "  ⏱ $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s，模块 ${#KOS[@]} 个"
for k in "${KOS[@]:-}"; do [ -n "$k" ] && echo "    ${k##*/}"; done

# ── 3) 收集产物 ────────────────────────────────────────────
echo "▶ [3/3] 收集产物到 $OUT"
cp arch/arm64/boot/Image "$OUT/Image"
mkdir -p "$OUT/dtb"
cp arch/arm64/boot/dts/rockchip/rk3588-orangepi-5-plus.dtb "$OUT/dtb/"
cp arch/arm64/boot/dts/rockchip/*.dtb "$OUT/dtb/" 2>/dev/null || true
mkdir -p "$OUT/modules"
for k in "${KOS[@]:-}"; do [ -n "$k" ] && cp "$k" "$OUT/modules/"; done

# 模块按内核源码里的目录结构摆好，板子侧直接 rsync + depmod，不用猜路径
STAGE="$OUT/modstage"
rm -rf "$STAGE"; mkdir -p "$STAGE/lib/modules/$RELEASE"
for k in "${KOS[@]:-}"; do
  [ -n "$k" ] || continue
  rel="${k#"$S"/}"; rel="${rel%/*}"          # 例如 drivers/net/wireless/realtek/rtw89
  dest="$STAGE/lib/modules/$RELEASE/kernel/$rel"
  mkdir -p "$dest"; cp "$k" "$dest/"
done

END=$(date +%s)
echo
echo "── 产物 ──"
ls -la "$OUT/Image" | awk '{printf "  Image:    %.1f MB\n", $5/1048576}'
echo "  dtb:      $(ls "$OUT/dtb"/*.dtb 2>/dev/null | wc -l) 个（含 rk3588-orangepi-5-plus.dtb）"
echo "  模块:     ${#KOS[@]} 个 → $STAGE/lib/modules/$RELEASE/"
echo "  总耗时:   $(( (END-START)/60 ))m$(( (END-START)%60 ))s"

MAN="$OUT/kernel-$RELEASE.manifest.txt"
{
  echo "release:      $RELEASE"
  echo "built:        $(date -Is)"
  echo "host:         $(uname -sr) / VM / $(nproc) vCPU / $(free -g | awk '/Mem:/{print $2}') GB"
  echo "toolchain:    $(${CROSS_COMPILE}gcc --version | head -1)"
  echo "host tools:   flex $(flex --version) | bison $(bison --version | head -1 | awk '{print $NF}') | m4 $(m4 --version | head -1 | awk '{print $NF}')"
  echo "targets:      $TARGETS"
  echo "module dirs:  $([ "$BUILD_ALL_MODULES" = 1 ] && echo ALL || echo "${MODDIRS:-none}")"
  echo "jobs/nice:    -j$JOBS nice=$NICE ionice=idle"
  echo "build time:   $(( (END-START)/60 ))m$(( (END-START)%60 ))s"
  echo
  echo "── 关键配置 ──"
  for s in ROCKCHIP_RKNPU ROCKCHIP_RKNPU_DEBUG_FS ROCKCHIP_RKNPU_DRM_GEM RTW89_8852BE CFG80211 MAC80211 DEBUG_INFO KALLSYMS; do
    printf '  %-28s %s\n' "$s" "$(grep -E "^(CONFIG_${s}=|# CONFIG_${s} is not set)" .config || echo '<未定义>')"
  done
  echo
  echo "── 相对板子原厂配置的全部差异 ──"
  diff <(grep -E '^CONFIG_' "$ROOT/build/.config.vendor") <(grep -E '^CONFIG_' .config) | grep '^[<>]' | sed 's/^/  /' || true
  echo
  echo "── 产物校验 ──"
  sha256sum "$OUT/Image" | sed 's/^/  Image  /'
  sha256sum "$OUT/dtb/rk3588-orangepi-5-plus.dtb" | sed 's/^/  dtb    /'
  for k in "${KOS[@]:-}"; do [ -n "$k" ] && sha256sum "$k" | sed 's/^/  mod    /'; done
} > "$MAN"
echo "  manifest: $MAN"
echo
echo "✅ 完成：$RELEASE"
