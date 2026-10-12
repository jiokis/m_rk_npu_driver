#!/bin/bash
# 在板子上安装新内核（只新增带版本号的文件，绝不碰正在生效的启动项）
#
# 设计原则：
#   安装与切换分离。"安装"只往 /boot 和 /lib/modules 里放新文件，
#   /boot/Image、/boot/dtb、/boot/uInitrd 这三个**决定下次启动谁**的符号链接一律不动。
#   要真正切换，装完之后显式执行 `kernel-select <release> && reboot`。
#   这样"装一半失败"绝不会把板子变成起不来的状态。
#
# 用法（在板子上）：
#   sudo scripts/board/install-kernel.sh 6.1.99-rockchip-rk3588-priv1
#   # 上一步产物默认在 /root/kernel-deploy/
set -euo pipefail

R="${1:?用法: install-kernel.sh <release> [deploy目录]}"
D="${2:-/root/kernel-deploy}"

[ "$(id -u)" = "0" ] || { echo "请用 root 运行（sudo）" >&2; exit 1; }
[ "$(uname -m)" = "aarch64" ] || { echo "✗ 这个脚本要在板子上跑（当前 $(uname -m)）" >&2; exit 1; }

echo "═══ 在板子上安装内核 $R ═══"
echo "  部署目录: $D"

# ── 0) 产物自检 ────────────────────────────────────────────
fail=0
chk() { [ -e "$1" ] && echo "  ✅ $2" || { echo "  ✗ 缺 $2 ($1)"; fail=1; }; }
chk "$D/Image"                       "内核镜像 Image"
chk "$D/dtb"                         "设备树目录"
chk "$D/modstage/lib/modules/$R"     "模块树"
[ "$fail" = 0 ] || { echo "✗ 产物不全，放弃（板子状态未改动）" >&2; exit 1; }
echo "  模块: $(find "$D/modstage/lib/modules/$R" -name '*.ko' | wc -l) 个"

echo
echo "── 当前启动项（安装前后都不动它）──"
for l in Image dtb uInitrd; do printf '  %-8s → %s\n' "$l" "$(readlink -f "/boot/$l" 2>/dev/null || echo '<缺失>')"; done

# ── 1) 空间检查 ────────────────────────────────────────────
echo
NEED_BOOT=$(du -sm "$D/Image" "$D/dtb" | awk '{s+=$1} END{print s+10}')
NEED_ROOT=$(du -sm "$D/modstage" | awk '{print $1+20}')
AVAIL_BOOT=$(df -Pm /boot | awk 'NR==2{print $4}')
AVAIL_ROOT=$(df -Pm /    | awk 'NR==2{print $4}')
printf '  需要: /boot ≈%s MB（可用 %s MB）  / ≈%s MB（可用 %s MB）\n' "$NEED_BOOT" "$AVAIL_BOOT" "$NEED_ROOT" "$AVAIL_ROOT"
[ "$AVAIL_BOOT" -ge "$NEED_BOOT" ] || { echo "✗ /boot 空间不足" >&2; exit 1; }
[ "$AVAIL_ROOT" -ge "$NEED_ROOT" ] || { echo "✗ / 空间不足" >&2; exit 1; }

# ── 2) 装模块 ──────────────────────────────────────────────
echo
echo "▶ 安装模块 → /lib/modules/$R/"
if [ -e "/lib/modules/$R" ]; then
  echo "  ⚠ /lib/modules/$R 已存在，先移到 .bak（不删除，保留回退）"
  rm -rf "/lib/modules/$R.bak"
  mv "/lib/modules/$R" "/lib/modules/$R.bak"
fi
cp -a "$D/modstage/lib/modules/$R" /lib/modules/
echo "  depmod -a $R"
depmod -a "$R"
echo "  已装: $(find "/lib/modules/$R" -name '*.ko' | wc -l) 个模块"
find "/lib/modules/$R" -name '*.ko' -printf '    %f\n' 2>/dev/null || true

# ── 3) 装内核与设备树（带版本号，不覆盖）──────────────────
echo
echo "▶ 安装内核与设备树"
install -m 0644 "$D/Image" "/boot/vmlinuz-$R"
[ -e "$D/System.map-$R" ] && install -m 0644 "$D/System.map-$R" "/boot/System.map-$R" || true
[ -e "$D/config-$R" ]     && install -m 0644 "$D/config-$R"     "/boot/config-$R"     || true

rm -rf "/boot/dtb-$R"
mkdir -p "/boot/dtb-$R/rockchip"
# boot.cmd 里加载的是 dtb/${fdtfile}，fdtfile 形如 rockchip/rk3588-orangepi-5-plus.dtb，
# 所以设备树要落在 /boot/dtb-<release>/rockchip/ 下
if [ -d "$D/dtb/rockchip" ]; then
  cp -a "$D/dtb/." "/boot/dtb-$R/"
else
  cp -a "$D/dtb"/*.dtb "/boot/dtb-$R/rockchip/"
fi
echo "  内核: $(du -h "/boot/vmlinuz-$R" | cut -f1)"
echo "  设备树: $(find "/boot/dtb-$R" -name '*.dtb' | wc -l) 个"
[ -e "/boot/dtb-$R/rockchip/rk3588-orangepi-5-plus.dtb" ] \
  && echo "  ✅ 关键设备树 rk3588-orangepi-5-plus.dtb 在位" \
  || echo "  ⚠ 没找到 rk3588-orangepi-5-plus.dtb —— 起来可能不对，先别切"

# ── 4) 生成 initramfs ──────────────────────────────────────
echo
echo "▶ 生成 initramfs"
if command -v update-initramfs >/dev/null; then
  update-initramfs -c -k "$R" 2>&1 | tail -5
  [ -e "/boot/initrd.img-$R" ] && echo "  ✅ /boot/initrd.img-$R ($(du -h /boot/initrd.img-$R | cut -f1))" \
                              || echo "  ✗ update-initramfs 没产出 initrd"
else
  echo "  ⚠ 没有 update-initramfs"
fi

# uInitrd = mkimage 包过的 initrd，U-Boot 直接 load 的那种格式
echo "▶ 打包 uInitrd（U-Boot 格式）"
MK=; command -v mkimage >/dev/null && MK=mkimage
if [ -z "$MK" ] && [ -x /usr/bin/mkimage ]; then MK=/usr/bin/mkimage; fi
if [ -z "$MK" ] && [ -e "$D/u-boot-tools_arm64.deb" ]; then
  echo "  本机没有 mkimage，用随包附带的 arm64 deb 解出来"
  mkdir -p /tmp/ubt && dpkg-deb -x "$D/u-boot-tools_arm64.deb" /tmp/ubt && MK=/tmp/ubt/usr/bin/mkimage
fi
if [ -z "$MK" ] && command -v apt-get >/dev/null; then
  echo "  尝试 apt-get install -y u-boot-tools"
  apt-get install -y u-boot-tools >/dev/null 2>&1 && command -v mkimage >/dev/null && MK=mkimage
fi

if [ -n "$MK" ] && [ -e "/boot/initrd.img-$R" ]; then
  "$MK" -A arm64 -O linux -T ramdisk -C none -a 0 -e 0 \
        -n "initramfs-$R" -d "/boot/initrd.img-$R" "/boot/uInitrd-$R" >/dev/null
  echo "  ✅ /boot/uInitrd-$R ($(du -h "/boot/uInitrd-$R" | cut -f1))  [$MK]"
else
  # 兜底：直接把 raw initrd 当 uInitrd 用（部分 U-Boot 配置能识别 gzip cpio）
  if [ -e "/boot/initrd.img-$R" ]; then
    cp "/boot/initrd.img-$R" "/boot/uInitrd-$R"
    echo "  ⚠ 没有 mkimage，已用 raw initrd 顶替。若新内核起不来，这是首要嫌疑。"
  else
    echo "  ✗ 既没有 initrd 也没有 mkimage" >&2
  fi
fi

# ── 5) 就绪自检（不切换）───────────────────────────────────
echo
echo "═══ 安装完成，尚未切换 ═══"
allok=1
for f in "vmlinuz-$R" "dtb-$R/rockchip/rk3588-orangepi-5-plus.dtb" "uInitrd-$R" "System.map-$R" "config-$R"; do
  if [ -e "/boot/$f" ]; then printf '  ✅ /boot/%s\n' "$f"; else printf '  ✗ /boot/%s\n' "$f"; allok=0; fi
done
[ -d "/lib/modules/$R" ] && echo "  ✅ /lib/modules/$R" || { echo "  ✗ /lib/modules/$R"; allok=0; }

echo
if [ "$allok" = 1 ]; then
  cat <<EOF
✅ 一切就绪。切换并重启：

    sudo $(dirname "$0")/kernel-select $R     # 切换启动指向（旧内核文件保持不动）
    sudo reboot

  起不来时的回退：
    · 如果还能进系统（比如切错了）：sudo $(dirname "$0")/kernel-select orig && sudo reboot
    · 如果完全起不来：需要串口进 U-Boot，或把 boot 分区/sd 卡重新刷回原厂镜像
EOF
else
  echo "✗ 有产物缺失，**先不要切换**。补全后重跑本脚本。"
  exit 1
fi
