#!/bin/bash
# 笔记本侧：把新内核产物送到板子并安装
#
# 用法：
#   scripts/build/deploy-to-board.sh                      # 只送 + 装（不切换）
#   scripts/build/deploy-to-board.sh --switch             # 装完顺带切换并重启
#   scripts/build/deploy-to-board.sh --release <release>  # 指定 release
#
# 板子上的 sudo 密码是 orangepi，用 SUDO_ASKPASS 免交互。
set -euo pipefail

ROOT=/home/dunaandone/rk-kernel
S="$ROOT/kernel"
OUT="${OUT:-$ROOT/out}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

BOARD_HOST="${BOARD_HOST:-192.168.31.128}"
BOARD_USER="${BOARD_USER:-orangepi}"
BOARD_KEY="${BOARD_KEY:-$HOME/.ssh/keys/opi_board}"
# 暂存目录放在 orangepi 用户家目录下，不放 /root ——
# scp 是以 orangepi 身份跑的，写 /root 会 Permission denied（踩过）。
# 安装脚本用 sudo 执行，读家目录没问题。
BOARD_DIR="${BOARD_DIR:-/home/$BOARD_USER/kernel-deploy}"

LV="${LOCALVERSION:--rockchip-rk3588-priv1}"
SWITCH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --switch)  SWITCH=1; shift ;;
    --release) LV="-${2#-}"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

SSH=(ssh -i "$BOARD_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$BOARD_USER@$BOARD_HOST")

echo "═══ 部署新内核到板子 ═══"
RELEASE="$(make -s -C "$S" LOCALVERSION="$LV" kernelrelease)"
echo "  release: $RELEASE"
echo "  板子:    $BOARD_USER@$BOARD_HOST"

# ── 0) 板子在不在 ──────────────────────────────────────────
if ! "${SSH[@]}" true 2>/dev/null; then
  cat >&2 <<EOF
✗ 连不上板子 $BOARD_HOST
  先确认它在跑：ping $BOARD_HOST
  如果连不上（板子假死的老毛病），需要断电重启。恢复后再跑本脚本。
EOF
  exit 1
fi
"${SSH[@]}" 'uptime; echo "  当前内核: $(uname -r)"' | sed 's/^/  /'

# ── 1) 本地产物自检 + 补齐 System.map / config ─────────────
echo
echo "▶ 检查本地产物"
fail=0
for f in "$OUT/Image" "$OUT/dtb/rk3588-orangepi-5-plus.dtb"; do
  [ -e "$f" ] && echo "  ✅ ${f#$OUT/}" || { echo "  ✗ 缺 $f"; fail=1; }
done
[ -d "$OUT/modstage/lib/modules/$RELEASE" ] \
  && echo "  ✅ modstage（$(find "$OUT/modstage/lib/modules/$RELEASE" -name '*.ko' | wc -l) 个模块）" \
  || { echo "  ✗ 缺 $OUT/modstage/lib/modules/$RELEASE"; fail=1; }
[ "$fail" = 0 ] || { echo "✗ 产物不全。先跑 scripts/build/build-kernel.sh" >&2; exit 1; }

# 顺手从构建树取 System.map / .config（板子上留着方便日后对照）
cp -f "$S/System.map" "$OUT/System.map-$RELEASE" 2>/dev/null || echo "  ⚠ 没有 System.map"
cp -f "$S/.config"    "$OUT/config-$RELEASE"     2>/dev/null || true

# ── 2) 准备板子上的目录 & sudo 免交互 ──────────────────────
echo
echo "▶ 准备板子上的 $BOARD_DIR"
# 先清掉上一轮残留：scp -r 到一个已存在的目录会**嵌套**成 modstage/modstage，
# 然后 install-kernel.sh 就会找不到模块树。
"${SSH[@]}" "mkdir -p $BOARD_DIR && rm -rf $BOARD_DIR/modstage $BOARD_DIR/dtb && df -h / /boot | tail -3"

# ── 2b) 设备树比对：新编的 dtb 和板子现值一样吗？────────────────
# 设备树决定时钟/电源/引脚，DT 变了可能直接影响能不能起来。
# 我们希望它是"同一份源码编出来的同一个东西"，所以先对哈希；不一样就要在切换前想清楚。
echo
echo "▶ 设备树比对（rk3588-orangepi-5-plus.dtb）"
LOCAL_DTB_HASH="$(sha256sum "$OUT/dtb/rk3588-orangepi-5-plus.dtb" | awk '{print $1}')"
REMOTE_DTB_HASH="$("${SSH[@]}" 'for p in /boot/dtb/rockchip/rk3588-orangepi-5-plus.dtb /boot/dtb/rk3588-orangepi-5-plus.dtb /usr/lib/linux-image-*/rockchip/rk3588-orangepi-5-plus.dtb; do [ -e "$p" ] && { sha256sum "$p"; break; }; done' 2>/dev/null | awk '{print $1}')"
echo "  新编:   $LOCAL_DTB_HASH  ($(stat -c%s "$OUT/dtb/rk3588-orangepi-5-plus.dtb") 字节)"
echo "  板子:   ${REMOTE_DTB_HASH:-<没找到>}"
if [ -z "$REMOTE_DTB_HASH" ]; then
  echo "  ⚠ 板子上没找到对照设备树，无法比对（不阻塞部署）"
elif [ "$LOCAL_DTB_HASH" = "$REMOTE_DTB_HASH" ]; then
  echo "  ✅ 完全一致 —— 设备树这条线零变化"
else
  cat <<'EOF'
  ⚠⚠ 设备树哈希不一致！可能原因：
       · 厂商在原厂镜像里用了打过补丁的 DT（不在这份内核源码里）
       · 编译选项/符号不同导致只是字节排布差异
     处置建议：先跑 scripts/board/kernel-select orig 之外的回退路径演练，
               或者本次**只装不切**，用 dtc 反编译两份 DT 逐节点 diff 确认差异范围。
EOF
fi

# ── 3) 传输 ────────────────────────────────────────────────
echo
echo "▶ 传输产物（$(du -sh "$OUT/Image" "$OUT/dtb" "$OUT/modstage" | awk '{s+=$1} END{print "约 "s}')）"
echo "  Image …"
scp -q -i "$BOARD_KEY" -o StrictHostKeyChecking=no "$OUT/Image" "$BOARD_USER@$BOARD_HOST:$BOARD_DIR/Image"
echo "  设备树 …"
scp -q -r -i "$BOARD_KEY" -o StrictHostKeyChecking=no "$OUT/dtb" "$BOARD_USER@$BOARD_HOST:$BOARD_DIR/"
echo "  System.map / config …"
scp -q -i "$BOARD_KEY" -o StrictHostKeyChecking=no "$OUT/System.map-$RELEASE" "$OUT/config-$RELEASE" "$BOARD_USER@$BOARD_HOST:$BOARD_DIR/" 2>/dev/null || true
echo "  模块树 …"
scp -q -r -i "$BOARD_KEY" -o StrictHostKeyChecking=no "$OUT/modstage" "$BOARD_USER@$BOARD_HOST:$BOARD_DIR/"
# 板子离线时也能拿到 aarch64 的 mkimage（打 uInitrd 用）
# ★ 注意路径：arm64 的 deb 放在 board-debs/，**绝不能**和 amd64 的混在一起 ——
#   两者文件名同名，混放会在解包时用 aarch64 的 mkimage 覆盖掉主机的 x86-64 版本。
UBT="$(ls "$HOME/.local/kbtools/board-debs/"u-boot-tools_*_arm64.deb 2>/dev/null | head -1)"
if [ -n "$UBT" ]; then
  echo "  arm64 u-boot-tools（兜底 mkimage）: ${UBT##*/}"
  scp -q -i "$BOARD_KEY" -o StrictHostKeyChecking=no \
      "$UBT" "$BOARD_USER@$BOARD_HOST:$BOARD_DIR/u-boot-tools_arm64.deb"
else
  echo "  ⚠ 没找到 arm64 u-boot-tools（板子若无 mkimage 就打不了 uInitrd）"
  echo "     补：scripts/build/setup-host-tools.sh"
fi
echo "  板子侧脚本 …"
scp -q -i "$BOARD_KEY" -o StrictHostKeyChecking=no \
    "$REPO/scripts/board/install-kernel.sh" "$REPO/scripts/board/kernel-select" \
    "$REPO/scripts/board/verify-kernel.sh" "$REPO/scripts/board/setup-perf.sh" \
    "$REPO/scripts/board/bench-ddr-ab.sh" \
    "$BOARD_USER@$BOARD_HOST:$BOARD_DIR/"

# ── 4) 板子上执行安装 ──────────────────────────────────────
echo
echo "▶ 在板子上安装（sudo）"
"${SSH[@]}" "cat > /tmp/.a.sh <<'EOS'
#!/bin/sh
echo orangepi
EOS
chmod +x /tmp/.a.sh
cd $BOARD_DIR && chmod +x install-kernel.sh kernel-select && \
SUDO_ASKPASS=/tmp/.a.sh sudo -A ./install-kernel.sh $RELEASE $BOARD_DIR" 2>&1 | sed 's/^/  /'

# ── 5) 可选：切换并重启 ────────────────────────────────────
if [ "$SWITCH" = 1 ]; then
  echo
  echo "▶ 切换启动项到 $RELEASE 并重启"
  "${SSH[@]}" "SUDO_ASKPASS=/tmp/.a.sh sudo -A $BOARD_DIR/kernel-select $RELEASE" 2>&1 | sed 's/^/  /'
  echo "  ⚠ 重启后 SSH 会断，等待板子回来…"
  "${SSH[@]}" 'SUDO_ASKPASS=/tmp/.a.sh sudo -A reboot' 2>/dev/null || true
  sleep 20
  for i in $(seq 1 40); do
    if "${SSH[@]}" true 2>/dev/null; then
      echo "  ✅ 板子回来了（第 ${i} 次探测）"
      "${SSH[@]}" 'uname -r; uptime; lsmod | grep -c rtw89; ip -4 addr show | grep "inet " | grep -v 127.0.0.1' | sed 's/^/    /'
      exit 0
    fi
    sleep 10
  done
  echo "  ✗ 等了约 7 分钟板子还没回来 —— 需要断电重启，然后 kernel-select orig 回退" >&2
  exit 1
fi

echo
echo "✅ 部署完成（尚未切换）。要生效："
echo "    ssh -i $BOARD_KEY $BOARD_USER@$BOARD_HOST"
echo "    sudo $BOARD_DIR/kernel-select $RELEASE && sudo reboot"
echo "   或直接重跑本脚本带 --switch"
