#!/bin/bash
# 启用内核硬件看门狗（板子卡死自动重启）
#
# 为什么用 systemd 而不是自己写喂狗程序：
#   systemd 原生支持硬件看门狗（RuntimeWatchdogSec=），由 PID 1 负责喂狗，
#   不依赖任何用户态服务的存活性，也没有"喂狗程序自己被调度饿死导致误重启"的风险。
#   自写 feeder 反而更不可靠。
#
# 现状（2026-10-12 实测）：
#   /dev/watchdog、/dev/watchdog0 存在，CONFIG_DW_WATCHDOG=y（编进内核），
#   设备树 watchdog@feaf0000 在位，但**没有任何人喂它** —— 所以内核真挂住时
#   板子只能等用户手动断电。这就是前面 5 次"卡死"里至少有几次本可自愈的原因。
#
# 注意：本脚本只治"内核挂住"。如果是 rtw89 固件挂了但内核还活着（更可能的那种），
#       硬件看门狗不会触发 —— 那种情况由 setup-net-watchdog.sh 负责。
set -euo pipefail

[ "$(id -u)" = "0" ] || { echo "请用 root 运行（sudo）" >&2; exit 1; }

echo "═══ 启用硬件看门狗 ═══"

# ── 1) 确认硬件看门狗真的存在 ──────────────────────────────
WD=""
for d in /sys/class/watchdog/watchdog*; do
  [ -e "$d/identity" ] || continue
  WD="$d"
  echo "  设备:        $d"
  echo "  identity:    $(cat "$d/identity" 2>/dev/null)"
  echo "  timeout:     $(cat "$d/timeout" 2>/dev/null) 秒"
  echo "  state:       $(cat "$d/state" 2>/dev/null)"
  echo "  nowayout:    $(cat "$d/nowayout" 2>/dev/null)   (0=可关闭，1=不可关闭)"
done
[ -n "$WD" ] || { echo "✗ 没找到硬件看门狗，放弃" >&2; exit 1; }

STATE="$(cat "$WD/state" 2>/dev/null || echo unknown)"
if [ "$STATE" = "active" ]; then
  echo "  ⚠ 看门狗已处于 active —— 可能 systemd 已经在喂了，或者有别的程序占用。"
  echo "    继续执行只会把超时时间调一下，无副作用。"
fi

# ── 2) 写入 systemd 配置 ───────────────────────────────────
CONF=/etc/systemd/system.conf
MARK="# ── RKNPU 逆向项目：启用硬件看门狗（由 setup-watchdog.sh 写入）──"

if grep -qF "$MARK" "$CONF" 2>/dev/null; then
  echo "▶ systemd 配置里已有本项目写入的段落，先移除旧的"
  sed -i "/$(echo "$MARK" | sed 's/[][\.*^$/]/\\&/g')/,/RuntimeWatchdogSec=/d" "$CONF"
fi

# 60 秒：足够宽松，不会因为一次长 GC / 大页拷贝 / 内核编译把板子误重启；
#         又足够短，卡死后一分钟内自愈。RebootWatchdogSec 让 reboot 卡住也能兜住。
cat >> "$CONF" <<EOF

$MARK
RuntimeWatchdogSec=60
RebootWatchdogSec=5min
EOF

echo "▶ 已写入 $CONF:"
grep -A3 "RuntimeWatchdogSec" "$CONF" | sed 's/^/    /'

# ── 3) 让 PID 1 重新读取配置并接管看门狗 ───────────────────
echo "▶ systemctl daemon-reexec（重启 PID 1，不断 SSH 连接）…"
systemctl daemon-reexec
sleep 3

echo
echo "── 验证 ──"
for d in /sys/class/watchdog/watchdog*; do
  [ -e "$d/identity" ] || continue
  printf '  %s: state=%s timeout=%s\n' "$d" "$(cat $d/state)" "$(cat $d/timeout)"
done
echo
if [ "$(cat "$WD/state" 2>/dev/null)" = "active" ]; then
  echo "✅ 看门狗已激活：内核真挂住时，60 秒内自动重启"
else
  echo "⚠ 看门狗 state=$(cat "$WD/state")，不是 active。"
  echo "  常见原因：这台机器上 systemd 版本较老（<v243）或 watchdog 被别的驱动占用。"
  echo "  用 'systemctl show -p RuntimeWatchdogUSec' 看实际值。"
  exit 2
fi
echo
echo "关闭方法：删除 $CONF 里那段，再 systemctl daemon-reexec"
