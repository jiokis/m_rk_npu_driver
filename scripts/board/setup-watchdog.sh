#!/bin/bash
# 启用内核硬件看门狗（板子卡死自动重启）
#
# 为什么用 systemd 而不是自己写喂狗程序：
#   systemd 原生支持硬件看门狗（RuntimeWatchdogSec=），由 PID 1 负责喂狗，
#   不依赖任何用户态服务的存活性，也没有"喂狗程序自己被调度饿死导致误重启"的风险。
#   自写 feeder 反而更不可靠。
#
# 现状（2026-10-12 上板实测）：
#   设备树 watchdog@feaf0000 status=okay、compatible=snps,dw-wdt
#   /dev/watchdog、/dev/watchdog0、/sys/class/watchdog/watchdog0 都在
#   但**没有任何人喂它** —— 内核真挂住时板子只能等人工断电。
#
# ⚠ 上一版脚本在这里踩了个坑，记下来：
#   我用 /sys/class/watchdog/watchdog0/identity 是否存在来判断"有没有看门狗"，
#   结果这个 dw_wdt 驱动**一个 sysfs 属性都不暴露**（identity/state/timeout/nowayout
#   全部不存在，目录里只有 dev/device/power/subsystem/uevent）。
#   于是脚本误报"没找到硬件看门狗"直接退出。
#   → 判断设备是否存在要认 /dev/watchdog* 和设备目录本身，而不是可选的属性文件。
set -euo pipefail

[ "$(id -u)" = "0" ] || { echo "请用 root 运行（sudo）" >&2; exit 1; }

PROBE=1
[ "${1:-}" = "--no-probe" ] && PROBE=0

echo "═══ 启用硬件看门狗 ═══"

# ── 1) 设备存在性：认设备节点，不认属性文件 ────────────────
WDDEV=""
for c in /dev/watchdog0 /dev/watchdog; do
  [ -c "$c" ] && { WDDEV="$c"; break; }
done
[ -n "$WDDEV" ] || { echo "✗ 没有 /dev/watchdog* 字符设备，放弃" >&2; exit 1; }
echo "  设备节点:    $WDDEV"

for d in /sys/class/watchdog/watchdog*; do
  [ -d "$d" ] || continue
  echo "  sysfs:       $d"
  for a in identity state timeout nowayout pretimeout; do
    [ -e "$d/$a" ] && printf '    %-11s = %s\n' "$a" "$(cat "$d/$a" 2>/dev/null)"
  done
  [ -e "$d/identity" ] || echo "    （该驱动不暴露任何属性 —— 正常，dw_wdt 就是这样）"
done

# ── 2) 写入 systemd 配置 ───────────────────────────────────
CONF=/etc/systemd/system.conf
MARK="# ── RKNPU 逆向项目：启用硬件看门狗（由 setup-watchdog.sh 写入）──"

if grep -qF "$MARK" "$CONF" 2>/dev/null; then
  echo "▶ 移除本项目此前写入的段落"
  sed -i "/RuntimeWatchdogSec=60/d; /RebootWatchdogSec=5min/d; /$(echo "$MARK" | sed 's/[][\.*^$/]/\\&/g')/d" "$CONF"
fi

# 60 秒：足够宽松，不会因为一次长拷贝/大页分配误重启；又足够短，卡死后一分钟内自愈。
cat >> "$CONF" <<EOF

$MARK
RuntimeWatchdogSec=60
RebootWatchdogSec=5min
EOF
echo "▶ 已写入 $CONF:"
grep -B1 -A2 "RuntimeWatchdogSec" "$CONF" | sed 's/^/    /'

# ── 3) 让 PID 1 接管看门狗 ─────────────────────────────────
echo "▶ systemctl daemon-reexec（重启 PID 1，不断 SSH）…"
systemctl daemon-reexec
sleep 3

WUSEC="$(systemctl show -p RuntimeWatchdogUSec --value 2>/dev/null || echo '')"
echo "  systemd 报告的看门狗周期: ${WUSEC:-<空>}"

# ── 4) 功能自证：真的能设超时、真的能读回来 ────────────────
# 顺序很重要：先让 systemd 接管，再做这个会"武装"看门狗的探测。
# 万一 nowayout=1 导致写 'V' 也关不掉，systemd 已经在喂狗，不会误重启。
if [ "$PROBE" = 1 ] && [ -n "$WUSEC" ] && [ "$WUSEC" != "0" ]; then
  echo "▶ 功能自证（设超时→读回→写 V 关闭）"
  # 注意：systemd 接管后会**独占持有** /dev/watchdog0，我们再去 open 会拿到 EBUSY。
  # 那个 EBUSY 本身就是"看门狗确实被 PID 1 拿着"的铁证，所以要把 EBUSY 当成功，
  # 不能让 python 退出码 1 触发 set -e 把整个脚本带崩。
  python3 - "$WDDEV" <<'PY' || true
import fcntl, struct, sys, os, errno
DEV = sys.argv[1]
WDIOC_SETTIMEOUT = 0xC0045706   # _IOWR('W', 6, int)
WDIOC_GETTIMEOUT = 0x80045707   # _IOR('W', 7, int)
def getto(fd):
    return struct.unpack('i', fcntl.ioctl(fd, WDIOC_GETTIMEOUT, struct.pack('i', 0)))[0]
try:
    fd = os.open(DEV, os.O_WRONLY)
except OSError as e:
    if e.errno == errno.EBUSY:
        print(f"  ✅ {DEV} 被独占占用（EBUSY）—— 正是 systemd 在持有它，看门狗确实在跑")
    else:
        print(f"  ✗ 打不开 {DEV}: {e}")
    sys.exit(0)
try:
    print(f"  当前超时:      {getto(fd)} 秒")
    fcntl.ioctl(fd, WDIOC_SETTIMEOUT, struct.pack('i', 60))
    print(f"  设为 60 秒后读回: {getto(fd)} 秒")
    print("  ✅ 硬件看门狗可设超时、可读回 —— 功能正常")
finally:
    # 写魔数 'V' 优雅关闭，否则这个进程一退出，看门狗就会在超时后重启板子
    try:
        os.write(fd, b'V')
    except Exception as e:
        print(f"  ⚠ 写 'V' 失败: {e}（若 nowayout=1 属正常，systemd 会接手喂狗）")
    os.close(fd)
PY
elif [ "$PROBE" = 1 ]; then
  echo "  ⚠ systemd 没接管看门狗（周期为空或 0），跳过功能自证以免误触发重启"
fi

# ── 5) 结论 ────────────────────────────────────────────────
echo
if [ -n "$WUSEC" ] && [ "$WUSEC" != "0" ]; then
  echo "✅ 硬件看门狗已激活：systemd(PID 1) 以 $WUSEC 为周期喂狗，内核真挂住时约 60 秒自动重启"
  echo "   查看: systemctl show -p RuntimeWatchdogUSec   /   journalctl -b | grep -i watchdog"
else
  cat <<'EOF'
✗ systemd 没有接管看门狗（RuntimeWatchdogUSec 为空或 0）。
  常见原因：
    1) systemd 版本太老（< v243）—— Ubuntu 24.04 不会是这个
    2) 该驱动不暴露 sysfs 属性，systemd 枚举设备时挑不出来（dw_wdt 正是这种）
  下一步：改用自写 feeder（scripts/board/ 下可加），或先确认
    cat /proc/sys/kernel/watchdog   /   dmesg | grep -i wdt
EOF
  exit 2
fi
