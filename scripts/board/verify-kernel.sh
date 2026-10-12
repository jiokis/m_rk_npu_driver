#!/bin/bash
# 新内核上板自检：切换并重启之后跑这个，一条命令判生死
#
# 用法（在板子上）：sudo scripts/board/verify-kernel.sh [期望release]
set -uo pipefail

EXPECT="${1:-6.1.99-rockchip-rk3588-priv1}"
PASS=0; FAIL=0; WARN=0
ok()   { printf '  ✅ %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  ✗ %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  ⚠ %s\n' "$*"; WARN=$((WARN+1)); }
sec()  { printf '\n── %s ──\n' "$*"; }

echo "═══════════ 新内核上板自检 ═══════════"
echo "  期望 release: $EXPECT"
echo "  实际 release: $(uname -r)"

sec "1. 内核版本"
[ "$(uname -r)" = "$EXPECT" ] && ok "release 匹配" || bad "release 不匹配（当前 $(uname -r)）"

sec "2. NPU 驱动（这是本次改造的主角）"
if dmesg 2>/dev/null | grep -qi rknpu; then
  ok "dmesg 里有 rknpu 记录"
  dmesg | grep -i rknpu | tail -5 | sed 's/^/      /'
else
  warn "dmesg 里没有 rknpu 记录（可能日志已滚掉）"
fi
if [ -e /dev/dri/renderD129 ]; then
  ok "/dev/dri/renderD129 在位（NPU 渲染节点）"
  ls -la /dev/dri/ | sed 's/^/      /'
else
  bad "/dev/dri/renderD129 不存在 —— NPU 没起来"
fi
if [ -e /sys/kernel/debug/rknpu/version ]; then
  ok "debugfs version: $(cat /sys/kernel/debug/rknpu/version 2>/dev/null)"
elif [ -d /sys/kernel/debug/rknpu ]; then
  ok "debugfs /sys/kernel/debug/rknpu 存在"
  ls /sys/kernel/debug/rknpu/ | sed 's/^/      /'
else
  warn "没有 /sys/kernel/debug/rknpu（ROCKCHIP_RKNPU_DEBUG_FS）"
fi
# 平台驱动绑定
if [ -e /sys/bus/platform/drivers/RKNPU/fdab0000.npu ]; then
  ok "平台驱动已绑定 fdab0000.npu"
else
  warn "/sys/bus/platform/drivers/RKNPU/fdab0000.npu 不存在"
fi
# devfreq
for d in /sys/class/devfreq/*npu*; do
  [ -d "$d" ] && ok "devfreq: $(basename $d) 当前 $(cat $d/cur_freq 2>/dev/null) Hz，governor=$(cat $d/governor 2>/dev/null)"
done

sec "3. WiFi（挂了就彻底失联，最关键）"
IFACE="$(ls /sys/class/net | grep -E '^wl' | head -1)"
if [ -n "$IFACE" ]; then
  ok "无线网卡 $IFACE 存在"
  if lsmod | grep -q rtw89; then
    ok "rtw89 模块已加载（$(lsmod | grep -c rtw89) 个）"
  else
    bad "rtw89 模块没加载 —— 新内核的模块树可能没装全"
  fi
  IP="$(ip -4 addr show "$IFACE" 2>/dev/null | awk '/inet /{print $2}')"
  [ -n "$IP" ] && ok "$IFACE 已获取 IP: $IP" || bad "$IFACE 没有 IP"
  GW="$(ip route show default | awk '/default/{print $3; exit}')"
  if [ -n "$GW" ] && ping -c2 -W2 "$GW" >/dev/null 2>&1; then ok "网关 $GW 可达"; else bad "网关不可达"; fi
else
  bad "没找到无线网卡接口"
fi
# 有线网卡（r8125，可选的备用通道）
if lsmod | grep -q r8125; then
  ok "r8125 有线网卡驱动已加载"
else
  warn "r8125 未加载（没接网线时正常）"
fi

sec "4. SSH / 远程访问"
if ss -ltn 2>/dev/null | grep -q ':22 '; then ok "22 端口在监听"; else bad "22 端口没监听"; fi
systemctl is-active ssh >/dev/null 2>&1 && ok "ssh.service active" || warn "ssh.service 不是 active"

sec "5. 风扇与温控"
if lsmod | grep -q pwm_fan || [ -e /sys/class/hwmon/hwmon*/pwm1 ]; then
  ok "风扇驱动在位"
else
  warn "风扇驱动不在（可能有内置方案）"
fi
for t in /sys/class/thermal/thermal_zone*/temp; do
  [ -e "$t" ] && printf '      %s = %s°C\n' "${t%/temp}" "$(( $(cat $t) / 1000 ))"
done | head -6

sec "6. 系统健康"
ok "uptime: $(uptime | sed 's/.*up/up/')"
ok "看门狗: $(systemctl show -p RuntimeWatchdogUSec --value 2>/dev/null)"
ok "net-watchdog: $(systemctl is-active net-watchdog 2>/dev/null) / $(systemctl is-enabled net-watchdog 2>/dev/null)"
dmesg 2>/dev/null | grep -iE "call trace|kernel panic|oops|BUG:" | tail -3 | sed 's/^/      /'
if dmesg 2>/dev/null | grep -qiE "kernel panic|Call trace"; then bad "dmesg 里有 panic/Call trace"; else ok "没有 panic / Call trace"; fi

sec "7. NPU 实跑一次（yolov5s）"
MODEL=/opt/rknn-models/yolov5s-640-640.rknn
PY=""
for c in "$HOME/rknn-gateway/venv/bin/python3" /usr/bin/python3 "$(command -v python3)"; do
  [ -x "$c" ] && "$c" -c "import rknnlite" 2>/dev/null && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  warn "找不到带 rknnlite 的 python，跳过 NPU 实跑"
elif [ ! -e "$MODEL" ]; then
  warn "找不到 $MODEL，跳过"
else
  echo "      用 $PY 跑 $MODEL …"
  "$PY" - "$MODEL" <<'PY' 2>&1 | sed 's/^/      /'
import sys, time, numpy as np
from rknnlite.api import RKNNLite
m = sys.argv[1]
r = RKNNLite()
assert r.load_rknn(m) == 0, "load_rknn 失败"
assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, "init_runtime 失败"
x = np.zeros((1,640,640,3), dtype=np.uint8)
for _ in range(3): r.inference(inputs=[x])
ts=[]
for _ in range(10):
    t=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t)*1000)
ts.sort()
print(f"跑通 ✅  中位 {ts[len(ts)//2]:.2f} ms  最快 {ts[0]:.2f} ms  最慢 {ts[-1]:.2f} ms（3核）")
r.release()
PY
  [ "${PIPESTATUS[0]}" = 0 ] && ok "NPU 推理跑通" || bad "NPU 推理失败"
fi

sec "结论"
echo "  通过 $PASS / 警告 $WARN / 失败 $FAIL"
if [ "$FAIL" = 0 ]; then
  echo "  ✅ 基本可用。若警告里有 WiFi 相关项，先别重启，就地排查。"
else
  cat <<'EOF'
  ✗ 有硬失败项。回退办法：
       sudo scripts/board/kernel-select orig   （或板上 /root/kernel-deploy/kernel-select）
       sudo reboot
EOF
fi
exit $([ "$FAIL" = 0 ] && echo 0 || echo 1)
