#!/bin/bash
# 网络自愈守护：治"WiFi 关联着但流量不通"的卡死
#
# 为什么需要它（而不是只靠硬件看门狗）：
#   板子前面 5 次失联，症状是 ARP 表里 MAC 还在、ICMP 全丢、SSH 超时，
#   但**内核明显还活着**（有几次过一会儿自己又通了）。这更像 rtw89 固件/驱动
#   卡在某个状态机里，而不是内核挂住 —— 硬件看门狗对这种"活着的假死"无能为力。
#
# 分级自愈（每级之间等待并复测，不通才升级）：
#   0) 网关连续 3 次探测失败（≈3 分钟）→ 认为链路假死
#   1) 重启 NetworkManager
#   2) 重装 rtw89 驱动模块（把固件状态机整个重置）
#   3) 网卡 down/up
#   4) 连续不可达 15 分钟 → 重启板子（最后的兜底：重启后 WiFi 会重新关联）
#
# 设计约束：
#   - 探测目标是默认网关，不是外网 —— 不依赖互联网可达。
#   - 所有升级动作都有冷却时间，避免抖动时反复折腾。
#   - 每一步都写 journald，事后可复盘（journalctl -t net-watchdog）。
#   - 想关掉自动重启：touch /etc/net-watchdog-noreboot
set -uo pipefail

IFACE_CONF=/etc/net-watchdog.conf
[ -r "$IFACE_CONF" ] && . "$IFACE_CONF"

PROBE_INTERVAL="${PROBE_INTERVAL:-60}"      # 探测间隔（秒）
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"       # 连续失败几次算假死
REBOOT_AFTER_SEC="${REBOOT_AFTER_SEC:-900}" # 连续不可达多久后重启（秒），0=不重启
COOLDOWN="${COOLDOWN:-300}"                 # 两次升级动作之间的冷却（秒）

log() { logger -t net-watchdog -- "$*"; echo "[$(date +%H:%M:%S)] $*"; }

gateway() { ip route show default 2>/dev/null | awk '/default/{print $3; exit}'; }
iface()   { ip route show default 2>/dev/null | awk '/default/{print $5; exit}'; }

# 探测：网关能通就算健康
probe() {
  local gw; gw="$(gateway)"
  [ -n "$gw" ] || return 1          # 连默认路由都没了 → 不健康
  ping -c 2 -W 2 -I "$(iface)" "$gw" >/dev/null 2>&1
}

recover() { # recover <级别序号> <描述> <命令...>
  local lvl="$1" desc="$2"; shift 2
  log "▲ 升级自愈 L$lvl：$desc"
  if "$@" >>/var/log/net-watchdog.log 2>&1; then
    log "  L$lvl 命令返回成功"
  else
    log "  L$lvl 命令返回失败（继续，后面还有更高等级）"
  fi
  sleep 20
  if probe; then log "✅ L$lvl 之后网络已恢复"; return 0; else log "  L$lvl 之后仍不通"; return 1; fi
}

log "启动：探测间隔 ${PROBE_INTERVAL}s，连续 ${FAIL_THRESHOLD} 次失败判定假死，不可达 ${REBOOT_AFTER_SEC}s 后重启（0=不重启）"

fails=0
down_since=0
last_action=0

while :; do
  if probe; then
    if [ "$fails" -gt 0 ]; then log "✅ 网络恢复（此前连续失败 $fails 次）"; fi
    fails=0; down_since=0
    sleep "$PROBE_INTERVAL"
    continue
  fi

  fails=$((fails + 1))
  [ "$down_since" = 0 ] && down_since=$(date +%s)
  log "⚠ 探测失败 $fails/$FAIL_THRESHOLD（网关 $(gateway || echo 无)，网卡 $(iface || echo 无)）"

  if [ "$fails" -lt "$FAIL_THRESHOLD" ]; then
    sleep "$PROBE_INTERVAL"; continue
  fi

  # ── 判定假死：开始分级自愈 ──
  now=$(date +%s)
  if [ $((now - last_action)) -lt "$COOLDOWN" ]; then
    log "  （冷却中，$((COOLDOWN - (now - last_action)))s 后再动手）"
  else
    last_action=$now
    recover 1 "重启 NetworkManager"            systemctl restart NetworkManager \
      || recover 2 "重装 rtw89 驱动模块"       bash -c 'modprobe -r rtw89_8852be rtw89_8852b rtw89_pci rtw89_core 2>/dev/null; sleep 2; modprobe rtw89_8852be' \
      || recover 3 "网卡 down/up"              bash -c 'i=$(ip route show default | awk "/default/{print \$5; exit}"); [ -n "$i" ] && ip link set "$i" down && sleep 3 && ip link set "$i" up' \
      || log "  ⚠ 三级自愈都没救回来"
  fi

  fails=0   # 重置计数，给这一轮动作留出观察期

  # ── 最终兜底：长时间不可达就重启 ──
  down_secs=$(( $(date +%s) - down_since ))
  if [ "$REBOOT_AFTER_SEC" -gt 0 ] && [ "$down_secs" -ge "$REBOOT_AFTER_SEC" ]; then
    if [ -e /etc/net-watchdog-noreboot ]; then
      log "✗ 已连续不可达 ${down_secs}s，但 /etc/net-watchdog-noreboot 存在 → 不重启"
      down_since=$(date +%s)   # 重新计时，避免刷屏
    else
      log "✗✗ 已连续不可达 ${down_secs}s，分级自愈无效 → 重启板子（重启后 WiFi 会重新关联）"
      sync
      systemctl reboot
      sleep 300
    fi
  fi

  sleep "$PROBE_INTERVAL"
done
