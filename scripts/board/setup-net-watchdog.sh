#!/bin/bash
# 安装网络自愈守护（systemd 服务 + 脚本 + 默认配置）
set -euo pipefail

[ "$(id -u)" = "0" ] || { echo "请用 root 运行（sudo）" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "═══ 安装网络自愈守护 ═══"

install -m 0755 "$HERE/net-watchdog.sh" /usr/local/sbin/net-watchdog.sh

if [ ! -e /etc/net-watchdog.conf ]; then
  cat > /etc/net-watchdog.conf <<'EOF'
# 网络自愈守护参数（改完 systemctl restart net-watchdog）
PROBE_INTERVAL=60      # 探测间隔（秒）
FAIL_THRESHOLD=3       # 连续失败几次判定链路假死
REBOOT_AFTER_SEC=900   # 连续不可达多久后重启（秒）；0 = 永不自动重启
COOLDOWN=300           # 两次升级动作之间的冷却（秒）
# ★ PEER：除了网关，还要能到达谁才算"健康"。
#   血的教训：只探网关会漏掉「板子能 ping 通路由器、但开发机够不着板子」这种假死，
#   守护会认为一切正常、永不升级（2026-10-12 实际发生过）。
#   ⚠ 别填经常关机的机器，否则误判；填 0 或留空即关闭这项检查。
PEER=192.168.31.74     # 开发机（那台跑编译的笔记本）
EOF
  echo "  写入 /etc/net-watchdog.conf"
else
  # 已存在：补上新加的 PEER 项（旧版本配置里没有，缺了它就退化成"只探网关"，
  # 会漏掉端到端假死 —— 这正是 2026-10-12 那次没能自愈的原因）
  if ! grep -qE '^\s*PEER=' /etc/net-watchdog.conf; then
    cat >> /etc/net-watchdog.conf <<'EOF'

# ★ 2026-10-12 补：只探网关会漏掉「板子能到路由器、但开发机够不着板子」的假死
#   （那次守护一直认为健康、永不升级）。PEER 必须端到端可达才算健康。
#   ⚠ 别填经常关机的机器；填 0 或留空即关闭这项检查。
PEER=192.168.31.74     # 开发机
EOF
    echo "  ⚠ 旧配置缺 PEER，已补上（这是上次没能自愈的根因）"
  fi
fi

cat > /etc/systemd/system/net-watchdog.service <<'EOF'
[Unit]
Description=网络自愈守护（rtw89 假死恢复 / 长时间不可达后重启）
After=network-online.target NetworkManager.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/sbin/net-watchdog.sh
Restart=always
RestartSec=30
# 用 root：要 modprobe / ip link / systemctl reboot
Nice=5

[Install]
WantedBy=multi-user.target
EOF

# 独立日志文件轮转（journald 里也有，双份方便复盘）
cat > /etc/logrotate.d/net-watchdog <<'EOF'
/var/log/net-watchdog.log {
  weekly
  rotate 4
  compress
  missingok
  notifempty
}
EOF

touch /var/log/net-watchdog.log

systemctl daemon-reload
systemctl enable --now net-watchdog.service
sleep 2
systemctl --no-pager -l status net-watchdog.service | head -12 || true
echo
echo "✅ 已安装并启动。查看日志：journalctl -t net-watchdog -f"
echo "   关闭自动重启：touch /etc/net-watchdog-noreboot"
echo "   完全停用：systemctl disable --now net-watchdog"
