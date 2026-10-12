#!/bin/bash
# 板子网络与远程访问加固：开机自动连 WiFi + 自动起 SSH + 关掉 WiFi 省电
#
# 为什么关 WiFi 省电（wifi.powersave=2）：
#   RTL8852BE 在省电模式下会进入 PS-Poll 状态机，掉线/假死的概率显著上升。
#   板子前面 5 次"关联着但流量不通"的失联，省电模式是头号嫌疑。
#   代价是待机功耗略增 —— 对插电常开的开发板无所谓。
#
# 装进新镜像的意义：
#   新内核的模块装在独立目录 /lib/modules/<新 release>/，只要模块树装全 + 网络配置在此，
#   开机就是"自动连 WiFi、自动起 SSH"，不需要人工干预（也不需要显示器/串口）。
set -euo pipefail

[ "$(id -u)" = "0" ] || { echo "请用 root 运行（sudo）" >&2; exit 1; }

echo "═══ 网络与远程访问加固 ═══"

# ── 1) 全局关闭 WiFi 省电 ──────────────────────────────────
echo "▶ NetworkManager: 关闭 WiFi 省电"
mkdir -p /etc/NetworkManager/conf.d
cat > /etc/NetworkManager/conf.d/99-wifi-powersave-off.conf <<'EOF'
# RKNPU 逆向项目：关闭 WiFi 省电，规避 RTL8852BE 的 PS 状态机假死
[connection]
wifi.powersave = 2
EOF
echo "  /etc/NetworkManager/conf.d/99-wifi-powersave-off.conf"

# ── 2) 现有的有线/无线连接都设成开机自动连接 ───────────────
if command -v nmcli >/dev/null; then
  echo "▶ 连接配置：自动连接 + 无限重试 + 高优先级"
  nmcli -t -f NAME,TYPE connection show | while IFS=: read -r name type; do
    case "$type" in
      wifi|802-11-wireless|ethernet|802-3-ethernet) ;;
      *) continue ;;
    esac
    nmcli connection modify "$name" \
      connection.autoconnect yes \
      connection.autoconnect-retries 0 \
      connection.autoconnect-priority 100 >/dev/null 2>&1 \
      && echo "  ✅ $name ($type)" \
      || echo "  ⚠ $name ($type) 修改失败"
  done
  echo "▶ 重新加载 NetworkManager"
  systemctl reload NetworkManager 2>/dev/null || systemctl restart NetworkManager
  sleep 5
  echo
  echo "── 当前连接状态 ──"
  nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status | sed 's/^/  /'
else
  echo "⚠ 没有 nmcli，跳过（可能用的是 systemd-networkd / netplan）"
fi

# ── 3) SSH 开机自启 ────────────────────────────────────────
echo "▶ SSH"
# 这里也踩过坑：原来用 `systemctl list-unit-files | grep -q '^ssh\.service'` 判断，
# 但板子上 ssh.service 明明是 loaded/active/running，grep 却没命中（列表输出格式
# 在不同 systemd 版本/终端宽度下不一致）。改成直接问 systemctl，别去解析列表文本。
SSH_UNIT=""
for u in ssh.service sshd.service ssh.socket; do
  if systemctl list-unit-files --no-legend "$u" 2>/dev/null | grep -q .; then SSH_UNIT="$u"; break; fi
done
if [ -z "$SSH_UNIT" ]; then
  # 最后兜底：只要 22 端口有人在听，就认为远程访问是通的，不做改动
  if ss -ltn 2>/dev/null | grep -q ':22 '; then
    echo "  ⚠ 没找到 ssh 相关 unit，但 22 端口在监听 —— 远程访问本身没问题，跳过"
  else
    echo "  ✗ 既没有 ssh unit，22 端口也没人听 —— 远程访问有风险"
  fi
else
  systemctl enable "$SSH_UNIT" >/dev/null 2>&1 || true
  systemctl start  "$SSH_UNIT" >/dev/null 2>&1 || true
  echo "  $SSH_UNIT: enabled=$(systemctl is-enabled "$SSH_UNIT" 2>/dev/null) active=$(systemctl is-active "$SSH_UNIT" 2>/dev/null)"
  echo "  22 端口: $(ss -ltn 2>/dev/null | grep -c ':22 ') 个监听"
fi

# ── 4) 顺手清掉 sources 里那个死掉的 Docker 代理 ───────────
# 现状：apt 源里挂着 192.168.31.74:3128 这个已经不存在的代理，apt 每次都要等超时。
PROXY_FILE=/etc/apt/apt.conf.d/95docker-proxy
if [ -e "$PROXY_FILE" ] || grep -rqs "192.168.31.74:3128" /etc/apt/; then
  echo "▶ 发现失效的 apt 代理配置，注释掉"
  mkdir -p /etc/apt/apt.conf.d
  [ -e "$PROXY_FILE" ] && mv "$PROXY_FILE" "$PROXY_FILE.disabled"
  for f in $(grep -rls "192.168.31.74:3128" /etc/apt/ 2>/dev/null); do
    sed -i 's|^\(.*192\.168\.31\.74:3128.*\)$|# \1   # 由 setup-network.sh 注释：代理已不存在|' "$f"
    echo "  $f"
  done
fi

echo
echo "✅ 加固完成"
echo "   ⚠ 若希望「关 WiFi 省电」立即生效，需要重连一次 WiFi："
echo "      nmcli con down <名字> && nmcli con up <名字>"
echo "      （会瞬断 SSH；网络自愈守护在场时也可以直接等它自己重连）"
