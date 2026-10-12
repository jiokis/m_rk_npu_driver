#!/bin/bash
# 性能固化：把 NPU 与 DDR 的频率钉在 performance
#
# ═══ 为什么值得做 ═══
# docs/25 实测：这块板子的 DMC（DDR 控制器）默认由 `dmc_ondemand` 调速，**默认就钉在
# 534 MHz，而硬件上限是 2400 MHz**。同一个进程内 A/B 对比：
#     534 MHz  → 51.80 ms
#     2400 MHz → 20.13 ~ 28.25 ms
# 也就是**一条 echo performance 就能拿到约 1.8~2.5 倍**。在"最大化 NPU 利用率"这件事上，
# 这是目前所有发现里单点收益最大的一个 —— 比多核调度（实测仅 1.28×）大得多。
# 原因很直白：NPU 三核跑不动不是因为算力不够，是 DRAM 带宽把它喂不饱。
#
# ═══ 代价（要知情）═══
# 钉频 = 放弃动态调频。板子功耗与发热会上升，风扇会更常转。
# 插电常开的开发板无所谓；如果是电池供电或者对温度敏感，就别开。
#
# 用法：
#   sudo scripts/board/setup-perf.sh            # 立即生效 + 装成开机自启
#   sudo scripts/board/setup-perf.sh --now      # 只本次生效，不持久化
#   sudo scripts/board/setup-perf.sh --off      # 关掉并取消自启
#   sudo scripts/board/setup-perf.sh --bench    # 顺带跑一次 A/B 对比，看真实提升
set -euo pipefail

[ "$(id -u)" = "0" ] || { echo "请用 root 运行（sudo）" >&2; exit 1; }

SVC=/etc/systemd/system/rknpu-perf.service
MODE=install
BENCH=0
for a in "$@"; do
  case "$a" in
    --now)  MODE=now ;;
    --off)  MODE=off ;;
    --bench) BENCH=1 ;;
    *) echo "未知参数: $a" >&2; exit 1 ;;
  esac
done

DMC=/sys/class/devfreq/dmc
NPU=/sys/class/devfreq/fdab0000.npu

show() {
  for d in "$NPU" "$DMC"; do
    [ -d "$d" ] || continue
    printf '  %-32s cur=%-10s min=%-10s max=%-10s gov=%s\n' \
      "$(basename "$d")" "$(cat "$d/cur_freq" 2>/dev/null)" \
      "$(cat "$d/min_freq" 2>/dev/null)" "$(cat "$d/max_freq" 2>/dev/null)" \
      "$(cat "$d/governor" 2>/dev/null)"
  done
}

echo "═══ NPU / DDR 性能固化 ═══"
echo "▶ 当前状态"; show

# ── 关掉并取消自启 ─────────────────────────────────────────
if [ "$MODE" = off ]; then
  systemctl disable --now rknpu-perf 2>/dev/null || true
  rm -f "$SVC"; systemctl daemon-reload
  for d in "$NPU" "$DMC"; do
    [ -d "$d" ] && echo ondemand > "$d/governor" 2>/dev/null || true
  done
  echo "▶ 已恢复动态调频"; show
  exit 0
fi

# ── 钉频 ───────────────────────────────────────────────────
pin() { # pin <devfreq目录> <名字>
  local d="$1" n="$2"
  [ -d "$d" ] || { echo "  ⚠ 没有 $d，跳过 $n"; return 0; }
  local govs; govs="$(cat "$d/available_governors" 2>/dev/null || echo '')"
  if ! echo "$govs" | grep -qw performance; then
    echo "  ⚠ $n 不支持 performance（可选: $govs），跳过"; return 0
  fi
  # 先把 min_freq 顶到 max，避免 governor 换过去又被下限拖住
  local mx; mx="$(cat "$d/max_freq" 2>/dev/null || echo '')"
  [ -n "$mx" ] && echo "$mx" > "$d/min_freq" 2>/dev/null || true
  echo performance > "$d/governor"
  printf '  ✅ %-14s → %s Hz (governor=performance)\n' "$n" "$(cat "$d/cur_freq")"
}

echo "▶ 钉频"
pin "$NPU" "NPU"
pin "$DMC" "DDR/DMC"

# ── 可选的 A/B 实测 ────────────────────────────────────────
if [ "$BENCH" = 1 ]; then
  echo
  echo "▶ A/B 实测（跑 yolov5s，先 performance 再换回 dmc_ondemand 对照）"
  MODEL=/opt/rknn-models/yolov5s-640-640.rknn
  PY=""
  for c in "$HOME/rknn-gateway/venv/bin/python3" /usr/bin/python3; do
    [ -x "$c" ] && "$c" -c "import rknnlite" 2>/dev/null && { PY="$c"; break; }
  done
  if [ -z "$PY" ] || [ ! -e "$MODEL" ]; then
    echo "  ⚠ 缺少 rknnlite 或 $MODEL，跳过实测"
  else
    run() { "$PY" - "$MODEL" <<'PY'
import sys, time, numpy as np
from rknnlite.api import RKNNLite
r = RKNNLite(); r.load_rknn(sys.argv[1]); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
x = np.zeros((1,640,640,3), dtype=np.uint8)
for _ in range(5): r.inference(inputs=[x])
ts=[]
for _ in range(30):
    t=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t)*1000)
ts.sort(); print(f"{ts[len(ts)//2]:.2f}")
r.release()
PY
    }
    echo "  performance(DDR 2400) 中位: $(run) ms"
    echo ondemand > "$DMC/governor" 2>/dev/null || true
    sleep 2
    echo "  dmc_ondemand(默认 534) 中位: $(run) ms"
    echo performance > "$DMC/governor" 2>/dev/null || true
    echo "  （已恢复 performance）"
  fi
fi

[ "$MODE" = now ] && { echo; echo "▶ --now：本次生效，未持久化"; show; exit 0; }

# ── 持久化 ─────────────────────────────────────────────────
echo "▶ 装成开机自启：$SVC"
cat > "$SVC" <<'EOF'
[Unit]
Description=把 NPU 与 DDR(DMC) 频率钉在 performance（RKNPU 逆向项目）
# devfreq 设备在 platform 驱动 probe 之后才出现，所以要等 sysinit 之后
After=sysinit.target
Wants=sysinit.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
  for d in /sys/class/devfreq/fdab0000.npu /sys/class/devfreq/dmc; do \
    [ -d "$d" ] || continue; \
    mx=$(cat $d/max_freq 2>/dev/null); [ -n "$mx" ] && echo $mx > $d/min_freq 2>/dev/null; \
    echo performance > $d/governor 2>/dev/null; \
  done; \
  echo "NPU=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null) DMC=$(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null)"'
ExecStop=/bin/bash -c '\
  for d in /sys/class/devfreq/fdab0000.npu /sys/class/devfreq/dmc; do \
    [ -d "$d" ] && echo ondemand > $d/governor 2>/dev/null; \
  done'
RemainAfterExit=yes
StandardOutput=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now rknpu-perf.service
sleep 2
systemctl --no-pager -l status rknpu-perf.service 2>/dev/null | head -10 || true

echo
echo "── 最终状态 ──"; show
echo
echo "✅ 已固化。关闭：sudo $0 --off"
echo "   ⚠ 代价提醒：钉频会抬高功耗与温度，风扇会更常转。"
