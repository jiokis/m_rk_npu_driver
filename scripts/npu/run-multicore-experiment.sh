#!/bin/bash
# 步骤1：多核对比实验（core_mask 0x1 vs 0x7）
# 在笔记本上执行；自动 scp 脚本、跑两次抓取、把结果拉回来
set -u
K=~/.ssh/keys/opi_board
B=orangepi@192.168.31.128
SSH="ssh -i $K -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
P="$(cd "$(dirname "$0")/../.." && pwd)"

echo "═══ 步骤1：多核对比实验 ═══"
scp -q -i $K "$P/scripts/npu/capture-run.py" "$P/src/probes/rknpu_capture.c" $B:/home/orangepi/ || exit 1
$SSH $B 'bash -s' <<'EOF'
cd ~
export SUDO_ASKPASS=/tmp/.a.sh
printf '#!/bin/sh\necho orangepi\n' > /tmp/.a.sh; chmod +x /tmp/.a.sh
gcc -shared -fPIC -O2 -o rknpu_capture.so rknpu_capture.c -ldl -lpthread || exit 1
rm -rf /tmp/cm1 /tmp/cm7
for cm in 1 7; do
  echo "── core_mask=$cm ──"
  /usr/bin/time -f "  用时 %e s" python3 capture-run.py /opt/rknn-models/yolov5s-640-640.rknn $cm /tmp/cm$cm cm$cm 2>&1 | tail -3
  echo "  样本 $(ls /tmp/cm$cm/regcmd_*.bin 2>/dev/null | wc -l) 个"
  echo "  NPU 负载: $(sudo -A cat /sys/kernel/debug/rknpu/load 2>/dev/null)"
done
cd /tmp && tar czf cm.tgz cm1 cm7
EOF
rm -rf "$P/data/multicore" && mkdir -p "$P/data/multicore"
scp -q -i $K $B:/tmp/cm.tgz "$P/data/multicore/" && (cd "$P/data/multicore" && tar xzf cm.tgz && rm cm.tgz)
echo "  结果已拉回 data/multicore/{cm1,cm7}"
