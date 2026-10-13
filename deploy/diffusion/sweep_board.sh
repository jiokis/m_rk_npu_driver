#!/usr/bin/env bash
# sweep_board.sh — 把 sweep/ 下所有 .rknn 拷到板子，统一计时，取回结果
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BOARD=orangepi@192.168.31.128
KEY=~/.ssh/keys/opi_board
SCP="scp -i $KEY -o StrictHostKeyChecking=no -q"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no"

$SSH "$BOARD" 'rm -rf /tmp/sweep && mkdir -p /tmp/sweep'
$SCP "$REPO/deploy/diffusion/sweep/"*.rknn "$REPO/deploy/diffusion/sweep/table.json" "$BOARD:/tmp/sweep/"
$SCP "$REPO/deploy/diffusion/bench_sweep.py" "$BOARD:/tmp/sweep/"

$SSH "$BOARD" 'cd /tmp/sweep && python3 bench_sweep.py 2>/dev/null' | grep -a "^SWEEP"
$SCP "$BOARD:/tmp/sweep/bench.json" "$REPO/deploy/diffusion/sweep/bench.json"
echo "✅ 结果 → deploy/diffusion/sweep/bench.json"
