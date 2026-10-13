#!/usr/bin/env bash
# deploy_board.sh — 一条命令把训练好的 DDPM 部署到板上并出图
#
# 用法：
#   bash deploy/diffusion/deploy_board.sh <ckpt.pt> <name> [n] [steps]
# 例：
#   bash deploy/diffusion/deploy_board.sh deploy/diffusion/ckpt/ddpm_mnist.pt ddpm_mnist 64 100
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
V=/home/dunaandone/rknn-venv/bin/python
BOARD=orangepi@192.168.31.128
KEY=~/.ssh/keys/opi_board
SSH="ssh -i $KEY -o StrictHostKeyChecking=no"
SCP="scp -i $KEY -o StrictHostKeyChecking=no -q"

CKPT="${1:?用法: deploy_board.sh <ckpt.pt> <name> [n] [steps]}"
NAME="${2:?需要 name}"
N="${3:-64}"
STEPS="${4:-100}"

RKNN="$REPO/deploy/diffusion/rknn/$NAME.rknn"
META="$REPO/deploy/diffusion/rknn/$NAME.json"
OUT="$REPO/deploy/diffusion/out"

echo "▶ 1/4 导出 + 转 RKNN"
$V "$REPO/deploy/diffusion/to_rknn.py" --ckpt "$CKPT" --out "$RKNN" --calib-n 32

echo "▶ 2/4 笔记本 torch 参考采样（同种子，用于对比）"
$V "$REPO/deploy/diffusion/sample.py" --backend torch --ckpt "$CKPT" \
   --n "$N" --steps "$STEPS" --seed 0 \
   --out "$OUT/${NAME}_torch.png" --save-raw "$OUT/${NAME}_torch.npy"

echo "▶ 3/4 拷到板子并在 NPU 上采样"
$SCP "$RKNN" "$META" "$REPO/deploy/diffusion/sample.py" "$BOARD:/tmp/"
$SSH "$BOARD" "cd /tmp && python3 sample.py --backend rknn --model /tmp/$NAME.rknn \
  --meta /tmp/$NAME.json --n $N --steps $STEPS --seed 0 \
  --out /tmp/${NAME}_npu.png --save-raw /tmp/${NAME}_npu.npy"

echo "▶ 4/4 取回成片"
mkdir -p "$OUT"
$SCP "$BOARD:/tmp/${NAME}_npu.png" "$OUT/${NAME}_npu.png"
$SCP "$BOARD:/tmp/${NAME}_npu.npy" "$OUT/${NAME}_npu.npy"

$V "$REPO/deploy/diffusion/check_numeric.py" --compare \
   "$OUT/${NAME}_torch.npy" "$OUT/${NAME}_npu.npy" || true

echo "✅ 完成："
ls -la "$OUT/${NAME}_npu.png" "$OUT/${NAME}_torch.png" "$RKNN"
