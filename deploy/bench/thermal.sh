#!/bin/bash
# 长时热稳定压测：循环跑 C 完整链路，记录 FPS/温度/频率
cd /home/orangepi/cbench
LOG=/home/orangepi/cbench/thermal.log
: > $LOG
echo "t_s,zone0,zone1,zone2,npu_freq,gpu_freq,cpu_freq,fps,boxes" >> $LOG
T0=$(date +%s)
for i in $(seq 1 60); do
  OUT=$(./bench_full /home/orangepi/y11/yolo11n_640_head_real.rknn anchor640.npy stride640.npy 600 6 tiles_coco27.raw 27 2>/dev/null | grep -E "FPS")
  FPS=$(echo "$OUT" | sed 's/.*= //; s/ FPS.*//')
  BOX=$(./bench_full /home/orangepi/y11/yolo11n_640_head_real.rknn anchor640.npy stride640.npy 1 1 2>/dev/null | grep "NMS 后" | sed 's/.*→ NMS 后 //; s/（.*//')
  NOW=$(( $(date +%s) - T0 ))
  Z0=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
  Z1=$(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null)
  Z2=$(cat /sys/class/thermal/thermal_zone2/temp 2>/dev/null)
  NF=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null)
  GF=$(cat /sys/class/devfreq/fde60000.gpu/cur_freq 2>/dev/null)
  CF=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq 2>/dev/null)
  echo "$NOW,$Z0,$Z1,$Z2,$NF,$GF,$CF,$FPS,$BOX" >> $LOG
done
echo DONE >> $LOG
