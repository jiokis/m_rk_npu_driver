#!/bin/bash
cd /home/orangepi/cbench
./bench_full /home/orangepi/y11/yolo11n_640_head_real.rknn anchor640.npy stride640.npy 2000 6 tiles_coco27.raw 27 > /tmp/bench.out 2>&1 &
PID=$!
sleep 8
RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null)
THR=$(awk '/Threads/{print $2}' /proc/$PID/status 2>/dev/null)
A=$(awk '{print $14+$15}' /proc/$PID/stat 2>/dev/null)
sleep 6
B=$(awk '{print $14+$15}' /proc/$PID/stat 2>/dev/null)
wait $PID
{
echo "RSS=${RSS}KB THREADS=${THR} CPU_TICKS=$((B-A))/6s => $(( (B-A)*100/600 ))%"
grep -E "FPS|NMS 后" /tmp/bench.out
} > /home/orangepi/cbench/res.txt 2>&1
