#!/bin/bash
# Sweep num_envs on the free GPU and report throughput + peak VRAM, so the
# starting point is measured rather than guessed. Reuses run_smoke.sh (same
# task/checkpoint/depth chain), only NUM_ENVS and STEPS vary.
MINE=/root/bayes-tmp/xiaoyulong/yunhanzheng
cd "$MINE"
DEV=${DEV:-cuda:1}
GPU=${DEV#cuda:}
STEPS=${STEPS:-60}
printf "%-7s %-5s %-11s %-11s %-9s %s\n" num_envs rc peakVRAM steps_per_s env_steps_per_s note
for N in "$@"; do
  STEPS=$STEPS NUM_ENVS=$N DEV=$DEV ./run_smoke.sh > "sweep_${N}.log" 2>&1 &
  pid=$!
  peak=0
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 0.4
  done
  wait $pid; rc=$?
  sps=$(grep -oE "[0-9.]+ steps/s" "sweep_${N}.log" | tail -1 | grep -oE "^[0-9.]+")
  if grep -q "out of memory\|CUDA error" "sweep_${N}.log"; then note=OOM; else note=ok; fi
  if [ -z "$sps" ]; then sps=0; fi
  printf "%-7s %-5s %-11s %-11s %-9s %s\n" "$N" "$rc" "${peak}MiB" "$sps" \
    "$(awk -v a="$sps" -v b="$N" 'BEGIN{printf "%.0f", a*b}')" "$note"
done
