#!/bin/bash
# Prune superseded checkpoints / terrain caches under yunhanzheng only.
# Policy: keep the newest checkpoint per run plus the warm-start anchors.
set -u
LOG=/root/bayes-tmp/xiaoyulong/yunhanzheng/go2-flyball-parkour/legged_gym/logs
CL=$LOG/terrain_cache

echo "===== 1) terrain_cache: drop superseded 1.3 GB r10xc80 caches ====="
find "$CL" -maxdepth 1 -name 'r10xc80_hs0.01_*.pkl' \
     ! -name 'r10xc80_hs0.01_e5c15e496376.pkl' -print -delete

echo "===== 2) finished runs: keep newest (+ anchors) ====="
prune() {
  d=$1; shift
  keep=" $* "
  [ -d "$d" ] || { echo "  (missing $d)"; return; }
  for f in "$d"/model_*.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    case "$keep" in
      *" $b "*) echo "  keep $b" ;;
      *) echo "  rm   $b"; rm -f "$f" ;;
    esac
  done
}
prune "$LOG/extreme-parkour/dualenc-flat-512"       model_3600.pt model_4800.pt
prune "$LOG/extreme-parkour/upstream-flat"          model_7500.pt model_10000.pt
prune "$LOG/extreme-parkour/dualenc-wall-mono-dr0"  model_7500.pt
prune "$LOG/extreme-parkour/dualenc-wall-512"       model_5000.pt
prune "$LOG/extreme-parkour/dualenc-wall-warm3600"  model_3000.pt
# dualenc-wall-warm3600-480p is the LIVE run: untouched.

echo "===== result ====="
df -h /root/bayes-tmp | tail -1
du -sh "$LOG"
