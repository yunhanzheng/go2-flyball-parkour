#!/bin/bash
# Seeta box (RTX 4090D / 128 cores / 503 GB RAM / autodl-tmp 50 GB).
# Resume the dualenc teacher from model_5000, same training shape as the bayesdl run:
# 512 envs, cuda:0, camera on, headless, 200000-iteration budget, save_throttle
# 1x up to 5000 then 2x. The optimizer resume is the stock one (no extra knobs).
set -e

MINE=/root/autodl-tmp
R=$MINE/go2-flyball-parkour
E=/root/miniconda3/envs/parkour
LOG=$R/legged_gym/logs/extreme-parkour/dualenc-wall-warm3600-480p

export PATH=$E/bin:$PATH
export LD_LIBRARY_PATH=$E/lib
export PYTHONPATH=$MINE/isaacgym_pkg/isaacgym/python
# Keep the warp kernel cache inside our own tree instead of the shared system one.
export WARP_CACHE_PATH=$MINE/cache/warp
mkdir -p "$WARP_CACHE_PATH"

# autodl-tmp has ~16 GB free and a checkpoint is 112 MB, so a 200000-iteration run
# (97 saves at 2x) would run the disk out around iteration 110k. Keep the anchor and
# the three newest checkpoints -- a resume only ever needs the latest.
( while sleep 600; do
    ls -1t "$LOG"/model_*.pt 2>/dev/null \
      | grep -v '/model_5000\.pt$' \
      | tail -n +4 \
      | while read -r f; do rm -f "$f"; done
  done ) &

cd "$R/legged_gym/legged_gym/scripts"
exec $E/bin/python -u train.py \
  --task go2_teacher --exptid dualenc-wall-warm3600-480p --proj_name extreme-parkour \
  --use_camera --headless --no_wandb --device cuda:0 --num_envs 512 \
  --resume --resumeid dualenc-wall-warm3600-480p --checkpoint 5000
