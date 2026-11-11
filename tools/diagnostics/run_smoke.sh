#!/bin/bash
# Smoke/benchmark the deployment-faithful depth chain (480x270 + self-geometry
# + RealSense hole fill) on the server's free GPU. NOT a training run.
#
# Directory rule on this box: /root/bayes-tmp/xiaoyulong/yunhanzheng is mine,
# everything else under /root/bayes-tmp/xiaoyulong is shared. So:
#   - the shared isaacgym (xiaoyulong/isaacgym/python) is only put on PYTHONPATH,
#     never written to;
#   - WARPs kernel cache is redirected into my dir instead of /root/.cache/warp;
#   - TORCH_EXTENSIONS_DIR is left alone: torch already resolves gymtorch to the
#     machine-level /root/bayes-tmp/system_caches, which is the sanctioned spot
#     and rebuilding it there would break a working install.
set -e
MINE=/root/bayes-tmp/xiaoyulong/yunhanzheng
E=$MINE/envs/parkour
export PATH=$E/bin:$PATH
export PYTHONPATH=/root/bayes-tmp/xiaoyulong/isaacgym/python
export LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH
export WARP_CACHE_PATH=$MINE/cache/warp
mkdir -p "$WARP_CACHE_PATH"
REPO=$MINE/go2-flyball-parkour
cd "$REPO/legged_gym/legged_gym/scripts"

exec $E/bin/python play.py \
  --task go2_flat --exptid dualenc-flat-512 --checkpoint 3600 \
  --use_camera --num_envs "${NUM_ENVS:-280}" --headless \
  --device "${DEV:-cuda:1}" --max_steps "${STEPS:-100}" --no_video
