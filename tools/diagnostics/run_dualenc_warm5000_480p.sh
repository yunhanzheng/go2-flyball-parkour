#!/bin/bash
# 续训：从自己 run 的 model_5000 接（200000 轮预算），跑到 GPU0。
#
# 为什么接 5000 而不是从平地 3600 重来：这一轮 09:33→21:44（iteration 1→9000）
# 已经被容器 OOM kill 掉，但 0→9000 都是在同一套新语义下跑出来的（dr_ramp_monotonic、
# 课程 level 0 冷启动），5000 之后没存盘（旧降频 5x=每 5000 轮一存），所以只能回到 5000，
# 丢 4000 轮 ≈ 5.2 h。现在 save_throttle 改成 ≥5000 后每 2000 轮一存，下次最坏丢 2.6 h。
# exptid 保持不变，所以日志/TB 仍落在 logs/extreme-parkour/dualenc-wall-warm3600-480p
# （tb_dualenc 下已有软链接）。
#
# 为什么 cuda:0：GPU1 被别人容器占了 5.8 GB（只剩 18.8 GB，512 env 需要 21.8 GB），
# GPU0 只剩 ~1.5 GB 占用。
#
# 课程侧语义：dr_ramp_monotonic=True（DR ceiling 只升不降），per-env U[0, ceiling]
# 抽样保持 main 原样。仍然不传 --delay：teacher 用相机只当实时传感器，延迟走
# action_delay_latch_mean_level=6.0 的舰队平均地形等级 latch（main 的 teacher 行为）。
set -e
MINE=/root/bayes-tmp/xiaoyulong/yunhanzheng
E=$MINE/envs/parkour
export PATH=$E/bin:$PATH
export PYTHONPATH=/root/bayes-tmp/xiaoyulong/isaacgym/python
export LD_LIBRARY_PATH=$E/lib:$LD_LIBRARY_PATH
# Keep warp's kernel cache inside my own dir instead of the shared /root/.cache/warp.
export WARP_CACHE_PATH=$MINE/cache/warp
mkdir -p "$WARP_CACHE_PATH"

cd "$MINE/go2-flyball-parkour/legged_gym/legged_gym/scripts"
exec $E/bin/python -u train.py \
  --task go2_teacher --exptid dualenc-wall-warm3600-480p --proj_name extreme-parkour \
  --use_camera --headless --no_wandb --device cuda:0 --num_envs 512 \
  --resume --resumeid dualenc-wall-warm3600-480p --checkpoint 5000
