#!/bin/bash
# 平地 3600 -> 复杂地形 热启动（dual-encoder, 512 env）
#
# 与上一版 dualenc-wall-warm3600 的差别：
#   1. 随机化 / 延迟 / 奖励 / 存档间隔回退到 main 的 teacher（只留 dualenc 必需的键）；
#   2. depth 段照抄 main 的 student：original=(480,270) 真机分辨率 + self-geometry
#      + RealSense 补洞(mode 1, 8 iter) + latency [4,5]；补洞走 gpu 后端（与 cv2 逐位相同）。
#      存档间隔同时改成 1000 轮（480x270 的光线投射约 20 倍代价，一次存档+评测更贵）。
#
# 全部沿用 main 的 teacher 训练流程，不加任何自己的旋钮（不设 --lr_floor）。
# 注意：**不传 --delay**。这一版 teacher 用相机只是当实时传感器，不是蒸馏 student，
# 所以 config 里把它从 train.py 的蒸馏守卫里豁免（domain_rand.action_delay_distillation_guard
# = False），延迟回到 main teacher 的真实行为：舰队平均地形等级首次到 6.0 才永久 latch，
# 在那之前的每一局都是 0 延迟。守卫本身仍是 main 的原逻辑，student/distill 配置默认为 True。
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
  --use_camera --headless --no_wandb --device cuda:1 --num_envs 512 \
  --resume --resumeid dualenc-flat-512 --checkpoint 3600
