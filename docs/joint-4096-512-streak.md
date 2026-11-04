# E09 joint-4096-512-streak：课程规则大改 + 全程随机延迟

**日期**：2026-11-02  
**状态**：运行中  
**exptid**：`joint-4096-512-streak`  
**日志**：`legged_gym/logs/extreme-parkour/joint-4096-512-streak/run.log`  
**TB**：`legged_gym/logs/extreme-parkour/joint-4096-512-streak/tb/`

---

## 1. 关键改动汇总

### 1.1 课程规则（vs E06-E08）

| 参数 | E06-E08 值 | E09 值 | 说明 |
|---|---|---|---|
| `promote_streak` | 1（单次成功） | **3** | 连续 3 次成功才升级 |
| `demote_streak` | 1（单次失败） | **2** | 连续 2 次失败才降级 |
| `random_level_frac` | 0.0 | **0.2** | 20% env 随机采样 [0, max_level_seen] |
| `promote_goal_frac` | 0.375 | 0.375 | 不变 |
| `demote_goal_frac` | 0.25 | 0.25 | 不变 |

### 1.2 地形随机化

| 参数 | E06-E08 值 | E09 值 | 说明 |
|---|---|---|---|
| `flyball_hurdle_jitter_scale_with_difficulty` | True | **False** | 所有行都用完整 [1.30, 1.90] m 范围 |
| `flyball_hurdle_zero_prob` | 0.25 | 0.25 | 25% tile 无杆 |
| `flyball_single_hurdle_prob` | 0.0（不存在） | **0.333** | 25/25/50 分布（0/1/2 根杆） |
| `flyball_add_second_hurdle` | True | True | 不变 |

### 1.3 Action Delay

| 参数 | E06-E08 值 | E09 值 | 说明 |
|---|---|---|---|
| `randomize_action_latency` | False | **True** | per-env 每 episode 随机采样 |
| `action_latency_range` | N/A | **[0, 2]** | 0/20/40ms 三档均匀随机 |
| `action_delay` | True | **False** | 被 randomize 取代 |
| `action_delay_latch_mean_level` | 6.0 | **None** | 不再使用 latch |
| `--nodelay`（launch flag） | True | **移除** | 不再关闭延迟 |

### 1.4 训练规模

| 参数 | E06-E08 值 | E09 值 | 说明 |
|---|---|---|---|
| `num_envs` | 6144 | **4096** | 2^12，warp 对齐 |
| `num_depth_envs` | 280 | **512** | 2^9，depth 覆盖 0.64/tile |
| `max_iterations` | 10000 | **20000** | 更长训练 |
| `save_every` | 400 | **1000** | 20 个 checkpoint |
| 预计 iter 时间 | 4.2s | **~2.8s** | 快 33% |
| 预计总时间 | 11.7h | **~15.6h** | 更长但更充分 |

### 1.5 代码改动

| 文件 | 改动 |
|---|---|
| `legged_robot.py` | 新增 `success_streak`、`failure_streak`、`max_level_seen` buffer |
| `legged_robot.py` | `_update_terrain_curriculum` 加 streak 追踪逻辑 |
| `legged_robot.py` | `reset_idx` 加 `random_level_frac` 采样 |
| `flyball_wall.py` | 新增 `single_hurdle_prob` 参数 + `_use_second_hurdle` 逻辑 |
| `flyball_wall.py` | 修复 goal 布局：`add_second_hurdle` → `_use_second_hurdle`（3 处） |
| `terrain.py` | 传递 `single_hurdle_prob` 到 builder |
| `train_joint.py` | `--resume_ckpt` 改为 `load_optimizer=False` + `start_it=0` |

---

## 2. 启动配置

```bash
python -u legged_gym/scripts/train_joint.py \
  --task go2_teacher \
  --resume_ckpt .../flat_baseline/model_2400.pt \
  --use_camera --render_on_demand \
  --num_envs 4096 --num_depth_envs 512 \
  --exptid joint-4096-512-streak \
  --max_iterations 20000 --save_every 1000 \
  --headless --no_wandb --device cuda:0
```

**热启动来源**：`flat_baseline/model_2400.pt`（平地策略，2400 iter 训练）  
**Optimizer**：全新（不加载旧 optimizer 状态，lr 从 2e-4 起步）  
**TB 起点**：iter 0

---

## 3. 设计动机

### 3.1 为什么改课程规则

- **E06 崩盘教训**：latch 在 level 6 的阶跃切换导致分布偏移，策略时序失调
- **E07 瓶颈**：`--nodelay` 虽然稳定，但策略没学会处理延迟，部署忠实度差
- **单次升降级太敏感**：一次偶然成功/失败就改变难度，课程信号噪声大

### 3.2 为什么用全程随机延迟

- **社区主流做法**：per-env 随机 latency 是 legged gym 社区最常见方案
- **避免分布偏移**：从第一天就接触延迟，策略不会在后期突然不适应
- **部署忠实**：真实 Go2 有 ~20ms 通信延迟，训练时随机 [0,40]ms 覆盖真实范围

### 3.3 为什么 25/25/50 杆数分布

- **0 根杆（25%）**：保留简单 tile，让策略有机会练基础跑步
- **1 根杆（25%）**：新增中间难度，课程更平滑
- **2 根杆（50%）**：维持原有挑战，确保策略学会双杆跨越

---

## 4. 已知风险

### 4.1 LR 快速触底（已观察到）

**现象**：iter 0-60 内 lr 从 2e-4 降到 1e-5（adaptive KL 触发）  
**原因**：平地策略突然面对墙 + 随机延迟，KL 散度飙升  
**影响**：策略可能无法有效学习  
**应对**：如果持续，考虑：
1. 先 `--nodelay` 跑 2000 iter 让策略适应墙，再切随机延迟
2. 降低 `randomize_action_latency` 初始范围到 [0, 1]
3. 调高 `desired_kl` 让 adaptive lr 更宽容

### 4.2 课程可能爬升慢

- `promote_streak=3` 要求连续 3 次成功，比单次更难
- `random_level_frac=0.2` 让 20% env 留在低难度，拖慢 mean_level
- 预期 mean_level 爬升到 4.0 需要 1000-2000 iter（vs E07 的 ~400 iter）

---

## 5. 观察指标

| 指标 | 健康范围 | 警戒线 | 应对 |
|---|---|---|---|
| `LVL mean` | 持续上升 | 停滞 > 1000 iter | 检查 promote/demote 比例 |
| `lr` | > 1e-4 | < 5e-5 | 考虑调高 desired_kl |
| `act_err` | 持续下降 | 停滞或上升 | depth encoder 可能过拟合 |
| `ev` | > 0.5 | < 0.3 | value function 可能崩了 |
| `surr` | 小幅波动 | 持续负值 | 策略在退化 |
| `prefix tiles` | > 600/800 | < 400/800 | 地形覆盖不足 |

---

## 6. 后续计划

1. **监控前 500 iter**：确认 lr 是否恢复、mean_level 是否开始爬升
2. **如果 LR 持续触底**：切换到渐进式延迟方案（`action_latency_ramp_*`）
3. **如果课程爬升正常**：跑到 5000 iter 评估是否超过 E07 的 6.99 峰值
4. **reward 退火实验**：实现手工 shaping 奖励权重线性衰减（待开发）

---

## 附录：launch_streak.sh 最终版本

```bash
#!/usr/bin/env bash
# L1+L2 curriculum: promote_streak=3 + random_level_frac=0.2
# Fleet: 4096+512 (both 2^N, warp-aligned, depth coverage 0.64/tile)
# Warm-start from flat_baseline/model_2400.pt (平地策略).
# Changes vs launch_coltypes.sh:
#   - promote_streak=3 (need 3 consecutive successes to promote)
#   - random_level_frac=0.2 (20% envs randomly sample level in [0, max_level_seen])
#   - num_envs 6144→4096, num_depth_envs 280→512 (faster iter, better depth coverage)
#   - randomize_action_latency=True, action_latency_range=[0,2] (全程随机延迟)
# Both are in go2_teacher_config.py.
export LD_LIBRARY_PATH=/root/miniconda3/envs/parkour/lib:$LD_LIBRARY_PATH
export PATH=/root/miniconda3/envs/parkour/bin:$PATH
cd /root/autodl-tmp/go2-flyball-parkour/legged_gym
mkdir -p logs/extreme-parkour/joint-4096-512-streak
exec /root/miniconda3/envs/parkour/bin/python -u legged_gym/scripts/train_joint.py \
  --task go2_teacher \
  --resume_ckpt /root/autodl-tmp/go2-flyball-parkour/legged_gym/logs/extreme-parkour/flat_baseline/model_2400.pt \
  --use_camera --render_on_demand --num_envs 4096 --num_depth_envs 512 \
  --exptid joint-4096-512-streak --max_iterations 20000 --save_every 1000 \
  --headless --no_wandb --device cuda:0 \
  > logs/extreme-parkour/joint-4096-512-streak/run.log 2>&1
```
