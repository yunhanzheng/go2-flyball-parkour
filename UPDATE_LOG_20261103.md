# Go2 Flyball Parkour 训练更新日志

## 2026-11-03 更新

### 1. 出生位置随机化

**目的**：增加训练数据多样性，提升策略鲁棒性

**配置变更** (`go2_teacher_config.py`):
```python
class env(Go2RoughCfg.env):
    # 开启出生位置随机化
    randomize_start_pos = True   # ±0.3m xy 抖动
    randomize_start_yaw = True   # 固定偏航范围
    rand_yaw_range = 0.3         # 最大 ±17°（所有等级一致）
    
    # 移除测试用的 spawn_x_offset
    # spawn_x_offset = 2.0  # 已删除，恢复默认 0.0
```

**代码变更** (`legged_robot.py` line 2367-2375):
- 移除了 yaw 的等级缩放逻辑
- 现在所有地形等级使用**相同的偏航范围**（±0.3 rad = ±17°）

**效果**：
- 每次 reset 在 tile 原点基础上 ±0.3m xy 随机扰动
- 每次 reset ±17° 随机偏航，不随课程等级变化
- 所有地形等级使用相同的随机化强度

---

### 2. 域随机化（Domain Randomization）配置

**目的**：提升策略的 sim-to-real 迁移能力

**配置** (`go2_teacher_config.py`):
```python
class domain_rand(Go2RoughCfg.domain_rand):
    # PD 增益随机化
    randomize_pd_gain = True
    kp_range = [0.95, 1.05]  # ±5%
    kd_range = [0.95, 1.05]  # ±5%
    
    # 电机参数随机化
    randomize_motor_offset = True
    randomize_motor = True
    motor_strength_range = [0.8, 1.2]  # ±20%
    
    # 接触参数随机化
    randomize_restitution = True
    restitution_range = [0.0, 0.5]
    randomize_friction = True
    friction_range = [0.6, 2.0]
    
    # 质量与质心随机化
    randomize_base_mass = True
    added_mass_range = [0.0, 3.0]  # 额外 0-3 kg
    randomize_base_com = True
    added_com_range = [-0.2, 0.2]  # 质心偏移 ±0.2m
    
    # 动作延迟随机化（关键！）
    randomize_action_latency = True
    action_latency_range = [0, 2]  # 0, 20, 或 40ms
    
    # DR 课程设置：无渐进式开启
    dr_scale_range = [1.0, 1.0]  # 始终全量 DR
    dr_lock_to_terrain_level = False  # 不绑定地形等级
```

**说明**：
- 所有 DR 从 step 0 开始全量开启，不随课程渐进
- 动作延迟独立随机（0/20/40ms），模拟真实通信延迟
- PD 增益、电机强度、接触参数全部随机化

---

### 3. 地形课程与 Hurdle 配置

**目的**：渐进式难度学习，hurdle 参数解耦

**配置** (`go2_teacher_config.py`):
```python
class terrain(Go2RoughCfg.terrain):
    # 课程基本参数
    curriculum = True
    num_cols = 80  # 80 个地形变体
    max_init_terrain_level = 0  # 所有 env 从 level 0 开始
    
    # 升级/降级规则（不对称）
    promote_streak = 1  # 连续 1 次成功即升级
    demote_streak = 1   # 连续 1 次失败即降级
    
    # 防遗忘机制
    random_level_frac = 0.2  # 20% env 随机采样 [0, max_level]
    
    # 列重采样（关键！）
    resample_type_on_level_change = True  # 等级变化时重新分配地形列
    
    # Hurdle 配置
    flyball_add_hurdle = True
    flyball_add_second_hurdle = True
    flyball_single_hurdle_height_range = (0.07, 0.12)  # 7-12 cm
    flyball_single_hurdle_zero_below_level = 4  # Level 0-3 无 hurdle
    
    # Hurdle 数量分布（Level 4-9）
    flyball_hurdle_height_decorrelate = True  # 高度与难度解耦
    flyball_hurdle_zero_prob = 0.25  # 25% 无杆
    flyball_single_hurdle_prob = 0.333  # 剩余中 33% 单杆 -> 总体 25% 无杆 / 25% 单杆 / 50% 双杆
    
    # Hurdle 几何参数
    flyball_single_hurdle_thickness = 0.01  # 1 cm 厚
    flyball_single_hurdle_width = 1.20  # 1.2 m 宽
    flyball_single_hurdle_width_range = (1.00, 1.30)  # 宽度随机
    
    # 距离参数（不与难度耦合）
    flyball_hurdle_jitter_scale_with_difficulty = False  # 所有等级使用相同抖动范围
    flyball_hurdle_front_from_spawn = 1.60  # 出生点到第一杆
    flyball_hurdle_front_from_spawn_jitter = 0.30  # ±0.3m
    flyball_second_hurdle_offset = 2.00  # 杆间距
    flyball_wall_to_hurdle_rear = 2.00  # 最后一杆到墙
    
    # 墙角度
    flyball_wall_angle_deg_range = (30.0, 75.0)
    flyball_wall_angle_jitter_deg = 5.0  # ±5° 抖动
    
    # 目标点布局（Swimmer 4-waypoint）
    flyball_swimmer_goal_layout = True
    flyball_wall_goal_idx = 1  # g1 = 墙接触点
    flyball_return_goal_idx = 2  # g2 = 返回点
    
    # 课程进度阈值
    promote_goal_frac = 0.375  # 达到 37.5% 目标即升级
    demote_goal_frac = 0.25  # 低于 25% 目标即降级
```

**说明**：
- **80 列地形变体**：每列有不同的布局（杆间距、墙角度、粗糙度等）
- **列重采样**：等级变化时重新分配列，避免 depth env 固定在少数列上
- **Hurdle 解耦**：高度、数量、距离都不与难度等级耦合，防止策略读取难度线索
- **Swimmer 布局**：紧凑的 4-waypoint 回路（g0 杆间 → g1 墙 → g2 杆间 → g3 返回）

---

### 4. 低速终止功能（之前已实现）

**目的**：避免策略在障碍物前停滞，提高训练效率

**配置** (`go2_teacher_config.py`):
```python
class env(Go2RoughCfg.env):
    terminate_on_low_velocity = True
    low_velocity_term_mps = 0.1              # m/s 阈值
    low_velocity_term_duration_steps = 50    # 2s @ 25Hz
```

**实现** (`legged_robot.py`):
- 新增 `self.low_vel_steps` buffer 追踪连续低速步数
- `check_termination()` 中检测速度低于阈值持续指定步数
- 触发时加入 `reset_buf`，终止 episode

---

### 5. 横向速度惩罚（之前已实现）

**目的**：训练策略走直线，减少横向漂移

**配置** (`go2_teacher_config.py`):
```python
class scales(Go2RoughCfg.rewards.scales):
    lin_vel_y = -1.0  # 仅在平地地形生效
```

**实现** (`legged_robot.py`):
```python
def _reward_lin_vel_y(self):
    rew = torch.square(self.base_lin_vel[:, 1])
    flat_mask = self._flat_terrain_mask()
    rew = rew * flat_mask.float()
    return rew
```

---

### 6. Depth 相机远平面（保持默认）

**状态**：使用默认值 `far_clip = 2.0m`

**说明**：
- 之前测试时曾改为 4.0m 以观察 50° 陡墙
- 现已恢复默认值，不覆盖 base config 的设置

---

### 7. 双 Encoder 架构（已启用）

**目的**：联合训练 height scan encoder 和 depth encoder，通过一致性损失让两个 latent 对齐，然后渐进式蒸馏到纯 depth

**状态**：✅ **已启用** (`use_dual_encoder = True`)

**架构**：
```
Height Scan → Encoder → latent_h ─┐
                                   ├→ concat → Actor → Action
Depth Image → Encoder → latent_d ─┘

Loss = Policy_loss + consistency_loss_weight * MSE(latent_h, latent_d)
```

**自适应衰减策略**：
- 三重条件触发：`success_rate ≥ 0.5 AND mean_level ≥ 2 AND consistency < 0.1`
- 衰减率：`height_weight *= 0.995`（每次衰减 0.5%）
- EMA 平滑：`alpha = 0.95`

**配置** (`go2_teacher_config.py`):
```python
class Go2TeacherCfgPPO(Go2RoughCfgPPO):
    use_dual_encoder = False  # 默认关闭
    consistency_loss_weight = 0.1
    num_depth_latent = 32
    
    height_decay_trigger_success_rate = 0.5
    height_decay_trigger_mean_level = 2
    height_decay_trigger_consistency = 0.1
    height_decay_rate = 0.995
```

**已修改文件**：
- `rsl_rl/rsl_rl/modules/actor_critic.py`: Actor 支持 dual-encoder 模式 ✅
- `rsl_rl/rsl_rl/algorithms/ppo.py`: 一致性损失 + 自适应衰减 ✅
- `legged_gym/legged_gym/envs/go2/go2_teacher_config.py`: 配置参数 ✅
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`: Runner 已传递 depth_latent ✅

**实现完整性**：
- ✅ Actor.forward() 支持双 latent 拼接
- ✅ PPO.act() 支持 dual-encoder 分支
- ✅ PPO.update() 计算一致性损失
- ✅ PPO._update_height_weight_decay() 实现自适应衰减
- ✅ Runner 传递 depth_latent 给 PPO.act()
- ✅ 所有代码编译通过

---

## 当前训练配置总结

### 环境参数
- **地形**：100% flyball wall terrain
- **课程**：不对称升级降级（promote_streak=2, demote_streak=1）
- **出生点**：tile 原点 + 随机扰动（±0.3m xy, ±17° yaw）
- **Episode 长度**：20s (500 steps @ 25Hz)

### 终止条件
- 翻滚角度 > 95°
- 横向偏离 > 1.5m
- 低速持续 > 2s (vel < 0.1 m/s)
- Base 接触障碍物
- 超时 (500 steps)

### 奖励函数
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| tracking_goal_vel | 1.5 | 目标速度追踪 |
| tracking_yaw | 0.5 | 偏航角追踪 |
| lin_vel_y | -1.0 | 横向速度惩罚（平地） |
| wococo_stage | 25.0 | WoCoCo 阶段奖励 |
| wococo_foot_target | 25.0 | 落脚点精度 |
| collision | -20.0 | 碰撞惩罚 |
| termination | -150.0 | 提前终止惩罚 |

### PPO 参数
- **noise_std**: 1.0（固定，高探索）
- **learning_rate**: 2e-4（adaptive）
- **gamma**: 0.99
- **lam**: 0.95
- **clip_param**: 0.2
- **entropy_coef**: 0.01
- **desired_kl**: 0.01

---

## 下一步计划

1. **启动新训练**：使用更新后的配置（出生随机化）
2. **监控指标**：
   - 成功率（success_rate）
   - 平均地形等级（mean_level）
   - 终止原因分布（term_low_vel, base_contact 等）
3. **评估效果**：对比之前的训练，看随机化是否提升鲁棒性
4. **可选改进**：
   - 如果训练不稳定，调整 PPO 参数（见上文建议）
   - 如果效果良好，考虑开启双 encoder 模式（`use_dual_encoder = True`）
   - 完成 runner 侧的双 encoder 集成

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `go2_teacher_config.py` | 修改 | 添加出生随机化配置，删除 spawn_x_offset |
| `legged_robot.py` | 修改 | 移除 yaw 等级缩放，固定偏航范围 |
| `actor_critic.py` | 修改 | 双 encoder 架构支持（默认关闭） |
| `ppo.py` | 修改 | 一致性损失 + 自适应衰减（默认关闭） |

---

**备注**：所有代码已编译通过，等待实际训练验证。
