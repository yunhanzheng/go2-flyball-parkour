# Depth-RL：切到 teacher 的目标函数

日期：2026-11-13
分支：`rl-main-seed`（相对 `main` 的工作区改动）

## 1. 背景与决定

原来的 `learn_depth_rl` 只更新 `depth_actor + depth_actor_std`：

- 没有 value loss，advantage 全部来自**冻结的 teacher critic**；
- `estimator` 与 `depth_encoder` 全程冻结；
- 每轮 120 步（`depth_encoder.num_steps_per_env`），而 teacher 是 24 步。

结果是 300 轮只涨 9.7%，且 `hurdle_clearance` 的步数课程被 5 倍速触发。

本次改动：保留深度链路（`depth → encoder → latent → depth_actor`），把
**teacher 的目标集**（surrogate + value + estimator）搬过来；课程、地形随机化、
`min_terrain_level = 4`、`model_10000` warm start 全部不动。

不采用"直接切 `learn_RL`"：`learn_RL` 没有 depth latent 通路
（`ActorCritic.act` 不透传 latent、`mini_batch_generator` 不返回 latent），
而 `depth_mini_batch_generator` 已经为此写好了。

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `rsl_rl/storage/rollout_storage.py` | `Transition` / buffer 新增 `priv_states_true`（懒分配）；`depth_mini_batch_generator` 多 yield 一列 |
| `rsl_rl/algorithms/ppo.py` | 新增 `critic_optimizer`（`rl_critic_learning_rate`，默认 3e-4）；`act_depth` 存真值 priv；`update_depth_rl` 增加 clipped value loss + estimator loss，三个 optimiser 分别 step；返回四元组 |
| `rsl_rl/runners/on_policy_runner.py` | `learn_depth_rl` 的 `num_steps` 改用 `self.cfg["num_steps_per_env"]`（120→24）；解包四元组；TB 增加 `Loss_depth/rl_value`、`rl_estimator`；`save/load` 增加 `critic_optimizer_state_dict` |
| `legged_gym/scripts/train.py` | `_OPTIMIZER_ATTRS` 加 `critic_optimizer`；depth-RL 的 `--freeze` skip 集合加 critic/estimator；fingerprint 收窄为 `('depth_encoder',)` |

### 为什么必须单独存真值 priv

`cfg.env.num_privileged_obs = None`，所以 `RolloutStorage.privileged_observations`
是空的，`depth_mini_batch_generator` 里的 `critic_observations` 回退到
`observations` —— 而这份 obs 的 privileged 段已被 estimator 的输出覆盖。
因此 estimator 的监督标签、以及 critic 的特权输入，都只能从环境原始 obs
（`critic_obs`）另存一份。

维度核算（go2）：`n_proprio=53, n_scan=132, n_priv=9, n_priv_latent=29,
history_len=10 → num_obs=753`；`priv_start=185`，拼回真值后仍是 753。

## 3. 冻结清单（改动后）

> 下表是 2026-11-13 上午的版本。当天下午 P0-A 把 encoder 改成端到端重放（见 §6），
> 晚上 ARM-A 又把它显式冻结了一次 —— 两次都是**开关**，不是结构改动。

| 模块 | 11-13 上午 | 11-13 18:30 起（ARM-A） |
|---|---|---|
| `depth_actor` | 训练（surrogate） | 训练（surrogate） |
| `depth_actor.history_encoder` | 未单列 | 训练（独立 optimiser，lr 2e-4） |
| `depth_actor_std` | 训练（surrogate 的 log-prob 项） | **冻结 0.05**（`rl_freeze_std`） |
| `actor_critic.critic` | 训练（value loss，lr 3e-4） | 训练（不变） |
| `estimator` | 训练（真值回归，lr 1e-4） | 训练（不变） |
| `actor_critic.actor`（teacher） | **冻结**，latent 标签源 | **冻结**（不变） |
| `depth_encoder` | **冻结**（latent 被 detach） | **冻结**（`rl_encoder_learning_rate = 0`） |

`--freeze` 的指纹只覆盖 `depth_encoder`：`actor_critic` 一个模块里同时装着要冻结的
actor 和要训练的 critic，模块级指纹无法区分，这一点在代码注释里写明了。
depth-RL 路径下 `--freeze` 的“冻结”是靠把除 `depth_rl / critic / estimator /
hist_encoder` 之外的 optimiser lr 全部置 0 实现的。

## 4. 启动

```bash
python train.py --task go2_student --exptid student-depth-ppo \
    --resume --resumeid student --checkpoint 10000 \
    --device cuda:1 --headless --no_wandb \
    --num_envs 1024 --max_iterations 10000 --save_interval 100 \
    --use_camera --delay --depth_rl --freeze
```

- 24 步/轮：每轮 24576 样本、20 个 minibatch，与 teacher 同节奏；
- `hurdle_clearance` 回到 iter 2000 arm / 6000 满；
- 旧 ckpt 可直接加载：缺 `depth_actor_std`（走 `rl_init_std=0.05`）、
  缺 `critic_optimizer_state_dict`（新建 Adam）。

## 5. 观察指标

| 指标 | 期望 |
|---|---|
| `Loss_depth/rl_value` | 单调下降，不爆 |
| `Loss_depth/rl_estimator` | 下降并稳定在 DAgger 阶段的量级 |
| `Train/terrain_level_mean` | 脱离 4~5 的初始带 |
| `Train/mean_episode_length` | 越过 210 步（旧 run 的平台） |
| `Loss_depth/rl_kl_mean` | 不长期贴 1e-5 地板 |

地形几何已钉到最大难度（`geometry_at_max_difficulty=True`、jitter ±0.5），
warm start 后前 100~200 轮会先掉一段，这是分布迁移，不是训练坏了。

## 6. encoder 端到端（P0-A 已实现，本节旧结论作废）

本节旧版本写的是“不能在 PPO 图里解冻：`update_depth_rl` 用的是 rollout 时 `no_grad`
算出的 latent，反向到不了 encoder；要端到端需缓存 24 步 × 1024 envs × 480×270 的深度图
（约 12.7 GB）”。P0-A 换掉了这个前提：**不缓存深度图，只缓存 encoder 的输入**。

rollout 时把每步的 `depth_images / depth_props / GRU hidden-in` 一起存进 storage，
`update_depth_rl` 用 `depth_encoder.forward_step(...)` 重放同一批输入，latent 就在
PPO 的 autograd 图里，梯度直接流到 encoder。显存按 `rl_replay_chunk = 512` 分片累积，
实测峰值 1.0 GB —— 这是 P0-A 相对于“缓存 12.7 GB”的关键。

代价是 **BPTT(1)**：重放的 hidden-in 是 detached 的，信用分配截断在单帧，跨帧的时序
梯度不存在。所以这里的“端到端”是三层里最窄的那一层：

| 层次 | 含义 | 本项目 |
|---|---|---|
| A | encoder + actor 在同一个 autograd 图里 | **是**（P0-A） |
| B | 像素 → 动作全链路可微 | 否（raycast / hole_fill / resize 不可微） |
| C | 单阶段训练，不经过蒸馏 | 否（仍是 蒸馏 → fine-tune 两阶段） |

冻结 encoder 仍是一个合法配置（`rl_encoder_learning_rate = 0`）。ARM-A 就是这样跑的，
目的是把“actor 能不能在**固定**特征上学”单独问出来（见 §8）。

## 7. 已知坑

- `learn_vision` 仍是 120 步/轮（`depth_encoder.num_steps_per_env`），
  `learn_depth_rl` 已改为 runner 的 24 步；两条路径节奏不同是刻意的。
- `log_vision` 里的 `Policy/mean_noise_std` 打印的是 teacher 的 σ（≈0.6），
  depth-RL 的探索宽度看 `Policy/depth_action_std`。
- 迭代计数：老的 `student/model_10000.pt` 里 `iter` 存的是 0，所以从它 warm start
  的相对编号从 0 起；但 `load()` 现在已经恢复 `current_learning_iteration`，此后每个
  `model_<n>.pt` 的 `iter` 字段都等于 n，可以只按文件名排序。watchdog 的绝对账仍是
  `absolute = BASE_ITER + relative`（`TARGET_ITER = 20000` 表示再跑 10000 轮）。
  **由此推论**：`TARGET_ITER` 与“轮数”相差一个 `BASE_ITER`——写 `10600` 是 600 轮，
  不是 60 轮。ARM-A 的第一版脚本就在这里写大了 10 倍。

---

## 8. long4 的 217 轮：三个缺陷与一个测量陷阱（2026-11-13 晚）

`student-depth-rl-long4`（`rl_step_ratio 0.4`、`rl_encoder_anchor_coef 10.0`、
`rl_init_std 0.05`、自适应 lr）自然跑到 217 轮，读数的结论是**策略在劣化**。

### 8.1 先修指标：`Mean reward (total)` 在 episode 变短时会假性回升

logger 的写法是 `episode_sums.mean() / max_episode_length_s`（20 s），所以 episode 越短，
**所有按步累积的负项绝对值都跟着缩水**。`rew_termination` 恒为 `-0.1500`
（= −3.0 × 1 / 20）就是这条归一化的指纹。唯一可比的读数是**每步回报**：

```
rew_per_step = rew * 20 / ep_len
```

| it | rew | ep_len | rew/step | goal_idx | term_roll |
|---|---|---|---|---|---|
| 10 | −5.33 | 121 | **−0.88** | 0.045 | 0.668 |
| 60 | −9.48 | 152 | −1.25 | 0.055 | 0.667 |
| 100 | −19.65 | 175 | **−2.25** | 0.085 | 0.608 |
| 140 | −7.13 | 156 | −0.91 | 0.116 | 0.610 |
| 180 | −5.40 | 78 | −1.39 | 0.008 | 0.806 |
| 216 | −6.51 | 79 | **−1.66** | 0.032 | 0.772 |

按 `rew` 看是“−19.65 回升到 −6.51”，按每步回报看是单调劣化。**这个坑会让任何
用 `rew` 判断“练好了没”的结论全错。**

### 8.2 σ 不是常数，而且它会自己把 trust region 撑大

`depth_actor_std` 是 `nn.Parameter`，且在 `depth_rl_optimizer` 的 group 0 里。
`rl_entropy_coef = 0` 只是去掉了**显式**的熵奖励，surrogate 的 `ratio`
（`exp(logπ − logπ_old)`）对 σ 仍然有梯度。实测：

```
depth_actor_std  mean 0.05006 -> 0.06862 (it100)  -> 0.08760 (it200)   +75%
```

日志里的 `Mean action noise std (student)` 是 0.054 / 0.069 / 0.088，与权重完全一致
—— **这条监控没有撒谎，只是没人盯**。后果是结构性的：`kl_target = 6·ratio² = 0.96`
是按 σ=0.05 标定的，而 `KL = 6·(Δμ/σ)²`，σ 涨到 0.088 后同一 Δμ 只测出 0.32 倍的 KL，
于是**允许的步长自己扩大了约 3 倍**。

### 8.3 自适应 lr 在地板上，等于没有调度 —— 且它量错了东西

217 轮里 `RL learning rate` 有 125/197（63%）落在 1e-5 地板，同时 kl 中位数 2.43、
最大 24.3。当时读成“梯度尺度太大”。**ARM-A 的 14 轮把这条推翻了**：

在 `encoder lr = 0`、`σ` 冻在 0.05、`lr` 固定 1e-5 的条件下，实测
`kl = 0.0123 ~ 0.0150`，是 target 0.96 的 **1.5%**。同样是 1e-5 的 lr，kl 从 2~4
掉到 0.014，唯一变化是 **encoder 不动了**。

原因在 trust region 自己的定义：它比较的是“重放的 latent”和“rollout 时的 latent”。
encoder 权重一动，同一帧的 latent 就变，μ 跟着变 —— **这笔账被算在了策略头上**。
所以 long4 的因果链要反过来读：

```
encoder 漂移 → 重放 latent ≠ rollout latent → KL 虚高 (2~4)
   → 自适应 lr 一路降到 1e-5 地板 → actor 每轮只走 ~0.002 rad → 学不动
   → 策略劣化 → 访问新状态 → encoder 继续漂        （正反馈）
```

根治方式是**先冻 encoder**，而不是把 anchor 调大；anchor 只能减速，断不开这个环。
反过来，`kl` 在 encoder 冻结后用来看 lr 够不够：0.0147 远低于 0.96 说明 **lr 小了
约 8 倍**（`kl ∝ lr²`），有效的控制臂应该跑在 6e-5–8e-5。

### 8.4 anchor 只减缓，不钉住

coef 10 确实压住了 anchor loss（峰值 1.144 @it9 → 0.883 @it106），但 checkpoint 逐层 diff：

```
depth_encoder_state_dict   GLOBAL_REL 8.6e-02 (it100) -> 1.2e-01 (it200)
  output_mlp.0.weight      rel 1.39e-01 -> 1.92e-01
model_state_dict (critic)  critic.6.weight rel 2.26 -> 5.84
```

也就是说 anchor loss 是**访问状态的函数**，不能当作“权重没漂”的证据。这一条同时
否定 §5 的观察指标：判断 encoder 稳不稳只能做 checkpoint 的 tensor diff。

---

## 9. ARM-A 控制臂：只让 actor 动（2026-11-13 18:34 起）

long4 之后唯一能归因的实验是“把所有其它变量钉死”。改动只有三处配置：

| 参数 | 值 | 作用 |
|---|---|---|
| `rl_freeze_std` | `True` | σ 成为常数 0.05：`requires_grad_(False)`，并**从 optimiser group 0 里移出** |
| `rl_adaptive_lr` | `False` | lr 钉在 1e-5（等于它 63% 的时间已经落在的值），kl 诊断照常记录 |
| `rl_encoder_learning_rate` | `0.0` | encoder 冻结在 DAgger 解上 |

`ppo.py` 侧新增两个开关，`rl_adaptive_lr=False` 时仍然累加 `kl_sum/kl_n`，否则会丢掉
唯一能证明“actor 到底走了多远”的指标。

**读法**（`rew*20/ep_len`，不看 `rew`）：

1. σ 在日志和 `model_<n>.pt` diff 里都必须是 0.050；
2. `term_roll` 回落到 it10 的 0.65 一带 —— 这是 σ 漂移的直接代价，也是最干净的预测；
3. 每步回报停止下滑；
4. `max_goal_idx` 抬过蒸馏基线 0.052–0.069。

若 1 成立而 3/4 不成立，则“在固定特征上 actor 也学不动”，瓶颈在蒸馏缺口而不是
PPO 回路 —— 那时才值得去改 terrain 或 reward。

### 9.1 前 14 轮实际读数（18:34–18:41）

| 项 | 值 |
|---|---|
| σ | **14/14 轮都是 0.050** —— 冻结生效 |
| kl | 0.0123 ~ 0.0150（target 0.96，即 1.5%） |
| lr | 固定 1.00e-05 |
| ep_len | 36.6 → 132.2（前 5 轮是从随机 episode 位置起的 warm-up，不是策略） |
| goal_idx | 0 → **0.0695** |
| term_roll | 0.87（warm-up）→ **0.70**，回到 long4 it10 的 0.668 一带 |
| encoder grad norm | 18.5（梯度在算但 lr=0，不生效） |

判据 1、2 初步成立。但 kl 只有 target 的 1.5% 说明这个 lr 下 60 轮只能走 long4
的 1/8 步长，所以这次运行的价值主要在“验冻”，而不是“验学”。要真正回答
“actor 能不能学”，下一轮应该把 lr 提到 6e-5–8e-5 重跑（或者直接把 `rl_adaptive_lr`
恢复成 `True`：encoder 冻结后 kl 是可测量的，调度器不再被谎报的 KL 骗）。

---

## 10. go2_student_maxdr：maxdr 任务 + student ckpt 热启动 + 纯 RL（2026-11-15）

### 10.1 决定链

1. 原方案是“完全从零端到端视觉 RL”（随机初始化、不 resume）；
2. 用户纠正：**不是从零，而是从 `student/model_10000.pt` 热启动**——目标函数仍是
   纯 RL（无 anchor、无 L2、无 BC），只改初始化；
3. ARM-A 微调档（冻 encoder + σ=0.05 + lr 钉 1e-5）作废：速度极慢不可用，且它是
   “验冻”臂不是“验学”臂（§9.1）。

### 10.2 配置（`envs/go2/go2_student_maxdr_config.py`，新 task `go2_student_maxdr`）

| 块 | 来源 |
|---|---|
| terrain / domain_rand / rewards | `Go2TeacherMaxDrCfg` 原样继承：curriculum=False、rows 0-9 均匀、height 0.06、max_error 2.0、dr_scale=1.0、20 ms latch、11-14 调权奖励（termination -60、tracking_goal_vel 2.2 等） |
| depth 相机 | `Go2StudentCfg.depth` 原样：480×270、hole_fill、`override_terrain_{mix,grid}_for_camera=False`（保住 maxdr 地形不被相机路径重写） |
| algorithm / policy | teacher 原样继承：lr 2e-4 adaptive、desired_kl 0.01、entropy 0.01、σ 1.0、epochs 5、minibatches 4、[512,256,128] |
| env.terminate_on_yaw_tracking | False（student 继承，深度看不到航向） |
| env.terminate_on_lateral_deviation | **True / 1.5 m，保留**——main teacher 也有（base 默认 False，teacher 显式开启，student 继承） |

rl_* 与 ARM-A 的差异：

| 参数 | 值 | 说明 |
|---|---|---|
| rl_encoder_learning_rate | 3e-4 | 可训（ARM-A=0） |
| rl_encoder_anchor_coef | 0 | 无 teacher anchor |
| rl_hist_via_surrogate | True | **新旗标**：history_encoder 走 surrogate，L2 段全跳过（§10.5） |
| rl_init_std / rl_max_noise_std | 1.0 / 1.0 | ckpt 无 depth_actor_std，这就是初始 σ；可训 |
| rl_freeze_std / rl_entropy_coef | False / 0.01 | teacher 探索配方 |
| rl_critic_learning_rate | **3e-5** | 见 §10.4 |
| rl_learning_rate | 2e-4 adaptive [1e-5, 1e-2]，desired_kl 0.01，step_ratio=None | σ≈1 直接用 desired_kl，不用 0.4 ratio |

### 10.3 ckpt 事实（实测 `student/model_10000.pt`）

- 有：`model_state_dict`（teacher actor + DAgger 期 critic）、`depth_encoder`、
  `depth_actor`（含已训 history_encoder + 冻 priv_encoder）、`estimator`；
- 无：`depth_actor_std`（σ 从 rl_init_std=1.0 起）、一切 RL optimizer 状态（新建 Adam）；
- `iter=0` → 计数从 0 起，`max_iterations=10000` 就是新跑 10000 轮。

### 10.4 critic warm-start 发散（已知坑，预防性降档）

ckpt 的 critic 按 **main student 环境**的回报训成；maxdr（冻结全难度 + 调权奖励）是
另一个回报分布。ARM-B long6 已验证：3e-4 × 20 步/轮过冲发散（rl_value 0.3→2.2e9），
`update_depth_rl` 的 advantage 不归一化，发散 value 直接污染 actor。
**预防性降到 3e-5**（验证过的修复值）。早期预警：`Loss_depth/rl_value` 在 it30 内
单调爬升即干预；it10 左右见顶回落到 2~3 为正常。

### 10.5 ppo.py 旗标（唯一算法改动）

`rl_hist_via_surrogate=True` → `hist_ids=空集` → history_encoder 留在
`depth_rl_optimizer` group 0 走 surrogate；`depth_hist_encoder_optimizer=None` →
`update_depth_rl` 的 zero_grad / L2 backward / clip / step 四处守卫全跳过。
默认 False，现有 run 零影响。priv_encoder 保持冻结但不再被用作目标。
热启动后 history_encoder 带着蒸馏权重进 surrogate，目标函数仍是纯 RL。

### 10.6 启动

```bash
python train.py --task go2_student_maxdr --exptid <id> --proj_name extreme-parkour \
    --headless --no_wandb --device cuda:0 --num_envs <sweep 胜者> \
    --max_iterations 10000 --save_interval 100 \
    --resume --resumeid student --checkpoint 10000 \
    --freeze --depth_rl --use_camera --delay
```

### 10.7 监控

| 指标 | 看什么 |
|---|---|
| `Loss_depth/rl_value` | it30 内单调爬升 = critic 发散（§10.4） |
| `RL learning rate` | 钉 1e-5 地板 = KL 失控；encoder 可训时 §8.3 的漂移污染机制仍在 |
| `Policy/depth_action_std` | σ 从 1.0 应收敛下行；持续上涨警惕 trust region 自扩（§8.2） |
| `rew*20/ep_len` | 唯一可比的回报口径（§8.1） |
| `per_level/L0..L9` | 低难度行是否先起来 |

### 10.8 num_envs sweep（bayesdl 实测，2026-11-15 10:55–11:26）

GPU0 独占（ sweep 期间另一租户已退，baseline 0 MiB），每档 8 轮（前 2 轮 warm-up），
地形 800 块首烤 ~14.5 min，后续档共享缓存。脚本 `sweep_student_maxdr_numenv.sh`，
原始日志 `sweep_maxdr_out/`。稳态口径（p50≈max，非瞬时尖峰）：

| NE | 稳态显存 | 稳态 s/轮 | samples/s（最优轮） | 备注 |
|---|---|---|---|---|
| 256 | 11481 MiB | ~3.5–4.3（噪声大） | ~1750 | |
| 512 | 13603 MiB | ~6.1 | ~2000 | |
| **1024** | **18967 MiB** | **~10.2–10.5** | **~2400** | **推荐：5.6 GB 余量** |
| 1536 | 24015 MiB | ~14.2 | ~2600 | 距上限仅 550 MiB，不可用 |
| 2048 | 21537 MiB | ~18.1–18.4 | ~2700 | 吞吐最高，余量 3 GB |

两个反直觉点：

1. **显存非单调**：1536（24.0 GB）> 2048（21.5 GB）。稳态值，不是测量噪声——
   PyTorch 缓存分配器对 1536 的 minibatch/replay-chunk 形状取整/碎片更差。
   1536 是“最坏档位”：吞吐只比 1024 高 8%，显存却贴顶。
2. **吞吐在 1024 基本到平台**：1024→2048 只 +12% samples/s（2400→2700），
   与 teacher 的 4096 平台（§9.9 之后的 teacher sweep）同构：渲染/物理成本随 env
   线性涨，加 env 不赚。每轮样本数 1024×24=24576，与旧 student run 一致。

**结论：默认 1024**（89% 的峰值吞吐 + 5.6 GB 抗租户余量 + 与旧 student run 同 batch）；
若 GPU0 确定独占且要极致吞吐，2048 可选（+12%，但 50 h 长跑中任何 ≥3 GB 的租户
回涌都会 OOM）。**避开 1536**。10000 轮 ETA：1024 ≈ 29 h，2048 ≈ 51 h。

sweep 同时验证了新代码路径端到端可用：resume 加载、σ=1.0 覆盖、hist via surrogate、
critic lr 3e-5、anchor 0 全部在日志确认，5 档均 rc=0。



