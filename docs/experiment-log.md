# 实验记录

任务：`go2_teacher`（Go2 + flyball 墙，PPO，`if_depth=False` 的 teacher 路径）
日志根目录：`legged_gym/logs/extreme-parkour/<exptid>/`
训练日志（含每轮全部指标）：`legged_gym/logs/wandb/<run-id>/files/output.log`

---

## 0. 共同基线

| 项 | 值 |
|---|---|
| 仿真 | Isaac Gym Preview 4 + legged_gym + rsl_rl，`parkour` env（Python 3.8.20 / torch 2.0.1+cu118） |
| env 数 / 每轮步数 | 4096 / 24 |
| 地形 | `num_rows = 10`，level 0~9，`wall = lerp(30°, 75°, level/9)` ±5° jitter |
| 回合长度 | `episode_length_s = 20`（1000 步 @ dt 0.02） |
| 晋级 / 降级门槛 | `promote_goal_frac = 0.375` / `demote_goal_frac = 0.25` |
| 初始地形 | `max_init_terrain_level = 5`（warm start 后从 0~5 级随机，不继承课程） |
| 算法 | PPO，`learning_rate 2e-4`，`schedule = adaptive`，`min_learning_rate 1e-5` |
| 终止（真） | `roll > 95°`、`pitch > 1.5`、`z < −0.25`、lateral > 1.5 m、`base`/`Head` 接触、超时、reach_goal、plant 失败项 |
| 奖励终止价 | `termination` scale −150 × dt 0.02 = **−3.0 / 次** |

---

## 1. 时间线总览

| # | exptid | 日期 | 轮次 | 关键改动 | 课程峰值 | 结局 |
|---|---|---|---|---|---|---|
| E01 | `flat_baseline` | 10-26 | 2400 | 平地，无墙 | — | 基线参考 |
| E02 | `teacher` | 10-26 | 3200 | 早期 teacher ckpt | — | 存档 |
| E03 | `student` | 10-26 | 10000 | 学生蒸馏 ckpt | — | 存档 |
| E04 | `d1` | 10-26~27 | 12500 | 早期探索 | — | 存档 |
| E05 | `my-teacher` | 10-27 | 3000 | 早期探索 | — | 存档 |
| **E06** | `new-run` | 10-27 | 10000 | 延迟 latch@6 + 豁免@6 + DR 满额@6 | 6.00 @ 5124 | **崩盘** |
| **E07** | `new-run-v2` | 10-28 | 4000 | 豁免关 + `--nodelay` | **6.99 @ 3871** | 卡在 6.9 |
| **E08** | `new-run-v3` | 10-29 | 4000 | 上述 + DR 满额推到 9 | 进行中 | 进行中 |

E01~E05 为本次分析之前的历史 run，未纳入复盘。

---

## 2. E06 new-run：三重难度在 6 级叠加导致崩盘

**起跑**：从 `d1` 或初始 cold start，`max_iterations 10000`，默认（延迟 latch + 姿态豁免 + DR 满额全部在 6 级触发）。

**关键事件**：iter 5124 `terrain_level` 首次触及 **6.001**，5125 同一轮内两个 latch 同时生效：

1. `action_delay_latch_mean_level = 6.0` → 20 ms 延迟永久锁存
2. `enable_wococo_armed_term_exempt = True` + `ramp_steps = 0` → armed 后取消 roll/pitch/base_contact 三条终止、`grace_steps = None` 无限宽限

| iter | base_contact | timeout | ep_len | reward |
|---|---|---|---|---|
| 5120 | 0.178 | 0.177 | 676 | +0.03 |
| 5125 | 0.248 | 0.120 | 633 | −3.02 |
| 5150 | 0.115 | **0.422** | 682 | **−48.57** |
| 5175 | 0.064 | 0.557 | 785 | −65.29 |
| 5300 | 0.018 | **0.914** | 952 | −18.28 |
| 9500（末） | 0.000 | **0.940** | 972 | −10.22 |

**机制**：

- 豁免把"摔倒 −3"变成"躺 20 秒超时 −45~−65"（`_reward_timeout` scale = 0，但期间所有正奖励归零）
- `wococo_armed` 只置位不清零 → 摔倒后退回重跑拿不到 tracking 奖励，只能吃 `contact_pts` 的 `n_wrong` 惩罚
- 仓库内无任何 get-up / self-right 奖励项，策略没有"爬起来"的梯度
- lr 被 per-minibatch KL 自适应压到 `min_learning_rate = 1e-5`
- `vfl` 冲到 1.85，value 崩塌后 4500 轮再没恢复；`s1_hold_ok` 从 0.303 → 0.000

**结论**：豁免本身在 60~75° 贴坡 plant 时有物理意义（`roll_target = −wall_angle`），但"无限宽限 + 取消三条终止 + 一步到位"的组合只是惩罚放大器，不能教会新技能。

---

## 3. E07 new-run-v2：6 级通过，卡在 6.9

**改动**（两处，均为"拆刀"）：

```python
enable_wococo_armed_term_exempt = False        # L405 附近
```

**起跑**：

```bash
python train.py --task go2_teacher --exptid new-run-v2 --proj_name extreme-parkour \
  --device cuda:0 --headless --resume --resumeid new-run --checkpoint 5000 \
  --nodelay --max_iterations 4000
```

**结果**（4000 轮，3.9 s/轮，总 14066 s，末权重 `model_3800`）：

| iter | level | reach | base_contact | stage1_ground | s1_hold | vfl | reward |
|---|---|---|---|---|---|---|---|
| 0 | 1.66 | 0.000 | 0.000 | 0.000 | 0.000 | 0.23 | −2.43 |
| 200 | 5.09 | 0.530 | 0.207 | 0.142 | 0.309 | 0.14 | +2.43 |
| 400 | 6.27 | 0.393 | 0.200 | 0.144 | 0.244 | 0.08 | −0.80 |
| 1800 | 6.43 | 0.582 | 0.138 | 0.053 | 0.393 | 0.07 | +1.36 |
| 2400 | 6.83 | 0.530 | 0.086 | 0.101 | 0.369 | 0.06 | +3.75 |
| 3400 | 6.84 | 0.546 | 0.064 | 0.114 | 0.355 | 0.05 | +5.67 |
| 3800 | 6.92 | **0.595** | 0.057 | 0.153 | 0.383 | 0.07 | **+5.32** |

- `terrain_level` 峰值 **6.9902 @ 3871**，最后 1600 轮在 6.8~6.99 震荡
- **无崩盘**：`vfl` 0.05~0.09，`reward` 首次稳定转正
- 两把刀确认未开：`action_delay/latched = 0`、`armed_term_exempt_armed = 0`

**分级评测**（`model_3800`，64 envs，clean 145 ep / full-DR 152 ep）：

| level | 墙 | reach（clean） | reach（full DR） |
|---|---|---|---|
| 5 | 54.6° | 0.683 | 0.725 |
| 6 | 58.3° | 0.882 | 0.767 |
| 7 | 66.3° | 0.757 | **0.450** |
| 8 | 69.0° | **0.909** | **0.238** |

**结论**：能力已到 69°（clean 91~94%），卡点不是 plant 技巧、不是 roll 对齐、不是 Head 撞墙，而是 `dr_ramp_full_at_mean_level = 6.0` —— 60° 墙时就按最难物理档训练。7 级 0.450 仅略高于 0.375 门槛，于是 6↔7 反复震荡，8 级（0.238）进不去。

---

## 4. E08 new-run-v3：DR 满额点推到 9 级（进行中）

**改动**：

```diff
-        dr_ramp_full_at_mean_level = 6.0
+        dr_ramp_full_at_mean_level = 9.0
```

| level | ceiling（6.0） | ceiling（9.0） |
|---|---|---|
| 6 | 1.00 | 0.67 |
| 7 | 1.00 | 0.78 |
| 8 | 1.00 | 0.89 |
| 9 | 1.00 | 1.00 |

**起跑**：

```bash
python train.py --task go2_teacher --exptid new-run-v3 --proj_name extreme-parkour \
  --device cuda:0 --headless --resume --resumeid new-run-v2 --checkpoint 3800 \
  --nodelay --max_iterations 4000
```

**启动校验**（iter 0~3）：

| 项 | 实测 | 期望 |
|---|---|---|
| `terrain_level` | 1.6638 | warm start 从 0~5 级随机 |
| `dr_scale_max` | 0.1859 | 1.6638 / 9.0 = 0.1849 ✅ |
| `dr_scale_mean` | 0.0934 | ceiling / 2 ✅ |
| `action_delay/latched` | 0.0000 | `--nodelay` 生效 ✅ |
| ETA | ~219 min / 4000 轮 | — |

方案与验收标准见 [new-run-v3-plan.md](new-run-v3-plan.md)。

---

## 5. 横向对比

| 指标 | E06 new-run | E07 new-run-v2 | E08 new-run-v3 |
|---|---|---|---|
| 课程峰值 | 6.001 @ 5124 | 6.990 @ 3871 | 待定 |
| 末段 reach_goal | — （崩） | 0.595 | 待定 |
| 末段 reward | −10.22 | +5.32 | 待定 |
| 末段 vfl | 1.85 | 0.07 | 待定 |
| 20 ms 延迟 | 开（6 级锁存） | 关 | 关 |
| 姿态豁免 | 开（6 级全开） | 关 | 关 |
| DR 满额点 | 6 级 | 6 级 | **9 级** |
| 8 级 clean 成功率 | — | 0.909 | 待定 |

---

## 6. 未解问题

| # | 问题 | 证据 | 计划 |
|---|---|---|---|
| 1 | 20 ms 动作延迟未进训练 | E07/E08 均 `--nodelay` | 9 级稳定后开 stage2：`--delay` + 从 4 级重爬，或把 latch 改到 8.0 |
| 2 | 8~9 级 roll 窗口过窄 | `roll_target −75°` vs `roll_term 95°`，余量 20° | 若 9 级批量 `term_roll`，提到 110° |
| 3 | `s0_roll_ok` 长期 0.08~0.14 | 一直不侧倾 | clean 下 8 级仍 91%，暂不动；若满 DR 下 `base_contact` 成主因再加强 `wococo_base_orient` |
| 4 | 5 级 `stage1_ground = 0.25`（满 DR） | 低等级反而最高 | 与 Plan B 第 1 条一起调 `wococo_stage1_ground_max_z` |
| 5 | `only_positive_rewards` 已关 | 未提交的改动 | 与本次改动一起提交，避免 diff 混淆 |

---

## 7. 记录规范（后续每轮必填）

1. 起跑命令全文 + exptid + wandb run id
2. 相对上一轮的**唯一变量**（一次只改一处）
3. 每 200 轮的 `terrain_level` / `term_reach_goal` / `term_base_contact` / `vfl` / `reward`
4. 课程峰值与到达轮次；是否出现 `vfl > 1` 或 `lr = 1e-5`
5. 末权重编号 + 一组分级评测（clean 与 full-DR 各一次）
6. 一句话结论：这轮证明了什么、下一轮该改什么

---

## 8. 工具链提速尝试（已全部回退，无净改动）

2026-10-29 在 `train.py` / `on_policy_runner.py` / `legged_robot.py` 上试了三条零语义改动，实测无收益，**当日已逐条回退，三个文件与 HEAD 逐字一致**（`git status` 只剩 `go2_teacher_config.py`）。三条都不触碰 rollout、PPO 更新、课程门槛与 DR：

| # | 改动 | 位置 | 影响 |
|---|---|---|---|
| 1 | `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 默认 4 | `scripts/train.py` 头部（早于 isaacgym/torch import） | 容器只有 18 核，宽线程池只是互抢；`OMP_NUM_THREADS=... python train.py` 可覆盖 |
| 2 | fleet-mean 地形等级缓存 | `envs/base/legged_robot.py`：`_refresh_fleet_mean_level` / `_fleet_mean_level_now`，在 `_update_terrain_curriculum` 与 `set_terrain_levels` 末尾刷新 | 每控制步少 2 次 `terrain_levels.mean().item()` 的 GPU→CPU 同步（stage-anneal、plant-fail 解锁、soft-latch 解锁都读它）。等价性：等级只在两处 update 里变，所以缓存值与实时值相同 |
| 3 | 日志降频到每 5 轮 | `rsl_rl/runners/on_policy_runner.py`：`EP_LOG_EVERY`（默认 5，=1 恢复逐轮） | 每轮少 ~60 次 ep_info 主机同步 + 一次 wandb 上传；`tot_time` / `tot_timesteps` / ETA 仍每轮累加，数值精确 |

**实测**（exptid `_perfcheck`，20 轮，6144 envs，从 `new-run-v2/model_3800` warm start）：

| | v3 后 1000 轮 | 本次 |
|---|---|---|
| Iteration time | median 3.43 s（p10 3.34 / p90 3.54） | 3.39 / 3.44 / 3.20 s（稳态 3 样本） |
| collection | median 3.176 s | 3.137 / 3.183 / 2.951 s |
| learning | 0.255 s | 0.253 s（PPO 未动，符合预期） |

结论：**收益 ≤3%，落在 v3 自身的迭代波动范围内**，不构成加速。collection 占 88.6% 的结构没变；真正的杠杆仍是 env 数（改 batch）与地形网格（改多样性），两者都会改变实验条件，本轮不动。

**回退理由**：收益落在噪声里，留着只增加代码面；且日志降频会把 wandb / stdout 变成 1/5 采样，后续实验的曲线密度反而与 E01~E08 不可比。回退后恢复逐轮记录。

**留下有用的基线**：体检跑（exptid `_perfcheck`，20 轮，6144 envs）顺带确认了重启后的健康值 —— `terrain_level 1.6639`、`dr_scale_max 0.2788 ≈ 1.6639/6.0`（DR 满额点确为 6 级）、`stage_anneal_frac 1.0`、`plant_fail_term_armed 0`、`armed_term_exempt 0`，无异常退出。这三个数就是 v4 起跑时的对照基线。

---

## 9. stage3 warm start 构造（`scripts/make_stage3_ckpt.py`）

2026-10-29 新增离线脚本，把 DAgger 学生的 checkpoint 转成 stage3（单阶段 depth PPO）的合法起点。**只读写 `logs/`，不占 GPU，不触碰正在跑的 `v2-student` 蒸馏。**

### 9.1 为什么需要它

`go2_depth` 的 `depth_encoder.train_with_ppo = True`，于是 `OnPolicyRunner` 建出的 `depth_actor = None`（[on_policy_runner.py:103](file:///root/autodl-tmp/go2-flyball-parkour/rsl_rl/rsl_rl/runners/on_policy_runner.py#L103) `None if self.train_depth_ppo else deepcopy(...)`）。`load()` 里那段 `if getattr(self.alg, 'depth_actor', None) is not None:`（L760-766）因此整段跳过 —— 直接从学生目录 resume 会得到：

| 模块 | 实际加载到的 | 应该是 |
|---|---|---|
| actor | **老师的** actor（对着 32 维真实 scan latent 训练） | 学生的 `depth_actor` |
| encoder | 学生的 CNN（输出 32 维深度 latent） | 同左 ✔ |
| critic / std | 老师的 | 同左 ✔ |

即“策略认的是一个它从没见过的 latent 坐标系”，等价于把 actor 随机初始化。

### 9.2 做法：键重映射，不改任何训练代码

学生的可部署策略存在 `depth_actor_state_dict` 里，而 `depth_actor = deepcopy(actor_critic.actor)`，**同一个类、同一套 key**，所以只需覆盖：

```
model_state_dict['actor.<k>'] <- depth_actor_state_dict['<k>']
```

| 内容 | 来源 | 理由 |
|---|---|---|
| actor | 学生 `depth_actor` | 部署策略；latent 坐标与 encoder 在 BC 阶段联合训出，天然一致 |
| critic | 老师 | 753 维特权 obs，不对称 actor-critic 必须保留上帝视角 |
| std | 老师 | `policy.continue_from_last_std = True`（[legged_robot_config.py:617](file:///root/autodl-tmp/go2-flyball-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py#L617) / [task_registry.py:165](file:///root/autodl-tmp/go2-flyball-parkour/legged_gym/legged_gym/utils/task_registry.py#L165)），所以 12 维 σ 会被沿用 |
| encoder | 学生 | `load()` 自动读 `depth_encoder_state_dict` |
| estimator | 学生 | 显式状态估计器，部署同款 |
| optimizer | **丢弃** | 学生是 1 组（35 参数，lr 2.56e-4），depth PPO 是 2 组（policy + encoder）；`load()` 的 `except (ValueError, KeyError)` 会降级重建 |

### 9.3 校验（`--dry-run` 与产物各跑一遍）

```
actor keys      26  (all remapped, 26/26 shape 匹配)
critic in       (512, 753)
overwrite delta max |teacher - student|: 2.830
std             (12,) dims, mean 0.598 (kept)
encoder         18 keys / estimator 6 keys / optimizer dropped
```

产物 `logs/extreme-parkour/stage3-from-student/model_0.pt`（39.9 MB）逐项二值比对：actor == 学生 `depth_actor`、critic == 老师、std == 老师、encoder == 学生、estimator == 学生，`iter=0`、`infos=None`。源文件 `student/model_10000.pt` 未被改动（mtime 仍是 10-26；该文件自身的 `iter` 字段本来就是 0，`10000` 只是文件名）。

### 9.4 起跑命令

```bash
cd legged_gym/legged_gym/scripts
python train.py --task go2_depth --exptid stage3-run1 \
  --resume --resumeid stage3-from-student --checkpoint 0 \
  --use_camera --delay --headless --device cuda:0
```

`--resumeid` 是**目录**、`--checkpoint N` 精确解析成 `model_{N}.pt`（`get_load_path`），所以产物必须命名为 `model_<int>.pt`；`exptid` 必须是新目录，不能填 `stage3-from-student` 自己（否则覆盖起点）。

**仍未解除的前置条件**：这份学生是“站着不动”的 BC 产物（天花板 = teacher 的 level 0.3）。stage3 的 PPO 能改写它的动作，但起点本身不会走；路线图里的 ①（关 `only_positive_rewards` 裁剪 + 惩罚缩到 IsaacLab 量级）仍是第一优先。

### 9.5 顺带实测：`v2-student` 蒸馏的吞吐

同日观察正在跑的蒸馏（`--task go2_student`，280 envs）：

| 指标 | 值 |
|---|---|
| collection | **101.13 s** / 轮 |
| learning | **0.132 s** / 轮 |
| 合计 | ~100 s / 轮（ETA 约 16120 min @ iter 20） |

即 280 个 env 每控制步烧掉 ~4.2 s，而老师 6144 env 只要 ~0.13 s。瓶颈 100% 在 `env.step()` 里的 warp 深度渲染链路（`update_depth_buffer` → `WarpDepthCamera.render` + 全 batch 的 `process_depth_batch`），不是 PPO。按此速率 10000 轮要 11 天以上 —— 若蒸馏必须跑满，`render_on_demand` 的 `active` 只挡掉了 BVH 遍历、全 batch 后处理仍在每步执行，这是下一步唯一值得动的点；但改它会改数据分布，须单独 A/B。

### 9.6 `hole_fill` 搬到 GPU：同一帧，7.7× 吞吐（2026-10-29）

#### 结论先行

`process_depth_batch` 里唯一可以“不改变任何一个字节”就能砍掉的开销，是 RealSense hole fill 的 numpy 实现：它把 280×270×480 的深度拷到 host、用 cv2 跑 8 次 dilate、再拷回卡上。改成等价的 `max/min pool` 后，**同一帧逐位相同**，蒸馏从 **97.3 s/轮 → 12.5 s/轮**，10000 轮的 ETA 从 11.3 天降到 **1.45 天**。

因此“降分辨率 / 关 hole_fill / 换光栅化相机”这套 fast-lane **全部取消**：那些改动都会改变观测分布、必须重新蒸馏，而现在没有必要付这个代价。

#### 实现

- `legged_robot.py::_hole_fill_gpu`：mode 1 的 `cv2.dilate` → `max_pool2d(3,1,1)`（padding 用 -inf，与 cv2 的 `BORDER_CONSTANT` 同义）；mode 2 的 `cv2.erode` → 负号 + `max_pool2d`；mode 0 的 `fill_from_left` → `cummax`。因为每一趟都是确定函数，“这趟没改动任何像素”就不可能让下一趟有改动，所以 CPU 版的 early-break 换成固定 8 趟结果不变。
- `hole_fill_backend`（`"cpu"` 默认 / `"gpu"`）与 `--hole_fill_backend` CLI 覆盖，方便随时 A/B。
- `tests/test_hole_fill_backend.py`：mode 0/1/2 × iters 1/3/8 共 9 组随机帧（6% 丢点 + 左侧 6 px 盲带 + 40 行近距带）+ 2 个边界用例，`torch.equal` 逐位比对，**11 项全过**。

#### A/B（同机、同会话、同 `--resume new-run-v2/model_3800`，唯一变量是后端）

| 后端 | 各轮耗时（s） | s/轮 |
|---|---|---|
| cpu | 97.22 / 96.80 / 101.59 / 96.43 / 94.46 | **97.3** |
| gpu | 12.06 / 13.14 / 12.46 / 11.60 / 11.35 / 12.02 / 14.13 / 14.16 | **12.5** |

7.7×，每轮省 84.7 s。这也顺带校正了 §9.5：101.13 s 不是被并发进程污染的异常值 —— 今天独占整卡重跑 cpu 后端仍有 97.3 s，所以那 85 s 确实是 hole fill。

（`_apply_pw_holes` 的 129,600 元素 `flat.sort` 和 129,600 rays/env 仍在，是下一个数量级的来源；但在 1.45 天已经可接受的前提下，为了“不改观测分布”先不动。）

### 9.7 `new-run-v4`：课程为什么停在 7.0

v3 的 4000 轮里 `terrain_level` 钉在 7.0，此前的解释是“升降级带太窄”。这次回到代码后有两个发现，其中一个推翻了原计划：

**（1）`promote_goal_frac` 从 0.375 提到 0.75 是个空操作。** `_update_goals` 里 padding goal 是即时推进的（`reach_goal_timer[padded_goal & reached] = delay/dt + 1`），而 swimmer layout 的 slot 4-7 全是终点返回点的副本。所以一旦到达索引 3（跑完整圈），4→8 会在 4 步内自动补齐。`max_goal_idx` 在 (0.375, 1.0] 之间的任何 promote 门槛都等价于“跑完一圈”。**保持 0.375，并在注释里写明这件事**，避免以后有人以为调高过。

**（2）真正把均值压到 7.0 的是天花板重采样。** `resample_at_max_level=True` 把到达第 9 行的 env 均匀重掷到 `[resample_min_level, 9)`，而 `uniform[4,9]` 的期望正好是 **6.5** —— 与实测 7.0 重合。也就是说 env 确实够到了第 9 行，却在那一行被训练够之前就被扔回 4~8 行。

v4 因此改三处：

| 旋钮 | v3 | v4 | 理由 |
|---|---|---|---|
| `roll_term_deg` | 95 | **120** | 95° 时贴墙的 bank 姿态已经逼近阈值；放宽到 120 让 plant 期间的姿态由奖励而不是终止去塑形 |
| `demote_goal_frac` | 0.25 (2/8) | **0.15 (1/8)** | 旧门槛下“漏一次 plant”就撤销一次晋级，在第 8 行接近抛硬币；新门槛要求“连第一个栏都没过去”才降级 |
| `resample_min_level` | 4 | **7** | 保留天花板重采样（防第 9 行过拟合），但把落点从 4~8 收到 7~8 |

`promote_goal_frac` 保持 0.375（见上）。`only_positive_rewards=False`、`enable_wococo_armed_term_exempt=False`、`dr_ramp_full_at_mean_level=6.0` 均与 v3 一致，未动。

#### 状态：已放弃（同日决定）

v4 起跑后跑到 iter 65（level 3.01，v3 同期 2.40，启动指纹一致：iter 1 的 level 1.64 / roll 0.19 / timeout 0.71）。按要求停掉，**三处改动已全部回退**（`roll_term_deg` 95、`demote_goal_frac` 0.25、`resample_min_level` 4），改为直接蒸学生。

上面两条发现仍然成立，留作下次推课程时的依据：`promote_goal_frac` 在 (0.375, 1.0] 区间是同一个门；`uniform[4,9]` 的期望 6.5 是“均值钉在 7.0”的主要解释。`new-run-v4/` 目录里只剩 `model_0.pt` / `model_100.pt` 两个中断产物（未删）。

### 9.8 `v3-student`：直接蒸 v2 的老师（2026-10-29 起）

```bash
python train.py --task go2_student --exptid v3-student --proj_name extreme-parkour \
  --resume --resumeid new-run-v2 --checkpoint 3800 --use_camera --delay --headless \
  --device cuda:0 --max_iterations 10000
```

与 `v2-student` 的**唯一差异是 `hole_fill_backend = "gpu"`**（§9.6，逐位一致）。观察量：`original=(480,270)`、`resized=(87,58)`、`min_terrain_level=4`、`flyball_box 1.0`、`only_positive_rewards=False`、`roll_term_deg=95` 全部与 v2-student 相同。

#### 实测吞吐（iter 0~9）

| iter | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| s/轮 | 12.34 | 13.26 | 12.30 | 11.34 | 11.78 | 12.23 | 14.60 | 14.37 | 14.54 | 15.06 |

均值 ~13.2 s/轮 ⇒ **10000 轮 ≈ 36.7 h（1.5 天）**，对比 v2-student 的 11.3 天。

#### 启动状态

| 项 | 值 | 说明 |
|---|---|---|
| `terrain_level` | 4.0000 | 恰好等于 `min_terrain_level`，全员贴地板 |
| `reach` | 0.0000 | 一开始跑不完一圈（预期：权重来自老师、观测却是深度） |
| `term_roll` | 0.0441 | 4.4% 摔倒 |
| `term_timeout` | 0.0000 | 4.4% 以外的那 95% 由非 roll 的条件终止 |

#### 已知天花板

老师 `new-run-v2/model_3800` 的峰值是 **level 6.99 @ iter 3871**。BC 蒸馏无法超过被模仿者，所以这一轮的目标是“把 6.99 的老师尽可能完整地搬到深度策略上”，不是“推到 9 级”。要后者，得先回去推老师（§9.7 的两条发现是现成的切入点）。

#### 看什么

1. `terrain_level` 能否从 4.0 爬起来（爬到 ~6 说明 BC 真的在传递技能）；
2. `depth_actor` 与老师的动作差（`update_depth_actor` 的 loss）是否单调下降；
3. `term_roll` 是否随轮次下降（学生摔倒率是“像不像老师”最直接的代理指标）。

### 9.9 `num_envs` 扫描：280 就是天花板，而且加 env 本来也不赚（2026-10-30）

#### 为什么要扫

`camera_num_envs = 280` 从初始提交（`04c2457`, 2026-08-13）写死至今，从未改过；`v2-student` 与 `v3-student` 的 `env_cfg.pkl` 都读出 `num_envs=280`。`--num_envs` 在 helpers.py L181 生效，且**晚于** L158 的 `num_envs = camera_num_envs`，所以能覆盖。

#### 实测（同一 checkpoint `v3-student/model_5000`，`--use_camera --delay`，6–8 轮）

| 地形 | N | s/轮 | 峰值显存 | 结果 |
|---|---|---|---|---|
| 10×80（真实） | 280 | 14.09 | 24040 MiB | 正常 |
| 10×80 | 560 | — | 22424 MiB | **OOM**：`_apply_pw_holes` 的 `flat.sort` 要 832 MiB |
| 10×80 | 1120 | — | 23132 MiB | **OOM** |
| 10×2（仅为压启动时间的斜率采样） | 70 | 6.24 | 9296 MiB | 正常 |
| 10×2 | 280 | 12.04 | 19462 MiB | 正常 |
| 10×2 | 560 | — | 23384 MiB | **OOM** |

`10×2` 是唯一能压启动时间的手段：800 块地形一次性建图 **764 s**（单核 CPU），20 块只要 **48 s**。注意 `--rows` 不能压到 `min_terrain_level`（=4）以下，否则 `_get_env_origins` 里地形索引越界、`device-side assert`。

#### 两点拟合

```
t(N) = 4.31 s + 27.6 ms x N        # s/iteration
m(N) = 5907 MiB + 48.4 MiB x N     # peak VRAM（20 块地形）
     + 4578 MiB                    # 800 块地形的固定增量
```

| N | s/轮 | 样本/s | 显存(800 块) |
|---|---|---|---|
| 70 | 6.24 | 269 | 13874 MiB |
| 280 | 12.04 | 558 | 24040 MiB |
| 385 | 14.94 | 618 | 29123 MiB（超） |
| 560 | 19.77 | 680 | 37595 MiB（超） |
| 1120 | 35.24 | 763 | 64704 MiB（超） |

#### 结论

1. **`N_max(800 块地形) = 290`，280 已经是天花板**，不是保守选择。显存 24040/24564 = 98%，没有任何余量。
2. **扩 env 本来也不赚**：样本/s 的渐近上限是 `24/0.0276 = 869`，280 已经站在 64%。560 只值 +22%、1120 只值 +37%，而这两档显存都不可能。所以不要把下一轮的时间花在 env 数上。
3. 每步的构成是 `0.18 s 固定 + N x 1.15 ms`。1.15 ms/env/步 ≈ 每帧每 env 5.75 ms / 129600 射线 = **44 ns/ray**，深度渲染才是斜率的全部来源，PPO 更新量级上根本不占。
4. **OOM 的炸点不是相机，是 `_apply_pw_holes` 的 `flat.sort(dim=1)`**：560 env 时它要 832 MiB 的瞬时分配（values + int64 indices + workspace），280 env 时也要 416 MiB。而 280 env 的稳态已经占到 98%，等于这条长跑没有余量：任何一处多出几百 MiB 的分配就会挂。**低风险加固：把阈值计算按 env 分块（如 64 一块）或换成 `torch.kthvalue`，结果逐位不变，瞬时占用与 N 解耦。**
5. **每 env 48.4 MiB 的构成尚未查清**：一个不开 Isaac 相机的 sim 不该有这么多。但即使查清并完全消掉，收益上限也只是 +56% 样本/s，所以优先级低于“减少射线”。

#### 若真要提速，唯一的杠杆是射线数

阶梯全部落在"改数据分布、须重训 + A/B"这一类：`original` 480x270（现在是相对 `resized` 87x58 的 5.5x 超采样）、`capture_interval`、`update_interval`。它们与 `hole_fill_backend="gpu"` 那种零分布变化的提速不同，不能顺手改。

---

## 10. 地形难度语义改造（2026-11-13，进行中）

### 10.1 唯一变量：地形的"难度怎么随行变化"，任务与权重不变

上一轮 `student-depth-rl-long` 的 `env_cfg.pkl`（11-13 08:41）里，地形是 main 的语义：
`flyball_hurdle_jitter_scale_with_difficulty = True`（抖动 ×d）、三处走廊抖动 0.30、
没有 `geometry_at_max_difficulty`。本轮把它换成"从第 0 行就是满难度、满抖动"，
其余（exptid、warm start、env 数、优化器、奖励）一律不动。

### 10.2 起跑命令全文

由 `maxdr-run/run_student_depth_rl_watchdog.sh` 托管（2026-11-13 10:26:52 起，PID 500940）：

```bash
python -u train.py --task go2_student --exptid student-depth-rl-long \
  --proj_name extreme-parkour --headless --no_wandb --device cuda:1 \
  --num_envs 1024 --max_iterations 10000 --save_interval 1000 \
  --freeze --depth_rl \
  --resume --resumeid student --checkpoint 10000 \
  --use_camera --delay
```

warm start：`legged_gym/logs/extreme-parkour/student/model_10000.pt`；
绝对目标 `TARGET_ITER = 20000`，watchdog 每次重启按 `20000 − ckpt` 重算 `max_iterations`。

### 10.3 施工内容

| 文件 | 改动 |
|---|---|
| `utils/flyball_wall.py` | `_jit()` 统一走廊抖动（双向/单向、是否乘 d 三个开关集中一处）；`geometry_at_max_difficulty` 把 `delta_x`、`delta_z` 钉到区间上界；`rough_h` 改为每 tile 在 `[0.02, 0.06]` 全范围抽，并保留 50% 全平；栏顶粗糙度上限钉到 0.01；g5 用 `min(gap, g4 − 0.30)` 保证终点 ≥ 0.30；compact 布局超界时**不重抽、不平移**，只取剩余空间（`max_gap_px`） |
| `utils/terrain.py` | 透传 `flyball_geometry_at_max_difficulty` / `flyball_hurdle_jitter_one_sided` / `flyball_return_goal_gap_jitter`；地形缓存 key = cfg + 生成器源码 hash + RNG |
| `envs/go2/go2_teacher_config.py` | `geometry_at_max_difficulty=True`、`jitter_scale_with_difficulty=False`、`jitter_one_sided=False`、三处走廊抖动 0.50、`return_goal_gap_jitter=0.30`、`wall_to_hurdle_rear_min=2.00` |

### 10.4 几何后果（每行 50 万次复算）

| 量 | main | 现在 |
|---|---|---|
| 出生点 → 栏 1 前沿 | 行 0 恒 1.60；行 9 1.10–2.10 | **全行** 1.10–2.10（最小 x = 2.10） |
| 栏 2 → 墙前脸（run-up） | 行 9 1.50–2.50 | 2.00–2.50；0.0027% 的 tile 被剪 ≤ 0.12 m |
| 墙前脸 x | 行 0 恒 6.96；行 9 6.02–7.52 | 全行 5.96–8.44 |
| g5 终点 x | 行 9 −0.40–1.20，13.3% 被钳到 0 | 0.30–1.20，40% 恰好取到 0.30 |
| `delta_x`（禁区宽） | 0.20–0.65 按行 lerp | 恒 0.65 |

两点说明：

1. g5 有 40% 的 tile 贴着 0.30 的下限，但每块 tile 独立，策略感知不到"多个终点重合"，只是这四成地块的回程由 1.70 m 缩到 1.30 m。
2. `forbidden_zone_contact` 的 scale 在 teacher config 里是 0.0（`_prepare_reward_function` 会 pop 掉零权重项），所以 `delta_x = 0.65` 目前**不产生任何奖励项**，只写进 `flyball_forbidden_x0/x1` 的 metadata。

### 10.5 启动代价

地形生成器源码变了 → 旧的 `r10xc80_hs0.01_*.pkl`（1.36 GB，11-13 00:05）hash 失效，需要冷构建：
heightfield 355 s + Delatin 186 s + PhysX BVH 168 s ≈ **708 s**，之后才出现第一次 iteration。

### 10.6 待填（每 200 轮）

`terrain_level` / `term_reach_goal` / `term_base_contact` / `vfl` / `reward` 暂无数据；
课程峰值与到达轮次、末权重编号、clean 与 full-DR 的分级评测都待本轮跑出来再补。

### 10.7 一句话

这轮验证的是：把"难度随行渐进"换成"从第 0 行就满难度/满抖动"后， 已经在**旧地形**上训到 10000 轮的 student 权重，换到新地形继续训 能否稳定收敛。

---

## 11. student 深度链路微调（long2 → long5，2026-11-13）

任务从 teacher 换到 student：`go2_student` + `--depth_rl --use_camera --delay`，
warm start `student/model_10000.pt`，1024 envs，cuda:1，地形保持 §10 的最大难度版本。
目标是回答"蒸馏后的 student 能不能靠 PPO 再涨一段"。到 long4 为止的答案是**不能，而且会掉**。

### 11.1 时间线

| # | exptid | 起止 | 轮次 | 关键改动 | 结局 |
|---|---|---|---|---|---|
| E09 | `student-depth-rl-long2` | 16:05–17:05 | 102 | anchor 1.0、`step_ratio 0.05` | 停：lr 102/102 在地板 |
| E10 | `student-depth-rl-long3` | 17:26–17:35 | 8 | 同上，修 double-backward 后重跑 | 停：lr 8/8 在地板 |
| E11 | `student-depth-rl-long4` | 17:39–18:29 | **217** | `step_ratio 0.4`、anchor 10.0 | 停：能力劣化（见下） |
| E12 | `student-depth-rl-long5` | 18:34–19:10 | 136（手动停） | ARM-A：σ 冻结 + encoder 冻结 + lr 固定 1e-5 | 验冻达标（判据 1/2 成立）；1e-5 步长仅有效步长 ~1/8，验学无效，停 |
| E13 | `student-depth-rl-long6` | 19:11–20:26 | **300**（rc=0） | ARM-B：仅 lr 1e-5 → 8e-5（kl 标定 ×8），其余同 ARM-A | 停：critic 发散（value loss 0.3→2.2e9），判据 3/4 不成立但结论无效（见 11.9） |
| E14 | `student-mainenv-s1` | 21:26–22:24 | ~150（手动停，用户授权） | 用户启动：mainenv 树（main 原环境）同回路对照 | 停：给 E15 让 GPU；116 轮 value 稳定 2.5~3.3（对照结论已拿到）；ckpt 留到 model_150 可续 |
| E15 | `student-depth-rl-long7` | 22:26–23:42 | **300**（rc=0） | ARM-C：仅 critic lr 3e-4 → 3e-5，其余同 ARM-B | critic 慢发散未根治（value 2.3→22.6）；86% 摔死、reach_goal≈0、站桩偏置实锤→瓶颈在蒸馏起点（见 11.10） |
| E16 | `teacher-maxdr-warm3200` | 23:49– | 进行中（10000 轮） | priv teacher 热启动 model_3200，修改地形，纯 0 课程 + 全局单向 DR，标准 PPO | 首 14 轮：dr_scale_mean==max==0（单向生效）、value loss 1.18→0.73 下降（critic 健康）、std 0.60；ETA ~22h |

E10 的中断原因是 `RuntimeError: Trying to backward through the graph a second time`：
anchor 项和 surrogate 都经由重放的 `latent`，单独再 `.backward()` 会二次遍历已被释放的图。
修法是把 anchor 直接加进 `loss`，算术上等价，一次遍历。

### 11.2 long4 的 217 轮

分段中位数（`n` = 该段轮数；**`rew/step` 一列由该段 `rew` 与 `ep_len` 的中位数换算**，
不是逐轮比值的再取中位，两者差在小数点后第二位）：

| 区间 | rew | ep_len | **rew/step** | goal_idx | term_roll | kl | lr 中位 | anchor |
|---|---|---|---|---|---|---|---|---|
| 20–79 | −8.72 | 150.3 | −1.16 | 0.060 | 0.655 | 1.85 | 1e-5 | 0.947 |
| 80–139 | −9.92 | 159.7 | −1.24 | 0.060 | 0.623 | 2.77 | 1e-5 | 0.892 |
| 140–216 | −6.78 | 100.6 | **−1.35** | 0.047 | 0.726 | 2.43 | 1e-5 | 0.839 |

单点：

| it | rew | ep_len | rew/step | goal | roll | lat | kl | lr | σ |
|---|---|---|---|---|---|---|---|---|---|
| 10 | −5.33 | 121 | **−0.88** | 0.045 | 0.668 | 0.136 | 1.59 | 1.5e-5 | 0.054 |
| 100 | −19.65 | 175 | −2.25 | 0.085 | 0.608 | 0.153 | 3.13 | 1.5e-5 | 0.069 |
| 200 | −5.88 | 87 | −1.35 | 0.032 | 0.749 | 0.079 | 1.47 | 1.0e-5 | 0.088 |
| 216 | −6.51 | 79 | **−1.66** | 0.032 | 0.772 | 0.069 | 4.19 | 1.0e-5 | 0.087 |

**测量陷阱**：logger 写的是 `episode_sums.mean() / max_episode_length_s`，episode 越短，
所有按步累积的负项绝对值同步缩水，于是 `Mean reward (total)` 会假性回升。`rew_termination`
恒等于 `−0.1500`（= −3.0 × 1/20）就是这条归一化的指纹。唯一可比的读数是
`rew_per_step = rew * 20 / ep_len`，按它看是从 it10 起单调劣化。

### 11.3 两组校准

**(a) teacher 在同一个 reward 上能学**（`maxdr-v1/v2.log`，同样除以 20 s 归一化）：

| run | rew/step | ep_len | goal |
|---|---|---|---|
| v1 it0 → it936 | −3.15 → −0.42 | 24 → 681 | 0 → 0.63 |
| v2 it0 → it501 | **−2.68 → −0.15** | **21 → 935** | **0 → 0.58** |

所以奖励函数本身是良态的，不是奖励设计让 student 停滞。

**(b) 蒸馏缺口**（同一 eval 配置，`--freeze` 下各跑 4 段）：

| 权重 | rew | ep_len | goal |
|---|---|---|---|
| teacher `model_3200` | −2.25 ~ −2.31 | 122 ~ 154 | **0.18 ~ 0.22** |
| student `model_10000` | −4.21 ~ −7.08 | 144 ~ 162 | **0.052 ~ 0.069** |

student 的目标推进只有 teacher 的 1/3，而 episode 反而更长 —— 蒸馏学到的是"站得住"，
不是"推得动"。这一组同时说明：long4 的起跑线本身就只有 0.06，不是 RL 把它练差了才开始差。

### 11.4 权重空间 diff（`model_0` → `model_100` → `model_200`）

anchor loss 是**访问状态的函数**，不能当权重漂移的证据；只有逐 tensor diff 能。

| 参数组 | it100 | it200 | 备注 |
|---|---|---|---|
| `depth_encoder_state_dict` | GLOBAL_REL 8.60e-2 | **1.20e-1** | `output_mlp.0.weight` 13.9% → 19.2% |
| `depth_actor_state_dict` | 3.66e-2 | 5.63e-2 | history_encoder 变化最大（0.47 → 0.57） |
| `estimator_state_dict` | 6.66e-2 | 9.77e-2 | 有监督，预期内 |
| `model_state_dict` | 2.04e-1 | 3.20e-1 | 只有 8/35 个 tensor 变，**全是 critic**；teacher actor 确认冻结 |
| `critic.6.weight` | rel 2.26 | rel **5.84** | value 函数始终没收敛 |
| `depth_actor_std` | 0.05006 → 0.06862 | → **0.08760** | +75%，不在任何监控阈值里 |

### 11.5 四个缺陷，以及一条把它们串起来的正反馈

1. **σ 无约束漂移**。`depth_actor_std` 是 `nn.Parameter` 且在 optimiser group 0 里；
   `rl_entropy_coef = 0` 只去掉了显式熵奖励，surrogate 的 `ratio` 对 σ 仍有梯度。
   而 `kl_target = 6·ratio² = 0.96` 是按 σ=0.05 标定的，`KL = 6·(Δμ/σ)²`，σ 涨到 0.088 后
   同一 Δμ 只测出 0.32 倍 KL → **允许的步长自己放大到 3 倍**。
2. **encoder 漂移被记到了 KL 账上**。trust region 比较的是“重放的 latent”与
   “rollout 时的 latent”，encoder 权重一动，同一帧的 latent 就变、μ 跟着变。所以
   long4 的 kl 2.43（max 24.3）里绝大部分不是策略位移。证据在 11.6：同样是 lr 1e-5，
   把 encoder 冻住后 kl 只有 **0.014**。
3. **anchor 只减缓不钉住**。coef 10 把 anchor loss 从 1.144 压到 0.883，但 encoder
   权重照样漂 8.6% / 12%。两个指标互相矛盾时，可信的是 tensor diff。
4. **critic 翻搅**。`critic.6.weight` 200 轮 rel 5.84，advantage 里噪声占主导。

串起来是一个自强化环：

```
encoder 漂 → KL 虚高 → 自适应 lr 降到底 → actor 每轮只走 0.002 rad → 学不动
      ↑                                                                    │
      └──────────── 策略劣化、访问新状态（ep_len 175 → 79）←──────────────┘
```

这也修正了 11.2 里“lr 在地板”的定性：它不是根因，是环中间的一个环节。


### 11.6 ARM-A 控制臂（E12）

把除 actor 之外的一切钉死，用一次实验区分"PPO 回路坏了"和"蒸馏缺口补不上"：

| 参数 | 值 | 实现 |
|---|---|---|
| `rl_freeze_std` | `True` | `depth_actor_std.requires_grad_(False)` + 移出 optimiser group 0 |
| `rl_adaptive_lr` | `False` | lr 钉 1e-5；`kl_sum/kl_n` 照常累加，诊断不丢 |
| `rl_encoder_learning_rate` | `0.0` | encoder 冻在 DAgger 解上 |

地形、奖励、DR、`--freeze`、`--delay`、1024 envs 全部与 long4 相同。

判据（`rew*20/ep_len`，不看 `rew`）：

1. σ 在日志和 checkpoint diff 里都是 0.050；
2. `term_roll` 回落到 it10 的 0.65 附近（σ 漂移的直接代价）；
3. 每步回报停止下滑；
4. `max_goal_idx` 抬过蒸馏基线 0.052–0.069。

前 14 轮（18:34–18:41，实际 `max_iterations` 写成了 600 轮而不是 60）：

| 项 | 值 |
|---|---|
| σ | **14/14 轮都是 0.050** |
| kl | **0.0123 ~ 0.0150**（target 0.96） |
| lr | 固定 1.00e-05 |
| ep_len | 36.6 → 132.2（前 5 轮是 warm-up，从随机 episode 位置起） |
| goal_idx | 0 → 0.0695 |
| term_roll | 0.87 → **0.70**（long4 it10 是 0.668） |

判据 1、2 已成立。但 kl 只有 target 的 1.5% —— 同样是 lr 1e-5，long4 的 kl 是 2~4，
这里只有 0.014，**唯一变化是 encoder 不动**。这同时确认了两件事：

- 11.5 的第 2 条成立：long4 的 KL 主体是 encoder 漂移，不是策略位移；
- 这个 lr 小约 8 倍（`kl ∝ lr²`）。有效控制臂应跑 6e-5–8e-5，
  或者直接恢复 `rl_adaptive_lr = True`（encoder 冻结后 kl 是可测量的，
  调度器不再被谎报的 KL 骗）。

若 1 成立而 3/4 不成立 → 固定特征上 actor 也学不动，瓶颈是蒸馏缺口，
下一步应重做蒸馏而不是继续改 RL 超参。

### 11.7 两个操作坑

- **`TARGET_ITER` 与轮数差一个 `BASE_ITER`**。watchdog 里 `max_iterations = TARGET_ITER − (BASE_ITER + rel)`，
  所以想要 60 轮写 `10060`，写成 `10600` 就是 600 轮。E12 的第一版脚本写大了 10 倍。
- **日志带 ANSI 转义**。`\x1b[1m Learning iteration 0/600 \x1b[0m` 让 `strip()` 后的
  正则一条都匹配不到，解析前必须先 `\x1b\[[0-9;]*m` 剥掉。

### 11.8 一句话

奖励是好的、回路是通的、student 起跑线只有 teacher 的 1/3 —— 而 long4 的 217 轮把
"σ 漂移 + critic 翻搅"两个自由度放到野外，结果是每步回报从 −0.88 掉到 −1.66。
ARM-A 存在的意义是把这两个自由度锁死，让"actor 到底能不能学"变成可回答的问题。

### 11.9 long6（ARM-B）300 轮：步长修好了，critic 炸了（2026-11-13 晚）

lr 8e-5 生效：it0 kl=0.9707 ≈ target 0.96，标定精确命中；稳态 kl 0.06~0.07。
分段中位数：

| 区间 | rew/step | ep_len | goal | roll | kl |
|---|---|---|---|---|---|
| 5–50 | −0.95 | 137 | 0.074 | 0.682 | 0.094 |
| 100–150 | −1.18 | 151 | 0.085 | 0.653 | 0.064 |
| 200–250 | −1.29 | 152 | 0.090 | 0.676 | 0.071 |
| 250–299 | −1.17 | 151 | 0.088 | 0.675 | 0.070 |

判据 1/2 成立（σ=0.050、roll 0.65~0.68）；3/4 不成立（rew/step 缓降、goal 仅略高于基线）。
**但这个"学不动"结论无效**，因为权重 diff 和 TB 给出了新的 smoking gun：

| 证据 | 读数 |
|---|---|
| `depth_encoder` GLOBAL_REL | **0.000**（冻结完美） |
| `depth_actor` GLOBAL_REL | 4.07e-2（actor 确实在动，比 long4 的 3.66e-2@100 略大） |
| `critic.6.weight` rel | **15.6**（long4 200 轮才 5.84） |
| `Loss_depth/rl_value` | it0 0.3 → it30 5.5 → **it50 破百** → 中段 8.6e5 → 末段 **2.2e9**（max 7.9e9） |
| `Loss_depth/rl_estimator` | 0.042 → 0.032（健康，排除共享链路问题） |

即 **critic 从 it0 单调发散**，而 `update_depth_rl` 的 advantage 不做归一化，
发散的 value → 垃圾 advantage → actor 被推着乱走（这解释了"权重在动、能力在掉"）。

**对照（E14，用户 21:26 启动）**：`student-mainenv-s1` 用 mainenv 树（main 原环境）
跑同一回路、同样 critic lr 3e-4，116 轮 value loss 稳定在 2.5~3.3，无发散。
→ 发散是**修改后环境（最大难度几何 + armed 终止 + DR 从 0 拉满）与 warm-start critic
（DAgger  checkpoint 里按 main 环境回报训的 value）的分布错配**触发的，不是 PPO 回路本身。
固定 critic lr 3e-4 × 20 步/轮在错配下过冲，bootstrap 正反馈炸掉；teacher 的 maxdr-v2
能学是因为 critic 从 0 联合训练且 lr 自适应。

**推论**：
1. §11.6 的"蒸馏缺口"判据必须在 critic 稳定的前提下才生效；long6 不满足前提。
2. 下一臂先修 critic：`rl_critic_learning_rate` 3e-4 → 3e-5（或 1e-5），或前 100~200 轮
   冻 critic；同时盯 `rl_value` 是否单调爬升作为早期预警（it30 前就能看出来）。
3. E14 若在 main 环境上涨能力，则同时证明回路可用 + 问题在修改环境的分布/门槛。

### 11.10 long7 中期（it233）：瓶颈不在优化器，在蒸馏起点活不下来

分段中位数：rew/step −0.95（it5–50）→ −1.20（it200–233），goal 0.08~0.096 持平，
ep_len ~147，roll ~0.65，课程钉在 level 4.00 不动。三个新证据：

1. **critic 3e-5 仍慢发散**：rl_value 每 20 轮均值 2.34 → 2.53 → 2.91 → 4.84 → 8.91 → 16.93 → 22.58，
   it60 后单调爬升。降 lr 只延缓，治不了 warm-start value 与修改环境回报的错配。
2. **终止原因：86% 是摔死**（term_roll 0.65 + term_base_contact 0.21），term_lateral 0.14；
   `term_reach_goal ≈ 0`、`term_plant_dwell = 0` —— 策略根本没碰到墙，瓶颈在生存而非蹬墙技术。
   rew_collision 从 −0.075 翻倍到 −0.160，是 rew/step 下滑的主项。
3. **站桩偏置实测确认**（旧 hypothesis 落地）：tracking_goal_vel 里 standstill +0.076 vs
   moving +0.030，正向信号 72% 来自站着不动。

对照 §11.3a：teacher v2 从 0 在同一 reward 上能学到 goal 0.58 / ep_len 935 ——
**环境可学，但蒸馏起点在修改地形上连站都站不稳**，σ=0.05 的 PPO 小步探索逃不出摔倒盆地。
按 §11.6 判据，下一步应重做蒸馏（用能在修改地形上做任务的 teacher，如 maxdr-v2，
在新分布上 DAgger），而不是继续调 RL 超参；若继续 PPO 需冻 critic + advantage 归一化，
但预期收益低。站桩偏置的奖励结构问题留待重蒸馏/微调时一并处理。

## 12. 运维事件与转向 maxdr「那一版」（2026-11-14）

### 12.1 磁盘清理（B 级）
服务器盘 99% 满（1.9T 剩 19G）。xiaoyulong 一级占用最大是他人项目
`latent-safety-reproduction`（134G）、`fanlineng`（129G），不动。yunhanzheng 28G 内
按用户勾选只清 **B 级：已结束的 student run**——删 `student-depth-*` 共 10 个目录
（long6/long7 各 946M + long2×2/long3/long/ppo/std005/rl/frozen），释放 ~3G（19G→22G）。
**保留**：`student`(model_10000 base)、`teacher`(model_3200)、`maxdr-v2`(goal0.58 参考)、
活跃 `teacher-maxdr-warm3200`、`envs`、旧树 `logs`（活跃 run 经 symlink 在用）。
A（旧地形缓存~3G）/C（其他项目~11G）/D（纯缓存~0.5G）未清，待用户决定。

### 12.2 TensorBoard 恢复
远端 TB（6110，tmux `tb_depthrl`）与本地隧道（6010）均已死。重建 watchdog 脚本
`maxdr-run/run_tb_depthrl.sh`（镜像 run_tb_mainenv.sh，port 6110、logdir=ppo 树
extreme-parkour、reload_interval=10）并重启 tmux 会话；本地隧道 setsid nohup 重连，
HTTP 200 验证通过。

### 12.3 SSH 封禁事故（教训）
网络抖动后，本地隧道脚本 `.tmp_tb_tunnel.sh` 的 **固定 5s 重连循环**持续冲击服务器，
叠加人工多次重试，触发远端 `kex_exchange_identification: Connection closed by remote host`
（疑似 fail2ban / sshd 限流），所有 SSH 被拒。已停掉隧道循环止血。**教训：隧道重连
应改指数退避，避免紧密重试触发封禁。** 训练由 watchdog 独立托管，不受 SSH 断连影响。
用户指示「先不连了」，待冷却后单次试连。

### 12.4 转向 maxdr「那一版」：调查结论
用户要求切回「难度随机 + 地形随机 + 随机化全开最大」的那一版。定位 = **`go2_teacher_maxdr`
任务（`Go2TeacherMaxDrCfg`，文件 `go2_teacher_maxdr_config.py`）**，文件头自述
"every randomization pinned at full strength, no promotion/demotion, no DR ramp"。
- **难度随机**：`curriculum=False`+`max_difficulty=False` → terrain.py `curiculum(random=True)`，
  每砖 `difficulty=np.random.uniform(0,1)`，不升降级。
- **DR 全满**：`dr_scale_range=[1.0,1.0]`+`dr_ramp_use_mean_level=False` → dr_scale≡1.0；
  `action_delay_start_latched=True`；奖励门槛 step0 全副武装。
- **代码无需移植**：ppo 树与旧树（maxdr-v2 跑的树）的 `terrain.py`/`flyball_wall.py`
  **字节完全一致**；DR/奖励侧开关 ppo 树全支持。
- **唯一缺口**：ppo 树 `envs/__init__.py` 只注册了 `go2_teacher`，**未注册 `go2_teacher_maxdr`**，
  需补 import + task_registry.register。
- **dead knob 说明**：`flyball_wall_angle_uniform`/`flyball_hurdle_count_uniform`/
  `flyball_roughness_uniform_amplitude` 在两树 terrain.py 里都无实现（当年 v2 亦被静默忽略）；
  但因 difficulty 本身 U(0,1)，墙角度/roughness 经 lerp 已铺满全幅，效果基本等价。

### 12.5 地形随机化对比（main vs maxdr）与栏数分布回退
用户决定：**栏数分布改回 main**，其余 maxdr 地形随机化保留。

| 维度 | main 基准 | maxdr 那一版 | 决定 |
|---|---|---|---|
| 难度分布 | curriculum=True 课程升降 | curriculum=False，d~U(0,1) | 保持 maxdr |
| 墙角度 | (30,75)+jitter±5，随 level | (30,80)+jitter0，随随机 d | 保持 maxdr |
| **栏数分布** | zero_prob=0.25 + teacher 默认计数 | count_uniform=True+count_probs=None+zero_prob=0 | **改回 main** |
| 栏高度 | (0.07,0.12) decorrelate max0.12 | 继承 (0.07,0.12) | 相同 |
| 栏行门槛 | zero_below_level=4 | =0（每行可有栏） | 保持 maxdr |
| 位置 jitter 缩放 | scale_with_difficulty=True | =False 全幅 | 保持 maxdr |
| 栏1/栏2 位置 | 1.60±0.30 / 2.00±0.30 | 同 | 相同 |
| run-up | 2.00±0.30 min2.00 | 2.30±0.50 min1.80 | 保持 maxdr |
| roughness | [0.02,0.06] zero_prob0.5 | 同 | 相同 |
| 几何钉最大难度 | main 无（用户后加） | 继承 ppo teacher=True | 保持 |

**栏数回退具体改法**：删 maxdr 配置三行覆盖 `flyball_hurdle_count_uniform=True`、
`flyball_hurdle_count_probs=None`、`flyball_hurdle_zero_prob=0.0`，使其继承 teacher/main
默认（zero_prob=0.25，约 1/4 砖无栏）。保留：本地 main 的 flyball_wall.py 与远端 ppo 树
版本不同（远端有 count_probs 逻辑），继承的默认计数分布具体值待 SSH 恢复读远端 ppo
teacher 配置确认。

### 12.6 最终方案（用户拍板）与待办
**方案**：经多轮迭代，最终定稿见 **§12.8**（取代本节早期“继承改好的地形
geometry_at_max_difficulty=True”设想）。核心：与墙角度相关的 delta_x/墙角度走 main 公式 +
共享随机 d（保持物理耦合）；与墙角度无关的两个 roughness 钉到 9 级；jitter 保留 ±0.50 宽幅；
curriculum=False 随机抽难度；全满 DR + 奖励武装。栏数继承 main 的 zero_prob=0.25。

待办（SSH 恢复后，均需用户点头）：
1. 停 `teacher-maxdr-warm3200`（watchdog+train，只停这一项；用户已授权停，但因 SSH 封禁未能执行，**训练仍在跑**）。
2. ppo 树 `envs/__init__.py` 注册 `go2_teacher_maxdr` task。
3. 小改 flyball_wall.py 拆分 delta_x 与 roughness 控制（delta_x 恒走 main lerp，roughness 钉 9 级）；配置 curriculum=False、jitter ±0.50+scale_with_difficulty=False、g5 去掉、全满 DR、武装奖励（详见 §12.8）。
4. 以 `--task go2_teacher_maxdr` 起新 run（热启动源待定：teacher/model_3200 vs student/model_10000 vs 从零）。

### 12.7 改好的地形 vs main teacher 地形（逐行 diff）
核心：改好的地形新增 `flyball_geometry_at_max_difficulty=True`（main 无此功能，flyball_wall.py
改 167 行），把原本随 row 渐难的几何钉到最大难度 + 全幅随机 jitter。动机：heightfield 启动
时一次 baked、env 仍 demote，钉死才能防止降级后地形变易。

| 维度 | main teacher | 改好的地形 | 变化 |
|---|---|---|---|
| 几何难度策略 | 随 row lerp（0易→9难） | geometry_at_max_difficulty=True 每行钉最大 | 新增功能 |
| delta_x（禁区宽） | lerp 0.20→0.65 | 钉 0.65（每行不采样） | 钉死 |
| 地面 roughness | 行内 lerp 0.02→0.06 常量 | 每砖 U[0.02,0.06] 全幅，p=0.5 光滑 | 随机化 |
| 栏顶 roughness | lerp 0→0.01 随难度 | 上限钉 0.01，每砖 U[0,0.01] | 钉死+随机 |
| corridor jitter 缩放 | scale_with_difficulty=True | =False 全幅 | 不缩 |
| 栏1 位置 jitter | 1.60±0.30 | 1.60±0.50 | 加大 |
| 栏2 偏移 jitter | 2.00±0.30 | 2.00±0.50 | 加大 |
| run-up jitter | 2.00±0.30 min2.00（余2.0~2.3） | 2.00±0.50 min2.00（余2.0~2.5） | 加大 |
| g5 返回目标 jitter | 无 | 新增 ±0.30（1.70±0.30） | 新增 |
| 墙角度 | (30,75)+±5 随难度 lerp | 同（未改，不钉） | 不变 |
| 栏高度 | (0.07,0.12) decorrelate | 同 | 不变 |
| 栏数分布 | zero_prob=0.25 | 同 | 不变 |
| 栏行门槛 | zero_below_level=4 | 同 | 不变 |

代码层：geometry_at_max_difficulty 钉 delta_x=pit_width_range[1]=0.65、delta_z、地面/栏顶
roughness；**墙角度不钉**（仍随 difficulty lerp）。选项 A（curriculum=False）下 difficulty~U(0,1)
随机，墙角度经 lerp 铺满 30~80°，整片地形 step0 即“满难度全随机”。栏高度/栏数/行门槛未改。

### 12.8 地形方案最终定稿（用户选 A：地面 roughness 固定 0.06）
**关键洞察（delta_x↔墙角度耦合）**：delta_x=lerp(0.20,0.65,d) 与墙角度=lerp(30,75,d) 共享同一
难度 d，物理上“陡墙需更早起跳→禁区更宽”。故二者必须共享随机 d 保持耦合，**不能钉死 delta_x**
（geometry_at_max_difficulty=True 会钉 delta_x=0.65 但墙角度仍变→打破耦合，出现缓墙配最大禁区
的不一致砖）。与墙角度无关的两个 roughness 则钉到 9 级（最大）。

| 量 | 定稿 | 说明 |
|---|---|---|
| curriculum / max_difficulty | False / False | 每砖 d~U(0,1) 随机抽 |
| delta_x | main lerp(0.20,0.65,d) | 不钉，随 d 与墙角度耦合 |
| 墙角度 | main lerp(30,75,d)+±5 | 不钉，随 d（30~80°） |
| 地面 roughness | **固定 0.06**（真 9 级，A） | 钉最大幅值，p=0.5 光滑保留 |
| 栏顶 roughness | U[0,0.01]（9 级） | 上限钉 0.01 |
| jitter 幅度 | **±0.50 保留** | scale_with_difficulty=False 全幅不缩 |
| g5 return_goal jitter | 去掉 | 回 main（无） |
| 栏数/栏高度/行门槛 | main 原样 | zero_prob=0.25、(0.07,0.12)、zero_below_level=4 |
| DR | 全满 | dr_scale=[1,1]、dr_ramp_use_mean_level=False |
| rewards | 武装 | maxdr 式 start_armed |

**实现要点**：
1. `geometry_at_max_difficulty` 是单一开关（同时钉 delta_x+delta_z+地面/栏顶 roughness），
   表达不了“delta_x 走 main + roughness 钉 9 级”。需小改 flyball_wall.py：把 delta_x 的钉死
   与 roughness 的钉死拆成独立控制——delta_x 恒走 main lerp；地面 roughness 钉固定 0.06、
   栏顶 roughness 钉 U[0,0.01]。
2. 配置层：curriculum=False、max_difficulty=False；jitter ±0.50 + hurdle_jitter_scale_with_difficulty=False；
   g5 return_goal_gap_jitter 去掉；domain_rand 全满；rewards 武装。
3. 地形源码改动 → 缓存 checksum 变 → 冷构建 ~10min（旧缓存 5000562807 失效）。

**与“改好的地形”差异**：改好的地形=geometry_at_max_difficulty=True（delta_x 钉 0.65、地面 roughness
随机 U[0.02,0.06]）；本定稿=delta_x 回 main 耦合、地面 roughness 固定 0.06（更硬）、栏顶 roughness
U[0,0.01] 不变、jitter ±0.50 不变。

---

## 13. np1 主线诊断与 anti-crouch 调权（2026-11-16）

主线 run = `teacher-maxdr9rw-np1-warm3200`（bayesdl GPU0，task `go2_teacher_maxdr`，6144 envs，
`wococo_stage1_n_persist=1`，热启动 teacher/model_3200）。本节记录 it10000 前后的行为诊断、
it0 vs it10000 奖励分解，以及据此拍板的纯权重 anti-crouch 调整。

### 13.1 续训与叫停
- 从 `model_10000.pt` 续训 +10000（`--max_iterations` 是**增量**，on_policy_runner L248
  `tot_iter = current + num_learning_iterations`，故 10000→it20000，编号无缝）。
- 用户叫停于 **it10322**（续跑了 322 轮约 40min）。**最新落盘 ckpt 仍是 model_10000.pt**：
  runner 在 it≥5000 后每 **500** 轮才存（下一存点 it10500），322 轮未落盘；日后续跑从
  `--checkpoint 10000` 恢复即可。TB（bayesdl 6110 / 本地 6010 隧道）保留未停。

### 13.2 it10000 L9 行为诊断（play-only，本地录制）
- it2000/3000/10000 三条 L9 视频均通关（stage3 disarmed + reach_goal），技能保留；与 fleet
  回退形成对照（fleet 稳定性问题，非能力丢失）。
- **approach 低蹲伏**：0:02 处 stage0，root_z≈0.244（min 0.224）、pitch≈+10°，搓步。
- **肘部不触地**：逐帧接触力 CSV（thigh/calf body）证实 0:02 窗口全 0N；全程仅 FL_calf
  t=8.58s stage3 单步 70.2N 落地刮擦。蹲伏不是肘撑地，是纯低姿态。
- **姿态奖励缺口**：`_reward_orientation`（legged_robot.py L3975-3981）mask 到平地 + over_pit
  置 0 → 墙道 env19 approach 姿态罚**恒 0**；且**无 base_height 项**。弱约束只剩
  dof_error=-0.04、hip_pos=-0.8、lin_vel_z=-1.0、ang_vel_xy=-0.02（只罚角速度）。
- **翻倒不终止之谜**（it6500 视频）：play `--level 9`→fleet_mean=9≥6 触发
  `armed_term_exempt` 满额豁免 roll/pitch/base_contact 三门→翻倒躺到 20s 超时；训练
  mean 冻结 4.5<6 永不 arm，故训练里翻倒会终止。`projected_gravity_xy`（重力在机体系 xy
  分量，‖g_xy‖²=sin²tilt）是 orientation/standstill_orientation 两奖励的倾斜信号，非观测。

### 13.3 it1000–2000 奖励下降归因
窗口内**无任何课程门控/DR/level 变化**（clearance it2000 才 arm 且 alpha=0；plant_fail 全程
armed；exempt 全程 0；mean 冻结 4.5）。系**行为回退**：term_roll 0.388→0.508、reach_goal
0.091→0.059、timeout −52%、ep_len↓、tracking↓；急性谷 it1200（mean_reward −0.416）。变化最快
项：term_roll +0.119、term_base_contact −0.060、rew_tracking_goal_vel −0.035、term_reach_goal
−0.032；运动正则几乎不动（<3%）。与旧 run 站桩退化同源（满 DR 冲击→approach 真翻）。

### 13.4 it0 vs it10000 奖励量级分解（share=|值|/Σ|值|，去重 tracking 子项）
| 项 | it0 值 | it0 share | it10000 值 | it10000 share | 绝对增长 |
|---|---|---|---|---|---|
| tracking_goal_vel | +0.236 | 37.7% | +0.636 | **39.3%** | ×2.7 |
| **dof_acc** | −0.037 | 5.9% | −0.164 | **10.1%** | **×4.4** |
| **action_rate** | −0.037 | 5.9% | −0.129 | **8.0%** | **×3.5** |
| tracking_yaw | +0.040 | 6.4% | +0.113 | 7.0% | ×2.8 |
| ang_vel_xy | −0.022 | 3.5% | −0.076 | 4.7% | ×3.5 |
| feet_contact_forces | −0.025 | 4.1% | −0.075 | 4.6% | ×2.9 |
| collision | −0.043 | 6.8% | −0.069 | 4.3% | ×1.6 |
| hip_pos | −0.028 | 4.5% | −0.068 | 4.2% | ×2.4 |
| **dof_error** | −0.034 | 5.5% | −0.060 | **3.7%** | ×1.8 |
| wococo_stage | +0.006 | 1.0% | +0.050 | 3.1% | ×8 |
| standstill_orientation | +0.017 | 2.6% | +0.048 | 3.0% | ×2.9 |
| **termination** | −0.057 | **9.2%** | −0.043 | **2.7%** | ×0.75 |
| orientation | 0 | 0 | 0 | **0** | 死项 |

**分组漂移**：运动正则组（dof_acc+action_rate+ang_vel_xy+torques）**15.6%→23.3%**；姿态罚组
（dof_error+hip_pos+lin_vel_z+dof_pos_limits）**14.3%→11.3%**（dof_error 单项 5.5→3.7%）。

**终止原因（episode 占比）**：roll 翻倒 59%→**34.6%**（仍头号）、base 触地 35%→**25.2%**、
reach_goal 1.9%→18.3%、stage1_ground 压墙失败 1.4%→**11.3%**、timeout 1.8%→9.8%。
**L9 fleet 成功率 1.3%→5.4%**，ep_len 174→418 步，mean_reward 0.163→4.465。

**解读**：策略落入「低蹲搓步」局部最优——运动正则涨成最大罚块（23.3%），姿态锚太弱
（dof_error 3.7%、orientation 恒 0）拉不回；蹲伏正是「省运动成本 + 保 39% tracking 收入」
的折中。终止恐惧已降到 2.7%，但 ~60% episode 仍以翻倒/触地收场→摔倒是主失败模式，非不敢动。

### 13.5 调权决策（用户拍板，纯 cfg，已改本地）
`go2_teacher_maxdr_config.py` rewards.scales：
| 项 | 旧 | 新 | 依据 |
|---|---|---|---|
| dof_acc | −1.2e-7 | **−0.6e-7** | 头号罚 10.1%、×4.4；减半松开「省能耗」压力 |
| action_rate | −0.05 | **−0.03** | 二号罚 8%、×3.5；搓步=低动作率，鼓励正常摆腿 |
| dof_error | −0.04（继承 base） | **−0.08**（显式 override） | 姿态锚 5.5→3.7% 太弱；参考站姿即 0.25m 高站姿，加倍≈base_height 权重版代理 |

- **termination 保持 −60**（恐惧已低，tracking 胡萝卜够强，本轮不动；若摔倒不降再议 −80~−100）。
- ang_vel_xy/tracking_goal_vel/wococo_stage/collision 一律保持。
- **dof_error 加倍安全性**：`_reward_dof_error` 在 `wococo_plant_window` 内已 ×0.25
  （legged_robot.py L4082-4084），压墙大关节 excursion 不受影响。
- 净效果：运动组 23.3%→~15%（回 it0 水平），dof_error 3.7%→~7%。
- **状态：仅本地改 + py_compile 通过；未同步 bayesdl、未重启训练**（部署/续训待用户指令）。

### 13.6 base_height 项提议（stage0/3，待实现，未动代码）
权重版 dof_error 是「关节偏离」代理，粗糙地面上不如直接高度精准；若调权后蹲伏仍在，叠加此项：
- **P4L stage 语义**：0=approach+前脚压墙、1=后脚、2=落地（已有 h_lo=0.15 单侧 shaping）、
  3=sentinel 返程。选 0/3 正好补两个未覆盖阶段。
- **高度基准**：stage0 用 measured_heights 机身正下/偏后采样点均值作局部地面（避前点墙遮挡）；
  stage3 复用 `_wococo_landing_base_height()`（地板基准）。**不用绝对 root_z**。
- **单侧过低罚** `scale×clamp(target−h, min=0)`：stage0 末段须起身压墙、stage3 入口是弹道弧，
  双侧罚会打架（现有 L4839 注释明确警告罚高度伤蹬离弧）。
- **stage0 近墙豁免** ~0.4–0.5m；目标 stage0=0.25 / stage3=0.18–0.20；scale stage0 −0.5~−1.0 /
  stage3 −0.5（线性 shortfall，典型 0.05m→预算 1.5–3%）。cfg 开关默认 0→pop，仅 env19/15，不 gate armed。

### 13.7 当前随机化快照（全满，dr_scale≡1.0 从 step0）
- 创建时：摩擦 U[0.6,2.0]（64 桶）、附加质量 +[0,3]kg、质心 ±0.2m/轴、电机强度 ×[0.8,1.2]。
- 每 episode：PD kp/kd ×[0.95,1.05]、电机零位 ±0.035rad、restitution U[0,0.5]。
- 固定 20ms 动作延迟（start_latched，step0）；**无** per-env 动作/本体延迟随机。
- 本体噪声 noise_level=1.0：rotation 0.05 / dof_pos 0.01 / dof_vel 0.05 / lin_vel 0.05 /
  ang_vel 0.05 / gravity 0.02 / height 0.02（+高度量化）；privileged 干净。
- 地形：行均匀随机冻结（curriculum=False）、地面 roughness 固定 0.06+p=0.5 光滑（**§14.3 已改为
  50% 全平 + 50% U(0.02,0.06)**）、墙角
  lerp(30,75,d)±5°、走廊/栏位 ±0.50m、栏数 zero_prob 0.25、栏宽 U[1.0,1.3]、栏顶 U[0,0.01]、
  delta_x lerp(0.20,0.65,d)、g5 间隙 1.70±0.30。
- 指令：墙道双峰，开头随机站立 0–4s→跑（lin_vel_x U[0,1]）→终点硬停。
- **关闭**：push_robots、接触阈值随机、初始位姿随机（pos/yaw/pitch/y）、深度 DR（teacher 无相机）。

### 13.8 一句话与下一步
**这轮证明**：it10000 蹲伏是「运动正则涨到 23% + 姿态锚缩到 3.7%/orientation 死」逼出的省能耗
局部最优，不是肘撑地、不是 FSM 卡死。**下一轮**：部署 §13.5 权重包到 bayesdl，从 it10000 续
+1000~2000 做 A/B（对照 it10322 基线），盯运动组 share↓ / dof_error share↑ / term_roll /
base_contact / per_level/L9/success（基线 5.4%），再录 L9 + root_z CSV 看 approach 高度是否抬升；
蹲伏仍在则叠加 §13.6 base_height 项。改权重会移动奖励地形，头 ~100 iter 盯 value_function loss。

## 14. sim2sim 对比 + 背墙改宽 + 粗糙度纠正 + it9000 肘/膝触地诊断（2026-11-16）

### 14.1 sim2sim 成功率对比（MuJoCo 免 ROS 栈，姊妹仓 go2-sim2sim2real）
入口 `lidar_runner.py`（privileged teacher、raycast scandots、753-D obs、`--scandot_grid legacy`）。
40 trial = 2 ckpt（t3200 `model_3200` / np1 `model_8000`）× 2 scene（`flyball50`≈L4/50°、
派生 `flyball75_2h_h10_aligned`≈L9/75°）× 10 seeds，frontfeet 闭环判据（`p4l_complete`：前脚触墙
≥10N）；另跑 stage2 对照。

| 引擎/口径 | 场景 | t3200 | np1 it8000 |
|---|---|---|---|
| Isaac A（干净确定性） | L4 | 99.9% | 92.4% |
| Isaac B（全 DR，金标准） | L9 | 3.2% | **57.6%** |
| MuJoCo sim2sim | f50(≈L4) | 9/10 | 7/10 |
| MuJoCo sim2sim | f75(≈L9) | 3/10 | 5/10 |

**判读**：越干净/简单 t3200 越好，越难/越随机 np1 越好 → 引擎间排序**方向一致**，佐证 B 臂
（DR 鲁棒性收益）跨引擎成立、非 Isaac 过拟合。np1 f50=7 vs t3200=9 在 10 seeds 下不显著；
失败集中在**下墙返航段**（摔倒点 x≈4.8–5.5、带侧漂、goal_idx=2），上墙两段两 ckpt 都近满分。
**建议**：把 B 臂 + sim2sim 双口径定为版本对比金标准，A 臂只用于出视频。

### 14.2 背墙改宽 = 斜坡宽 + it9000 默认 play 录制
`flyball_wall.py` 背墙切片由全 tile 宽 `[:, :]` 改为与斜坡掩码等宽的 `[:, _plate_y_lo:_plate_y_hi]`
（真实飞球道背墙结构与斜坡同宽）。地形缓存 checksum 含 flyball_wall.py 源码 → 自动失效重建。
scp ac run 最新 `model_9000.pt` 本地，用 isaacgym 默认 play 口径（DR-off、确定性、L9、1 env×1000 步、
960×544 headless、**不启用 `--use_camera`**）录制 →
`teacher-maxdr9rw-np1-ac-warm8000_ckpt9000.mp4`，目检确认背墙变窄、闭环无异常终止。

### 14.3 地面粗糙度纠正：50% 平地不变 + 粗糙半 → U(0.02, 0.06)
用户澄清「平地不变，0.06 改成随机」：保持 `flyball_roughness_zero_prob=0.5` 的 coin flip（50% tile
完全平），把粗糙那一半从固定 0.06 松开成逐 tile 随机。
- `go2_teacher_maxdr_config.py`：`height = [0.06, 0.06]` → **`[0.02, 0.06]`**。`geometry_at_max_difficulty=True`
  下 `rough_h = np.random.uniform(height[0], height[1])` = U(0.02,0.06)。
- 回退误加的 `flyball_roughness_zero_range` 三处（flyball_wall.py 参数声明 + coin flip 分支恢复
  `rough_h=0.0`、terrain.py 透传行、maxdr config 旋钮）；全仓 `roughness_zero_range` 0 命中。
- 最终分布 = 50% flat + 50% U(0.02,0.06)；粗糙上限仍 0.06 → worst-case 网格不变，`max_error=2.0` 仍安全。
- **状态：仅本地改 + py_compile 通过；未同步 bayesdl、未重启训练**（部署/续训待用户指令）。

### 14.4 it9000 肘/膝触地诊断（EP_KNEE_DIAG，更新 §13.8「不是肘撑地」）
给 play.py 加 `EP_KNEE_DIAG=1` 门控诊断（默认关闭、不改 play 行为）：逐步统计 thigh/calf（"肘/膝"）
碰撞，并用 **calf 连杆原点（即膝关节）世界 z − 正下方地形采样 z** 区分「真·砸平地」vs「压墙/摆动
蹭碰」——墙与地面同属一个 heightfield，net contact force 无法按对象分辨，故必须走几何判据。

16 envs×600 步、DR-off、确定性、L9：
| 指标 | 值 |
|---|---|
| 任意 >10N thigh/calf 碰撞 | 6/600 步（1.0%），峰值 638 N |
| **膝关节真砸平地**（离地 <0.09 m） | **3/600 步（0.5%），全 FL_calf**（215/587/638 N，命中时 root_z 0.23–0.29） |
| 小腿悬空蹭碰（离地 0.14–0.16 m） | 3 次（FR_calf×2、RR_calf×1） |
| 至少碰一次的 env | 5/16 |
| 窗口内摔倒终止 | 0（12 s < 60 s episode） |

**结论**：ac 调权确实显著压低了下蹲，但**残余偶发的左前膝磕地仍在**，且发生在**平地接近段**
（terr_z=0、ep_step 229–410 ≈ 4.6–8.2 s），不是压墙时的正常小腿接触。样本仅 16×12 s 属定性证据；
**下一轮**：跑 `--max_steps 3000` + `--full-dr` sweep 给出稳定触地率，据此决定是否叠加 §13.6
base_height 项或针对左前腿加下蹲惩罚 / 调 calf 接触阈值。

**运维坑（本轮踩到并修复）**：裸 python 缺 `LD_LIBRARY_PATH=$ENV/lib`（libpython3.8.so）；gymtorch
加载需 ninja 在 PATH；torch_extensions 缓存在只读盘 → 需可写；isaacgym 必须先于 torch 导入；
`terrain_utils` 来自 `from isaacgym import terrain_utils`（非 legged_gym.utils）。

## 15. sim2sim 窄背墙定案 + student 基线（2026-11-17）

### 15.1 窄背墙口径修正与 teacher 三臂重跑

§14.2 把**训练侧** `flyball_wall.py` 背墙切片改窄后，发现 sim2sim 场景 XML 的背墙与训练口径
不符：`scene_flyball50.xml` / `scene_flyball75_2h_h10_aligned.xml` 的 `flyball_wall_back` y 半宽
均为 2.0（4 m 全宽），而训练侧背墙 = 斜坡等宽（1.3 m，半宽 0.65）。f50 的 XML 注释早已写着
0.65 而实体是 2.0（注释与实体长期不一致）。两场景 y→0.65 后重跑三臂
（`sweep_2026-11-17-narrowback_s30`，30 seeds × 2 场景 × 3 ckpt，脚本 `.tmp_sim2sim_narrowback.sh`）：

| ckpt | f50 strict | f75 strict | 汇总 |
|---|---|---|---|
| m3200 | 27/30 | 17/30 | 44/60 |
| np1it10000 | 23/30 | 21/30 | 44/60 |
| r2it14000 | 26/30 | **28/30** | **54/60** |

Fisher f75：r2 vs m3200 p=0.0021、r2 vs np1 p=0.0419，双 SIG → r2 最强结论在窄背墙口径下成立。
宽→窄成功率变化 ≤1 seed/格：背墙宽度不是决定因素，旧口径结论全部稳健。
f50 的 fail@2（返航段）计数仍是陡坡特化 signature（m3200 0 / np1 7 / r2 4）。

### 15.2 student 基线 sweep 作废事故：loader 吃到的是冻结的 DAgger teacher

用同一脚本跑 `student/model_10000.pt`（tag `studentit10000`，60/60 完成）后发现统计数字与
m3200 **逐 trial 逐字段完全相同**（连 tau_peak 126/131 都一样）。排查链：

1. `diff` 两个 tag 的 per-trial RESULT 行：除 tag 名外 bit-identical（固定 seed + 确定性 MuJoCo
   下只有同一个 policy 才能复现到三位小数）。
2. 两个 tag 的日志都显示加载了正确的不同文件（student 46 MB / teacher 9.7 MB）。
3. **根因**：`lidar_runner.py` 只从 `model_state_dict` 提取 `actor.*`（ActorCritic.actor）。
   DAgger student ckpt 的 `model_state_dict` 里躺着的正是训练它时冻结的 teacher actor ——
   python 逐张量对比证实：student/model_10000.pt 与 teacher/model_3200.pt 的 35 个键
   **完全相同**（tensors differing: 0）。student 真正的 deployable 本体在顶级键
   `depth_actor_state_dict`（26 键）+ `depth_encoder_state_dict`（18 键），runner 从不读取。

→ 60 个 trial 是 m3200 的完美重放，基线作废。**教训：任何 DAgger student ckpt 喂给只认
`actor.*` 的部署 runner 都会静默评测 teacher；两者 md5 不同但行为逐位相同**。

### 15.3 lidar_runner.py 新增 `--policy depth` 模式

从姊妹仓 `distill_runner.py`（旧 student 的部署镜像，1231 行）逐段移植进 `lidar_runner.py`
（16 处补丁，脚本留档 `.tmp_patch_lidar_depth.py`；默认 `--policy teacher` 行为逐字节不变）：

- **加载**：depth 模式下 actor ← `depth_actor_state_dict`；estimator 不变；另建
  `DepthOnlyFCBackbone58x87` + `RecurrentDepthBackbone` ← `depth_encoder_state_dict`。
  从 `output_mlp.0.weight.shape[0]` 推断 32（no-yaw）vs 34（+2 yaw），经 `EP_USE_YAW`
  环境变量切 encoder 输出头；model_10000 实测 32-output no-yaw。
- **渲染**：MuJoCo `front_depth_cam`（go2.xml 定义在 base_link，pitch 20° 下俯，fovy 56.49，
  随动）106×60 EGL 深度渲染 → `clip_mujoco_too_close_to_far`（= Isaac empty_below_m 重映射的
  sim 侧替身）→ `normalize_depth`（crop [:-2,4:-4] → clip [0,2] → bicubic 58×87 → /2-0.5，
  与 Isaac process_depth_image 逐位对齐）。`mujoco_depth_scene_option` 剔除机器人自身 geom
  = 训练的 terrain-only raycast（self_geometry 等价）。
- **推理**：每 `update_interval=5` tick（10 Hz）渲染+encode，`--depth_latency_s 0.08`
  → 4 tick 交付延迟（pending 队列，训双边 latency[4,5] 的下限口径）；latent 经
  `scandots_latent=` 旁路 scan_encoder（obs 的 scan 段置 0）；`hist_encoding=True` 保持
  history 路径；no-yaw student 的 prop[6:8] 恒 0（goal FSM 的 dy/dny 不进 obs，FSM 只驱动
  cmd_vx 与 RESULT 记账）；`MUJOCO_GL=egl` 在 depth 模式自动设置。
- 冒烟 1 trial full-loop 通过；重跑 sweep 用 `.tmp_sim2sim_narrowback_depth.sh`
  （头部显式 `export MUJOCO_GL=egl`）。速度意外快：60 trial 5.5 min（成功 trial 提前终止
  + EGL 深度渲染开销可忽略）。

### 15.4 studentd10000 真实基线与失败模式

| 口径 | f50 | f75 |
|---|---|---|
| **studentd10000 strict**（returned & upright & p4l_complete） | **15/30** | **23/30** |
| studentd10000 loose（returned & upright） | 16/30 | 28/30 |
| wall（p4l_complete） | 26/30 | 25/30 |
| tau_peak max | 116 | 87 |

- **假 full 陷阱**：strict vs loose 的差 = 没碰墙却触发 return-x 的 trial（f50 1 个、
  f75 5 个；teacher 三臂 fake-full 全 0）。return-x 的 arm(x≥2.0)/stop(x≤0) 独立于墙 FSM，
  机器人走歪绕开墙再回来也会 `returned=1`。**以后 student 类 ckpt 的 full-loop 必须
  加 p4l_complete 联合判定**。
- **f50 的 15 个失败里 14 个是导航走丢，不是运动失败**：13 个直立存活（roll 50–70°）
  却朝 +y 漂移 3.8–9.3 m 收不回来；唯一真摔是 s24（墙根下墙段 roll 179.9°）。
- **机制**：no-yaw student 没有 heading 命令输入口（训练时 obs[6:8] 被 encoder yaw 头/
  0 覆盖），返航全靠深度视觉识别「墙在身后」；缓坡下墙姿态把朝向带歪后**没有任何通道
  能纠正**（teacher 靠 FSM dyaw/dnext 命令拉回）。f75 陡坡下墙是训练特化强项
  （姿态受控→朝向正→返航成功），故 student f75 反而好于 f50 —— 与 m3200 的镜像同源，
  但丢掉了 teacher 靠 heading 命令兜底的那层保险。
- Fisher（strict）：f50 student 显著差于全部三臂（vs m3200 p=0.0001 / np1 p=0.015 /
  r2 p=0.0006）；f75 与 np1 持平（p=0.38）、差于 r2（p=0.026）、好于 m3200（p=0.054）。
- **蒸馏缺口定性**：不是感知断裂（f75 23/30 说明陡坡特化传下来了），是「导航保险缺失
  + 上墙退化（f75 wall 25/30 vs teacher 30/30/29）」。
- 失败视频（确定性复现，含 meta.json）：`fail_videos_2026-11-17/studentd10000_f50_s02_lost.mp4`
  （走丢样本，fy=+6.79 全程直立）、`..._s24_rollover.mp4`（roll 179.9° 翻车）。

### 15.5 下一步

- **C 臂（原计划）**：等 r3 it20000 → 重新蒸馏。预期继承 r3 的 f75 28/30 + 更强上墙；
  但 no-yaw 导航短板是**结构性的**，重蒸馏救不了 f50 走丢。
- **补导航通道（治本）**：给 student obs 加回 heading 命令（6:8 不清零）→ 改 DAgger
  输入布局 + 重蒸馏。与 C 臂可合并成一次改动。
- runner 的 depth 模式已就位，后续任何 student ckpt（含 stage-B RL 微调产物）都可直接
  用同口径测：`.tmp_sim2sim_narrowback_depth.sh <ckpt> <tag>`。
