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

