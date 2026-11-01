# 实验记录 — stage7 plant 锁死归因 + stage9 验证

日期：2026-10-31
任务：`go2_depth`（depth student，相机 + 单阶段 PPO）
起点策略：`logs/extreme-parkour/stage3-v3/model_0.pt`

---

## 0. 摘要

1. **stage7-lvl4 的 `plant` 锁死是观测侧配置的锅，不是学习能力问题。**
   同一套部署权重（`stage3-v3/model_0.pt`，与 `v3-student/model_5000.pt` 的
   actor / depth_encoder / estimator 逐位相同），换到 stage7 的配置下评估，
   stage-0 → stage-1 通过率 `stage_1_unlocked_frac` 只有 **0.0055**，
   `s0ever_near` 从 `v3-student` 训练环境的 0.739 掉到 **0.0193**。

2. **5 组析因 eval 定位到两个可加的因子**：`depth.self_geometry=False`
   （相机看不见自己的腿）与 `terrain.min_terrain_level=4`（起始地形锁在最难行）。
   两者各贡献约 +0.30，合计把 `s0ever_near` 0.0193 抬到 0.6549、
   `stage_1_unlocked` 0.0055 抬到 0.5781，几乎完全可加。

3. **16 项深度退化里 `self_geometry` 一项独占几乎全部恢复**：
   只开 `self_geometry`（C 组，`stage_1_unlocked` 0.3937 / `s0ever_near` 0.3377）
   ≈ 恢复完整退化集（B 组，0.3365 / 0.3291）。

4. **但按最优组合重训的 stage9 失败了**：从 `s0ever_near` 0.6549 起跑，2000 iter 后
   确定性评估跌到 **0.0436**（−15×），`plant_complete` 归零，`frac past g0`
   0.3051 → 0.1120。训练过程本身在退化。

5. **训练日志里 `stage_*_unlocked` 类是 fleet 粘性累计，不能当通过率读**：
   `wococo/stage_{1..4}_unlocked_frac` / `plant_complete_unlocked_frac` /
   `sentinel_unlocked_frac` 用 `|=` 累积（`legged_robot.py:1528-1536`），
   含义是“480 个 env 里有多少个曾经成功过一次”，永不回落。stage9 的
   `stage_1_unlocked` **在 iter 100 就是 0.0938**，到 1900 iter 只爬到 0.0979 ——
   不是“训练在进步”，而是“再没多一个 env 成功”。真正的 per-episode 指标
   `s0ever_near` 全程 ≈ 0（2.5 万个 episode 里没有一个把双前脚落到 foothold 球域）。
   所以在两个口径对齐前，**训练日志不能当策略能力的度量，只能用确定性 eval**。

6. **排除清单继续加长，env 数这一层被重新量化**：`dagger_update_freq`
   （depth PPO 硬编码 `hist_encoding=True`，不适用）、`--delay`（F1/F2 差异在
   噪声内）、DR 斜坡（两个口径下 `dr_scale` 都是 0）、depth latent 的 10 Hz tick
   （I1/I2）**全部不成立**。env 数 128 → 480 在“原口径”下确实掉（0.578 → 0.333），
   但把 eval 换成**训练环境的精确镜像**（10 Hz tick + action_delay + DR prearm）
   后，480 env 又回到 **0.5021**（L2）、128 env 回到 **0.6069**（L1）——
   所以“eval 口径虚高”这个解释本身也不成立，G 组的腰斩有一半是口径假摔。

7. **结论反转为：训练真的在破坏起点能力。** 起点权重在四种 eval 口径下
   都保有 `stage_1_unlocked` 0.50–0.61（D 0.5781 / L1 0.6069 / G 0.3328 /
   L2 0.5021），而**任何**训练——depth 通道（stage9、t11 冻结 encoder、
   t13 降噪声）、scan 通道（t10 teacher）——都在 100 iter 内把它压到 0.1 以下。
   最硬的一条：teacher 通道训练侧粘性指标能到 st1 **0.9208** / plant **0.7479**，
   同一权重的确定性 eval 只有 st1 **0.0669** / plant **0.0098**。
   **训练度量与确定性执行严重背离，且两种观测通道、冻结/不冻结 encoder、
   高噪声/低噪声都一样**——所以根因在奖励/训练信号，不在相机、不在 encoder、
   不在探索噪声。

8. **（11-01 凌晨，已破案）根因是 σ=0.57 探索噪声 × lr=3e-4 的组合；
   t15 用 σ0.10/cap0.15 + lr3e-5 训 200 iter，确定性 eval 首次超过起点**
   （st1 **0.6537** / plant **0.5933** / s0ever 0.7079，起点 L1 是
   0.6069/0.4876；训练侧 st1 0.971 / plant 0.892）。
   机制：高 σ 下 PPO 优化的是随机策略期望，均值漂到“顶墙、靠噪声抽奖”
   照样拿分（t14 实锤：同权重确定性 0.023 vs 加噪 0.152，×6.6）；
   单降 lr 不改变方向（t14 仍崩），单降 σ 会让 KL≈Δμ²/2σ² 爆炸（t13 更差），
   **两者必须同降**。详见 §6.5。

---

## 1. 统一评测口径

| 项目 | 取值 |
|---|---|
| task / 脚本 | `go2_depth`，`/tmp/eval_policy3.py` |
| CLI | `--use_camera --headless --num_envs 128 --no_wandb` |
| rollout | 3000 步，`act_inference`（确定性），无 reset 上限裁剪 |
| depth 通路 | estimator(prop) → `priv_explicit`；encoder(depth, prop) → `scandots_latent`；`hist_encoding=True` |
| 地形 | cache checksum `1984689146`（r10×c80，hs 0.01） |
| 天气/DR | 沿用 cfg（未加 `--no-dr`） |

评测脚本通过 `EP_EVAL_CFG_OVERRIDE`（JSON）在 `make_env` 之前覆盖 `env_cfg`，
通过 `EP_EVAL_ACT_NOISE` 给动作叠加 `σ = ACT_NOISE × ac.std` 的高斯噪声
（`ac.std` 在 actor 中是 std 本身，非 log）。

**指标含义**

- `stage_1_unlocked_frac` — **stage-0 的真实通过率**。stage-0 的推进条件是
  `双后脚在 liftoff AABB 且 force > stance_threshold` + `base z > 0.20` +
  `双前脚腾空` + `approach_vel ≥ max(0.5·cmd, 0.3)` + `cur_goal_idx ≥ wall_goal_idx`
  （env19 为 g2、env15 为 g4）。**它不看 `near`。**
- `s0ever_near` — 一个 episode 内是否**至少一次**双前脚同时落在各自 pre-hurdle
  foothold 目标点的球邻域内（半径 0.198–0.500 m，按地形行插值）。
  **纯诊断量**：它只在 `_wococo_contact_pts_p3` 的 s0 diag 统计与 s1/s2 的
  `correct_now` 里出现，stage-0 的奖励与推进条件都不引用它
  （`legged_robot.py:4937-4965` 用的是 `_in_region(rear_pos, liftoff_region)`）。
  因此 `s0ever_near` 与 `stage_1_unlocked` 高度相关但不互为因果。
- `s0_near_frac` — 上述条件成立的**步**占比。
- `s0_press_frac` — 双前脚与 hurdle 墙面 hard contact 的步占比。
- `plant_complete_unlocked_frac` — plant 全阶段完成并解锁 g0 的 episode 占比。
- `frac past g0` — `cur_goal_idx >= 1` 的 step 占比。

**为什么用同权重多点评估**：`stage3-v3/model_0.pt` 与
`v3-student/model_5000.pt` 的 `depth_actor_state_dict`（26/26 keys）、
`depth_encoder_state_dict`（18/18）、`estimator_state_dict`（6/6）
逐位相同（max |Δ| = 0）。因此"同一策略在不同 cfg 下的表现"是干净的对照。

---

## 2. 2×2 析因：观测退化 × 起始地形

固定权重 `stage3-v3/model_0.pt`，其余全部相同。

| 组 | `self_geometry` | `min_terrain_level` | `stage_1_unlocked` | `s0ever_near` | `frac past g0` |
|---|---|---|---|---|---|
| **A** | False（stage7 现状） | 4 | **0.0055** | **0.0193** | 0.0624 |
| **C** | True | 4 | **0.3937** | **0.3377** | 0.3043 |
| **E** | False | 0 | **0.3582** | **0.3070** | 0.2169 |
| **D** | True | 0 | **0.5781** | **0.6549** | 0.3051 |

主效应（`s0ever_near`）：

- `self_geometry` False→True：lvl4 下 +0.318，lvl0 下 +0.348 → 均值 **+0.333**
- `min_terrain_level` 4→0：clean 下 +0.288，selfgeo 下 +0.317 → 均值 **+0.303**
- 可加预测：0.0193 + 0.333 + 0.303 = **0.625**；实测 D 组 **0.6549**。
  → 两个因子基本无交互，纯叠加。

主效应（`stage_1_unlocked`，与上表同向）：

- `self_geometry`：+0.388（lvl4）/ +0.220（lvl0）
- `min_terrain_level`：+0.353（clean）/ +0.184（selfgeo）
- 可加预测：0.0055 + 0.304 + 0.269 = 0.578；实测 **0.5781**（拟合极好）

参照：`v3-student` 在其自身训练环境后期的 `s0ever_near` ≈ 0.739，
落在 D 组附近（其 `stage_1_unlocked` 未单独记录）。也就是说
**A（stage7）是这套权重所有观测组合里最坏的一格**。

### 2.1 完整指标表

| 组 | g0 | armed | max_goal | max_stage | depth均值 | s0ever | s0near | s0press | s0roll | s0rear | s0hold | s1act | s1ever | stmax | plant | term_base | term_roll | term_timeout | term_lateral | rew_track |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A clean/lvl4 | 0.0624 | 0.0214 | 1 | 0 | 0.0905 | 0.0193 | 0.0004 | 0.1153 | 0.2222 | 0.3262 | 0.0000 | 0.0002 | 0.0000 | 0.0007 | 0.0000 | 0.2049 | 0.5983 | 0.0049 | 0.1897 | 0.1223 |
| C selfgeo/lvl4 | 0.3043 | 0.1418 | 3 | 3 | 0.0223 | 0.3377 | 0.0060 | 0.0991 | 0.4416 | 0.1900 | 0.0026 | 0.1436 | 0.1189 | 0.2887 | 0.1986 | 0.5806 | 0.0932 | 0.0422 | 0.1588 | 0.2885 |
| B fulldeg/lvl4 | 0.3047 | 0.1178 | 2 | 3 | 0.1567 | 0.3291 | 0.0054 | 0.0904 | 0.4228 | 0.2127 | 0.0026 | 0.1541 | 0.1363 | 0.2687 | 0.1709 | 0.4695 | 0.0457 | 0.0590 | 0.2490 | 0.2943 |
| E clean/lvl0 | 0.2169 | 0.0954 | 2 | 3 | 0.1011 | 0.3070 | 0.0173 | 0.3398 | 0.9378 | 0.1747 | 0.0038 | 0.2196 | 0.2043 | 0.4127 | 0.2239 | 0.1245 | 0.0805 | 0.0637 | 0.6987 | 0.2368 |
| D selfgeo/lvl0 | 0.3051 | 0.1995 | 2 | 2 | 0.0252 | 0.6549 | 0.0396 | 0.3183 | 0.9392 | 0.1653 | 0.0085 | 0.4589 | 0.4409 | 0.7657 | 0.2982 | 0.3152 | 0.0594 | 0.0919 | 0.4976 | 0.2664 |

**行为指纹**

- A 组：`term_roll = 0.598`（六成 episode 是躺倒收场），`s0press 0.115`、
  `s0roll 0.222`，`max_stage = 0` —— 走不到墙前就已经翻倒。
  这与 stage7 训练曲线的 `tracking_goal_vel_moving 0.028 → 0.002`、
  `standstill 0.105 → 0.126` 一致：PPO 只能收敛到站桩。
- D 组：`s0roll 0.939`、`s0press 0.318`，`s1_active 0.459`，
  `max_stage = 2` —— 姿态稳定、能顶墙、能连续过 2 个门。
- 有意思的是 **E 组 `term_lateral = 0.699`** 远高于 D 组 0.498：
  看不见自己的腿时，机器人靠横向漂移"摸"过去，代价是被 lateral 终止。

---

## 3. C vs B：16 项退化做减法

C 组只覆盖 `{"depth.self_geometry": true}`；B 组覆盖 16 项
（`self_geometry / empty_below_m=0.15 / hole_fill / latency_range=[4,5] /
pw_noise / pw_hole_prob=0.007 / pw_edge_prob=1.0 / pw_blur_sigma / dis_noise /
noise_gaussian / noise_dropout / stereo_shadow / hole_fill_backend=gpu /
hole_fill_upscale=false / hole_fill_height=270 / hole_fill_width=480`）。

| | `stage_1_unlocked` | `s0ever_near` | `frac past g0` | `plant` | depth 均值 |
|---|---|---|---|---|---|
| C（只开 self_geometry） | 0.3937 | 0.3377 | 0.3043 | 0.1986 | 0.0223 |
| B（完整退化集） | 0.3365 | 0.3291 | 0.3047 | 0.1709 | 0.1567 |

**结论**：C ≈ B（差异在单次 3000 步的噪声内，且 C 略优）。
`hole_fill`、`latency`、`pw_noise`、`stereo_shadow` 等 15 项加起来
也不及 `self_geometry` 一项。深度图均值差异巨大（0.022 vs 0.157）但行为一致，
进一步说明 **决定 plant 的信息不是"图更好看"，而是"图里有自己的腿"**。

---

## 4. stage7 训练侧证据（回归背景）

`logs/extreme-parkour/stage7-lvl4`，2000 iter，480 envs，10:20 → 11:57（97 min）。

| iter（近似） | episode_len | s0_press | s0_roll_ok | s0ever_near | plant |
|---|---|---|---|---|---|
| 早期若干 | 16.6 → 125.6 | 0.062 → 0.087 | — | 0.0000 | — |
| 末段 | 787 → 834 | 0.139 → 0.214 | 0.42 → 0.54 | 0.0000 | 0.0521 |

`terrain_level` 全程钉在 `4.0000`。`Mean action noise std` 全程 0.57
（与 v3-student 相同，排除探索不足）。全程 `s0ever_near = 0`。

---

## 5. stage9：按最优组合重训

### 5.1 配置

```bash
EP_TRAIN_CFG_OVERRIDE='{"depth.self_geometry": true, "terrain.min_terrain_level": 0}' \
python /tmp/train_stage9.py --task go2_depth --use_camera --delay --headless \
  --num_envs 480 --max_iterations 2000 --no_wandb \
  --proj_name extreme-parkour --exptid stage9-selfgeo-lvl0 \
  --resume --resumeid stage3-v3
```

`/tmp/train_stage9.py` 通过 monkey-patch `task_registry.get_cfgs` 在 `make_env`
之前注入覆盖，**仓库代码零改动**。日志确认生效：

```
[stage9] override depth.self_geometry: False -> True
[stage9] override terrain.min_terrain_level: 4 -> 0
[Terrain] CACHE HIT (0.7s) checksum 1984689146: .../r10xc80_hs0.01_eeb68991a573.pkl
Loading model from .../stage3-v3/model_0.pt...
Saved depth encoder detected, loading...
```

耗时：2000 iter / 4345 s（**72 min**，2.17 s/iter）。产物
`model_0 / 500 / 1000 / 1500`（`model_0.pt` 在 17:55 被收尾 save 覆盖为最终权重）。

### 5.2 训练曲线

⚠️ 两类指标的聚合口径完全不同，必须先分清：

- **fleet 粘性累计**（只增不减，跨 episode 累积，永不回落）：
  `wococo/stage_{1,2,3,4}_unlocked_frac`、`wococo/plant_complete_unlocked_frac`、
  `wococo/sentinel_unlocked_frac`。实现见 `legged_robot.py:1528-1536` 的
  `self.wococo_stage_unlocked[env_ids, i] |= (stage_pre_reset >= i)`、
  `:1646-1652` 的 `self.wococo_stage_unlocked[:, i].mean()`。
  **它的含义是「训练开始以来，480 个 env 里有多少个 env 至少成功过一次」，
  不是当前通过率。**
- **per-episode 窗口**（只看本步刚 reset 的那批 env）：
  `wococo/s0ever_near`、`s0_near_frac`、`s0_press_frac`、`s0_roll_ok_frac`、
  `wococo/max_goal_idx`、`wococo/stage_max`。
- **滑动窗口**：`episode_len`、`Reward/`、`Mean action noise std`。

| iter | len | terrain_lvl | s0ever_near | s0near | s0press | s0roll | st1 † | plant † | max_goal | σ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 17 | 1.65 | 0.0000 | 0.0000 | 0.0775 | 0.2846 | 0.0000 | 0.0000 | 0.0000 | 0.57 |
| 100 | 870 | 0.02 | 0.0000 | 0.0000 | 0.3101 | 0.9985 | **0.0938** | **0.0229** | 0.0833 | 0.57 |
| 300 | 930 | 0.00 | 0.0208 | 0.0001 | 0.4417 | 0.9989 | 0.0938 | 0.0229 | 0.2500 | 0.57 |
| 600 | 907 | 0.00 | 0.0000 | 0.0000 | 0.4071 | 0.8300 | 0.0938 | 0.0229 | 0.1667 | 0.58 |
| 900 | 872 | 0.00 | 0.0000 | 0.0000 | 0.4851 | 0.9832 | 0.0938 | 0.0229 | 0.2500 | 0.58 |
| 1200 | 943 | 0.00 | 0.0000 | 0.0000 | 0.4468 | 0.9238 | 0.0958 | 0.0229 | 0.1250 | 0.58 |
| 1500 | 890 | 0.00 | 0.0000 | 0.0000 | 0.4348 | 0.9956 | 0.0958 | 0.0229 | 0.1875 | 0.58 |
| 1800 | 734 | 0.00 | 0.0000 | 0.0000 | 0.4035 | 0.8194 | 0.0958 | 0.0229 | 0.0208 | 0.58 |
| 1900 | 863 | 0.00 | 0.0000 | 0.0000 | 0.4564 | 0.9948 | 0.0979 | 0.0229 | 0.1736 | 0.58 |

† 粘性累计，不可当通过率读。

真实情况是：**`stage_1_unlocked` 在 iter 100 就已经是 0.0938**（不是我上一版写的
“iter 1311 才抬头”），之后 1900 iter 只从 0.0938 爬到 0.0979 —— 也就是说
**480 个 env 在 53 个 episode 的机会里，只有 47 个 env 成功过一次 plant**。
`plant_complete` 更死：iter 100 的 0.0229（11 个 env）到末期**一个都没增加**。

相对 stage7 的改善是真实的：episode 长度 16 → 900、`s0_press` 0.46–0.58（stage7 末段 0.21）、
`s0_roll_ok` 0.94–1.00（stage7 0.42–0.54）。但 per-episode 的
`s0_near` / `s0ever_near` **全程严格 0**，`stage_max` 从未离开 0。

对照确定性 eval（同配置）：

- 起点权重（128 env / 3000 步）：`stage_1_unlocked` 0.5781、`s0ever_near` 0.6549
- 起点权重（480 env / 2500 步，G 组）：`stage_1_unlocked` 0.3328、`s0ever_near` 0.3593
- 训练末期（128 env / 3000 步）：`stage_1_unlocked` 0.0436、`frac past g0` 0.1120、`s0ever_near` 0.0436

**两边的差距现在有了量级**：训练给了每 env 53 个 episode 的机会，只有 9.8% 的 env
成功过一次；eval 只给 3.3 个 episode，128 env 里有 55.8% 成功。**这是环境级差异，
不是统计噪声。**

### 5.3 确定性评估（D 组同款 cfg）

| ckpt | s0ever_near | frac past g0 | s0near | s0press | plant | max_stage | s1act | term_base | term_roll | term_timeout | depth均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 起点（= D 组） | **0.6549** | 0.3051 | 0.0396 | 0.3183 | **0.2982** | 2 | 0.4589 | 0.3152 | 0.0594 | 0.0919 | 0.0252 |
| stage9 iter1500 | 0.0000 | 0.0000 | 0.0000 | 0.8243 | 0.0000 | 0 | 0.0000 | 0.2497 | 0.0000 | 0.6307 | −0.0229 |
| stage9 final | 0.0436 | 0.1120 | 0.0123 | 0.4581 | 0.0000 | 0 | 0.0023 | 0.0627 | 0.0012 | 0.2692 | 0.0096 |

**训练把 0.6549 打成了 0.0436，plant_complete 从 0.298 归零。**
iter1500 更极端：`s0_press 0.824`（一直在顶墙）+ `term_timeout 0.631`
（从来不做决定）+ `term_roll 0`（稳稳当当）——
典型的"学会了原地顶墙、学会了不摔倒，但没有 plant"的局部最优。

---

## 6. 噪声判定实验 + 未解矛盾

### 6.1 同权重、只改动作噪声

`EP_EVAL_ACT_NOISE = k`，动作 `= act_inference + N(0, (k·σ)²)`，σ ≈ 0.58（训练值）。

| 组 | ACT_NOISE | 有效 σ | s0ever_near | g0 | s0near | s0press | plant | s1act | stmax | term_base | term_roll |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 0（确定性） | 0 | **0.6549** | 0.3051 | 0.0396 | 0.3183 | 0.2982 | 0.4589 | 0.7657 | 0.3152 | 0.0594 |
| H1 | 1.0 | 0.58 | **0.3972** | 0.1689 | 0.0267 | 0.2330 | 0.1130 | 0.1352 | 0.2113 | 0.1894 | 0.1487 |
| H2 | 0.3 | 0.174 | **0.6267** | 0.2850 | 0.0438 | 0.3053 | 0.2711 | 0.4138 | 0.6577 | 0.2701 | 0.0765 |

**结论**：训练同款噪声确实显著损伤 plant（0.655 → 0.397，−39%，
`plant_complete` 0.298 → 0.113），同时把 `term_roll` 从 0.059 推到 0.149。
但噪声**压不到 0**，所以它解释“为什么学得慢”，不能单独解释 stage9 的失败。

### 6.2 未解决：训练度量与确定性 eval 对不上

已排查（均排除）：

| 假设 | 检查 | 结果 |
|---|---|---|
| foothold 目标未生成 / 全零 | 读地形 cache，`flyball_foot_target` shape (10,80,4,3) | **每行 80/80 tile 非零**，row0 前脚目标 z=0.10–0.23 m 合理 |
| 判定半径不一致 | diff 训练 vs eval 的 `[wococo]` 半径表打印 | **完全一致**（r_wall 0.198–0.500，per-angle 11 行） |
| schedule 没打开 | `dr_ramp_armed` / `hurdle_clearance/armed` / `armed_term_exempt_armed` | 从 iter 1 起全是 1.0 |
| 地形行太简单无墙 | row0 `s0_press` 0.46 | 有墙 |
| 姿态不达标 | `s0_roll_ok` 0.94–1.00 | 达标 |
| 起点权重本身不行 | D 组 eval 0.6549 | 行 |
| 动作探索噪声 | H1 组（σ=0.58）0.397 | 显著损伤但非零 |
| 短 episode 污染 | iter 0 length 17.04，之后 800+ | 只解释 iter 0 |
| `stage 0` 奖励依赖 `near` | 读 `legged_robot.py:4937-4965` | **不依赖**：`near` 只是 diag；stage-0 用 `_in_region(rear, liftoff_region)` |

**关键的口径事实**（读 `legged_robot.py:1470-1536, 1610-1652` 确认）：

- `wococo/stage_*_unlocked_frac` / `plant_complete_unlocked_frac` /
  `sentinel_unlocked_frac` 是 **fleet 粘性累计**（`|=` 只增不减，跨 episode 不清零），
  读作“多少 env 曾经成功过一次”，**不是当前通过率**。它永远不会回落，
  所以 stage9 的 0.0958 → 0.0979 不是“几乎没进步”，而是“再没多一个 env 成功”。
- `wococo/s0ever_near` / `s0_near_frac` / `max_goal_idx` 才是 per-episode 窗口，
  训练里全程 ≈ 0 是**真的**：那 2.5 万个 episode 里没有任何一个把双前脚
  落到 foothold 球域内。

**本轮（10-31 下午）新增的三条排除**：

| 假设 | 做法 | 结果 |
|---|---|---|
| `dagger_update_freq=20` 让训练只有 1/20 的 iter 用 `hist_encoding=True` | 读 `on_policy_runner.py:218` 与 `learn_depth_ppo` | **不成立**：`learn_RL` 才用 `it % dagger_update_freq`，depth PPO 在 `:375` 硬编码 `hist_encoding=True` |
| `--delay`（20 ms 动作延迟）造成 train/eval 不一致 | F1：起点权重 + `action_delay` + `start_latched`；F2：stage9 终权重 + 同样设置 | **不成立**：F1 `stage_1_unlocked` **0.5581** / `s0ever_near` 0.6162（无 delay 是 0.5781 / 0.6549）；F2 0.0206 / 0.0352（无 delay 0.0436）。两组都在噪声内 |
| DR（`dr_scale` 斜坡）在训练与 eval 间不同 | 读 env_cfg.pkl | **不成立**：`dr_ramp_full_at_mean_level=6.0`，两个口径下 mean level 都是 0 → `alpha=0` → `dr_scale=0`。`push_robots=False` |

**env 数 + eval 口径的 2×2（L 组是本轮最后一块拼图）**：

| 组 | envs | steps | eval 口径 | `stage_1_unlocked` | `s0ever_near` | `plant_complete` | `frac armed` |
|---|---|---|---|---|---|---|---|
| D | 128 | 3000 | 原口径 | 0.5781 | 0.6549 | 0.2982 | 0.1995 |
| L1 | 128 | 3000 | 训练镜像 | **0.6069** | **0.7126** | **0.4876** | — |
| G | 480 | 2500 | 原口径 | 0.3328 | 0.3593 | 0.1440 | 0.1143 |
| L2 | 480 | 2500 | 训练镜像 | **0.5021** | **0.4565** | **0.3203** | — |

L 组的“训练镜像”= `{"depth.self_geometry": true, "terrain.min_terrain_level": 0,
"domain_rand.action_delay": true, "domain_rand.action_delay_start_latched": true,
"domain_rand.dr_ramp_start_armed": true}` + `EP_EVAL_DEPTH_TICK=1`，
即把 `train.py` 里真正生效的那几个开关和 encoder 更新节奏全部对齐。

读法：

- **横向（env 数）**：同口径下 128 → 480 是 0.6069 → 0.5021（−17%），
  不是原来说的“腰斩”。env 数有影响，但远小于训练带来的落差。
- **纵向（eval 口径）**：480 env 下 0.3328 → 0.5021（+51%）。
  也就是说 G 组的“低”有一半是**逐帧重算 encoder + 无 delay + DR 未 arm**
  造成的假摔 —— tick 不是训练专有的偏差，反而是让 eval 更贴近训练的必要条件。
- **对角线（最关键）**：起点权重在任何口径下都是 **0.50–0.61**，
  而 stage9 终权重是 **0.0436**、t11 冻结 encoder 是 **0.0000**、
  训练侧 iter100 就掉到 0.0938。**没有任何一种口径能把起点压到 0.1。**

**因此“训练度量与 eval 对不上”这条线索到此结案**：它解释了 G 组一半的落差，
解释不了训练本身的退化。真正的矛盾是——

> 起点权重 480 env 下有 0.50 的 plant 通过率，而 PPO 第一个 iter 的更新
> 就让训练侧掉到 0.09，且 2000 iter 都爬不回来。

**剩余可疑点（按优先级）**：

1. **训练在优化一个 eval 不关心的目标**。`stage_max` 始终为 0、
   `term_timeout` 在 iter1500 达 0.63、`s0_press` 达 0.82 —— 
   策略学会了“稳站 + 顶墙”，能拿满无 plant 的密集奖励，
   但在**确定性**下反而不如未训练的起点。奖励与 eval 指标的错位是硬问题。
   这是当前唯一还没被证伪的机制解释。
2. **adv/PPO 的更新幅度**。起点权重是“能用但奖励不高”的策略，
   第一个 minibatch 的 advantage 符号可能整体为负 → 把 actor 推离现有解，
   而奖励地形（顶墙可得分）立刻接住它。t13 把 σ 从 0.58 压到 0.15/0.2 反而更差
   （`s0_press` 0.46 → 0.77）说明不是探索噪声把它推走的。
3. **确定性 eval 与随机策略的分布差**。训练侧 `s0_press` 高但 `s0ever_near`=0，
   说明随机策略在 repeatedly 顶墙；确定性动作是均值，可能正落在
   “顶住不动”这个点上。若如此，则起点权重的 0.5 是**随机性带来的**
   （H1 组 σ=0.58 时 plant 0.298 → 0.113，方向吻合但幅度不够）。

### 6.3 训练通道四路对照（本轮新增）

四个实验的起点都是“能用但 plant 不高”的权重，目的都是回答“是不是某个**训练侧
机制**把起点推坏了”。全部 480 env、`selfgeo + min_terrain_level=0`。
缺省设置：`--resume` 同一份 ckpt，只有表中列出的那一列不同。

| 实验 | 起点 | 唯一改动 | iter | 训练侧 st1 † | 训练侧 plant † | 确定性 st1 | 确定性 plant | s0_press |
|---|---|---|---|---|---|---|---|---|
| stage9 | stage3-v3 | —（基线） | 2000 | 0.0979 | 0.0229 | 0.0436 | 0.0000 | 0.4581 |
| t11 | stage3-v3 | **只冻结** depth_encoder（`requires_grad_(False)`+`eval()`） | 600 | 0.3771 | 0.1479 | **0.0000** | **0.0000** | 0.5012 |
| t12 | stage3-v3 | `latent.detach()`，encoder 只由 aux_loss 更新 | 170 | — | — | — | — | — |
| t13 | stage3-v3 | reset σ=0.15、`max_noise_std`=0.2 | 600 | 0.1479 | 0.0583 | 未测 | 未测 | **0.7671**（训练侧） |
| t10 | teacher-3200（scan 通道） | — | 500 | **0.9208** | **0.7479** | **0.0669** | **0.0098** | 0.5449 |

† 粘性累计（`|=`，只增不减），跨 episode 不清零。

关键读数（三条全部指向同一个方向）：

1. **冻结 encoder 没救，而且更差。** t11 训练侧的 st1（0.3771）比 stage9 同期
   （iter100 的 0.0938；iter500 的 0.0938）高 4 倍，但确定性 eval 是 **0.0000 /
   0.0000**，还伴随 `term_timeout` 0.3077。它把“plant 能力”彻底变成了只有随机
   噪声才能触发的行为。⇒ **encoder 不是被 PPO 梯度破坏的**（否则冻结应当保住起点）。
2. **降探索噪声没救，而且更差。** t13 把 σ 从 0.57 压到 0.15、cap 0.2，训练侧
   `s0_press` 反而从 0.46 涨到 **0.7671**，st1 只有 0.1479。若根因是“噪声把策略
   从门口推走”，降噪应当改善——结果相反。⇒ `s0_press` 不是噪声的副作用，
   **它本身就是策略收敛到的解**。
3. **换到 scan teacher 通道也一样。**

| 组 | 权重 | `stage_1_unlocked` | `s0ever_near` | `plant_complete` | `frac past g0` | `max goal` | `term_base` | `term_timeout` |
|---|---|---|---|---|---|---|---|---|
| T0 | teacher-3200 起点 | 0.2140 | 0.2022 | 0.1247 | 0.3910 | 8 | — | — |
| T1 | t10 model_500 | **0.0669** | 0.4931 | **0.0098** | 0.5465 | 8 | — | — |

T1 比它自己的起点还低（0.2140 → 0.0669，plant 0.1247 → 0.0098），**却比起点走得
更远**（`frac past g0` 0.3910 → 0.5465，`s0ever_near` 0.2022 → 0.4931）。
⇒ 训练确实学会了“走得更靠后”，但**学会了不 plant**。
⇒ “训练度量 vs 确定性执行”的背离不是 depth 特有、不是相机特有、不是 encoder 特有。

t12（detach latent）本来要验证“PPO 梯度穿过 latent 是否让 encoder 漂移”，
两次都在 170 iter 内被 kill，**无结论，不计入证据**。

### 6.4 起点权重的“四口径”定标

见 §6.2 的 2×2 表。要点：

- 起点权重 `stage3-v3/model_0.pt`：128 env 原口径 0.5781、训练镜像 0.6069；
  480 env 原口径 0.3328、训练镜像 0.5021。
- 四种口径的**下界**是 0.3328，是训练 100 iter 就到达的 0.0938 的 **3.5 倍**。
- ⇒ “stage9 失败是因为 eval 口径与训练不一致” —— **证伪**。
  口径确实会 ±50% 地影响绝对值，但改变不了“训练让它掉了 5–10 倍”这个方向。
- 附带修正：`s0_press`（0.27–0.32）和 `term_lateral`（0.43）在 L1/L2 里都稳定，
  说明起点策略有近一半 episode 从**侧向**出界，这一点 earlier 的 D 组没单独记录。

### 6.5 t14/t15：lr 单变量否定 → σ+lr 组合破案（11-01）

**t14（只降 lr：3e-4→3e-5、encoder 1e-3→1e-4、上限 1e-2→1e-4，σ 仍 0.57，50 iter）**：

- iter0 KL 从 13.35 降到 **0.10**（lr 确实能修 KL）；
- 但确定性 eval 仍崩：st1 **0.0230**、plant 0.0191、term_lateral **0.78**；
- **同一权重的加噪 eval（M1，ACT_NOISE=1.0）**：st1 **0.1520**、s0ever 0.1994、
  plant 0.0676 —— 确定性/随机执行差 **×6.6**，与训练侧量级对上。
- ⇒ lr 不是主因（更新量只有 stage9 的 ~1/40，崩幅却相同），
  **方向才是**：策略被训成“均值顶墙、噪声抽奖”。

**t15（σ0.10/cap0.15 + lr3e-5 同时降，480 env / 200 iter，起点 stage3-v3）**：

| 口径 | 指标 | 值 | 对比 |
|---|---|---|---|
| 训练侧 iter199 | st1† / plant† / s0ever | **0.9708 / 0.8917 / 0.5729** | stage9 全程 0.098/0.023/0 |
| 确定性 eval（128 env，镜像） | st1 / plant / s0ever / stage_max | **0.6537 / 0.5933 / 0.7079 / 1.34** | 起点 0.6069/0.4876/0.7126/1.16 |
| 同上 | s0_press / term_lateral / term_roll | 0.2903 / 0.4725 / 0.1704 | 与起点持平 |

iter0 KL 7.01（σ 降导致 KL 放大 ~32×）→ iter1 1.41，adaptive lr 自己把它压住，
终态 σ 仅涨到 0.11（未触 cap）。

**机制结论**：高 σ 下 PPO 优化的是随机策略期望，均值可以漂到“顶墙等待”
而靠噪声抽 plant 奖；BC 起点没有这个问题是因为它的均值就是老师动作。
单降 σ（t13）会把 KL 爆炸放大、把更新毁掉；单降 lr（t14）不改变方向；
**只有 σ+lr 同降**，训练目标才和部署行为对齐。

**未解决**：t15 的 terrain_level 只到 0.23（lvl0、无栏），跨栏与课程提升未验证；
term_lateral 0.47 仍是最大死因。

---

## 7. 结论

**可以确认的**

1. stage7 配置下的起点策略本身就只有 `s0ever_near` 0.0193、
   `stage_1_unlocked` 0.0055（A 组）。这不是“训练学坏了”，
   而是**起点就已经废了**。stage7 训练 2000 iter 全程 `stage_max = 0`
   也是同一个事实的另一个面。
2. 废掉的原因是 `self_geometry=False` + `min_terrain_level=4` 两个因子叠加，
   各占约一半，几乎完全可加（`s0ever_near` 预测 0.625 vs 实测 0.655；
   `stage_1_unlocked` 预测 0.578 vs 实测 0.5781）。
3. 在 16 项深度退化里，`self_geometry` 是唯一有实质贡献的一项
   （C 组只开这一项就到 0.3377 / 0.3937，B 组 16 项全开到 0.3291 / 0.3365）。
4. 训练同款动作噪声（σ≈0.58）会把 D 组起点从 0.655 压到 0.397，
   但它压不到 0；**噪声不是 stage7 锁死的根因**。
5. 仅调整这两个开关**不足以**让训练学会 plant：stage9 从 0.655 起跑，
   2000 iter 后确定性 eval 跌到 0.044、`plant_complete` 归零。
   **训练目标/信号仍有问题，且不在这两个开关上。**
6. **这个退化不是评测口径假摔（本轮结论，L 组定标）**：把 eval 换成训练环境的
   精确镜像（10 Hz tick + action_delay + DR prearm），起点权重在 128 env 是
   0.6069、在 480 env 是 0.5021；四种口径的下界（0.3328）仍然是训练 100 iter 后
   （0.0938）的 3.5 倍。**训练确实在破坏一个本来能用的策略。**
7. **四个候选机制先被自己的数据推翻**：冻结 depth encoder（t11）→ 确定性
   0.0000；`latent.detach()`（t12）→ 未跑完，无结论；降探索噪声 σ=0.15/0.2
   但 lr 不动（t13）→ `s0_press` 反而 0.7671；换 scan teacher 通道（t10）→ 训练侧
   0.9208/0.7479 但确定性 0.0669/0.0098，比它自己的起点还低。
   ⇒ 当时结论“不是相机、不是 encoder、不是噪声”中，**噪声半错**：
   σ 确实是根因之一，但必须与 lr 同降才有效（§6.5 t15）。
8. **teacher 通道给出了最后的指向**：t10 训 500 iter 后 `frac past g0` 从 0.3910
   升到 0.5465、`s0ever_near` 从 0.2022 升到 0.4931，但 `plant_complete` 从
   0.1247 降到 0.0098。策略**学会走得更远，同时学会不 plant**。
   这与 stage9 的 `s0_press` 0.46–0.82 + `term_timeout` 高完全一致：
   **奖励地形把“贴近墙/顶墙/驻留”变成了比 plant 更划算的解，PPO 忠实优化了它。**

**因此**

`self_geometry=True` 与 `min_terrain_level=0` 应作为后续所有 run 的
**默认基线**（它们把起点从 0.02 拉到 0.65，而代价是一次地形重建），
但**不要再指望只改 cfg 就能训出 plant**。

根因已收敛到**奖励/任务信号**：环境在鼓励“不 plant 也能活得很好”，
而 plant 的门槛又卡（`base z>0.2` + 双前脚腾空 + `approach_vel` + `goal_idx`）。
PPO 不是学不会，是**正确地学会了错的东西**。

**下一步建议（按性价比排序，11-01 重写：§6.5 已破案，以下取代旧清单）**

1. **正式 run（t15 配方拉长）**：σ0.10/cap0.15 + lr3e-5（上限 1e-4），
   起点 stage3-v3，lvl0，1000–2000 iter（~45–90 min），每 200 iter 做一次
   确定性 eval 防回归。验收：st1 稳定 ≥0.6 且随 iter 不降。
2. **课程推平**：plant 稳后把 `min_terrain_level` 2→4（栏只存在于 ≥4，
   `flyball_single_hurdle_zero_below_level=4`），验证跨栏；注意换 lvl 会重建地形缓存。
3. **盯 term_lateral（0.47，最大死因）**：若正式 run 不改善，再查走廊侧向
   约束与 yaw 指令通道（`terminate_on_yaw_tracking=False` 下 lateral 仍 ON）。
4. 奖励层面的 `termination=-150` vs `timeout=0` 风险比问题降为次要：
   σ+lr 同降后策略已能主动尝试并成功（t15 plant 0.59）。

**不建议做的**：

- 继续冻结 encoder / detach latent / 调噪声——§6.3 已四路证伪，再扫是浪费 GPU。
- 再扫 `self_geometry` / `hole_fill` / `pw_noise` 或 env 数——§2、§3、§6.2 已吃干净。
- 再用训练日志的 `stage_*_unlocked` 当进度指标——它是粘性累计，只能证明
  “曾经有一个 env 成功过”，不能证明“现在能”。

---

## 8. 复现命令

```bash
cd /root/autodl-tmp/go2-flyball-parkour/legged_gym
export PATH=/root/miniconda3/envs/parkour/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/parkour/lib

# A: stage7 现状
EP_EVAL_MODE=depth EP_EVAL_CKPT=logs/extreme-parkour/stage3-v3/model_0.pt \
EP_EVAL_STEPS=3000 EP_EVAL_LABEL=A_clean_lvl4 \
python /tmp/eval_policy3.py --task go2_depth --use_camera --headless \
  --device cuda:0 --num_envs 128 --no_wandb

# C: 只开 self_geometry
EP_EVAL_CFG_OVERRIDE='{"depth.self_geometry": true}' ...

# B: 完整退化集
EP_EVAL_CFG_OVERRIDE="$(cat /tmp/cfg_b.json)" ...

# D: self_geometry + min_terrain_level=0（recommended baseline）
EP_EVAL_CFG_OVERRIDE='{"depth.self_geometry": true, "terrain.min_terrain_level": 0}' ...

# H1 / H2: 同权重加动作噪声
EP_EVAL_ACT_NOISE=1.0   # 或 0.3

# I: 逐帧 vs 10 Hz tick
EP_EVAL_DEPTH_TICK=1 ...

# L: 训练环境精确镜像（tick + delay + DR prearm）
OV='{"depth.self_geometry": true, "terrain.min_terrain_level": 0,
     "domain_rand.action_delay": true, "domain_rand.action_delay_start_latched": true,
     "domain_rand.dr_ramp_start_armed": true}'
EP_EVAL_CFG_OVERRIDE="$OV" EP_EVAL_DEPTH_TICK=1 EP_EVAL_MODE=depth \
EP_EVAL_CKPT=logs/extreme-parkour/stage3-v3/model_0.pt \
python /tmp/eval_policy3.py --task go2_depth --use_camera --headless \
  --device cuda:0 --num_envs 480 --no_wandb     # L2（128 即 L1）

# t10: scan teacher 对照（480 env / lvl0 / selfgeo）
EP_TRAIN_CFG_OVERRIDE='{"terrain.min_terrain_level": 0}' \
  python /tmp/train_stage9.py --task go2_teacher --resume --resumeid teacher \
  --checkpoint 3200 --exptid t10-scan-lvl0 --num_envs 480 --max_iterations 2000 --no_wandb

# t11: 冻结 depth encoder
EP_FREEZE_DEPTH_ENCODER=1 EP_TRAIN_CFG_OVERRIDE='{"terrain.min_terrain_level": 0}' \
  python /tmp/train_t11.py --task go2_depth --use_camera --resume --resumeid stage3-v3 \
  --checkpoint 0 --exptid t11-frozenenc-lvl0 --num_envs 480 --max_iterations 600 --no_wandb

# t13: 降探索噪声
EP_RESET_STD=0.15 EP_STD_CAP=0.2 python /tmp/train_t13.py ... --exptid t13-lowstd-lvl0
```

注意：`min_terrain_level` 参与 terrain cache key，首次会重建 ~1.3 GB /
约 15 min（c80 网格）；之后命中 `r10xc80_hs0.01_eeb68991a573.pkl`（0.7 s）。

---

## 9. 产物清单

| 文件 | 内容 |
|---|---|
| `/tmp/eval_policy3.py` | 评测脚本（cfg override + 动作噪声） |
| `/tmp/train_stage9.py` | 训练启动器（monkey-patch `get_cfgs`） |
| `/tmp/cfg_b.json` | B 组 16 项覆盖串 |
| `/tmp/eval_warmstart.log` | A 组 |
| `/tmp/eval_C_selfgeo.log` | C 组 |
| `/tmp/eval_B_fulldeg.log` | B 组 |
| `/tmp/eval_E_clean_lvl0.log` | E 组 |
| `/tmp/eval_D_selfgeo_lvl0.log` | D 组 |
| `/tmp/eval_stage9_1500.log` | stage9 iter1500 |
| `/tmp/eval_stage9_final.log` | stage9 final |
| `/tmp/eval_noise1.log` / `/tmp/eval_noise03.log` | 噪声判定 |
| `/tmp/eval_F1_D_delay.log` / `/tmp/eval_F2_S9_delay.log` | F 组：action_delay |
| `/tmp/eval_G_480.log` | G 组：480 env 原口径 |
| `/tmp/eval_I1_tick.log` / `/tmp/eval_I2_tick.log` | I 组：10 Hz tick |
| `/tmp/eval_J1.log` / `/tmp/eval_J2.log` | J 组：tick + 噪声 1.0 |
| `/tmp/eval_K_t11.log` | K 组：t11（冻结 encoder）终权重 |
| `/tmp/eval_T0.log` / `/tmp/eval_T1.log` | T 组：teacher 起点 / t10 model_500 |
| `/tmp/eval_L1.log` / `/tmp/eval_L2.log` | L 组：训练环境镜像（128 / 480 env） |
| `/tmp/eval_t14.log` / `/tmp/eval_t14_noise.log` | t14 确定性 / 加噪（M1） |
| `/tmp/eval_t15.log` | **t15 确定性 eval（首次超起点）** |
| `/tmp/train_t14.log` / `/tmp/train_t15.log` | t14/t15 训练日志 |
| `/tmp/train_t14.py` / `/tmp/train_t15.py` | 训练 wrapper（lr override / σ+lr 合并） |
| `logs/extreme-parkour/t14-lowlr-lvl0/` | t14（`model_0.pt`=iter49，名字误导）|
| `logs/extreme-parkour/t15-lowstd-lowlr/` | **t15（`model_0.pt`=iter199；model_50/100/150 为中间点）** |
| `/tmp/train_t10.log` | t10 scan teacher 训练日志 |
| `/tmp/train_t11.log` | t11 冻结 encoder 训练日志 |
| `/tmp/train_t12b.log` / `/tmp/train_t13.log` | t12 detach（kill）/ t13 低噪声 |
| `/tmp/train_t11.py` / `/tmp/train_t12.py` / `/tmp/train_t13.py` | 训练 wrapper（cfg override / 冻结 / detach / σ） |
| `/tmp/extract_s9.py` | 从训练日志提取 11 列曲线 |
| `logs/extreme-parkour/t10-scan-lvl0/` | teacher 对照（model_0/100/200/300/400/500）|
| `logs/extreme-parkour/t11-frozenenc-lvl0/` | 冻结 encoder（model_0/500）|
| `logs/extreme-parkour/t13-lowstd-lvl0/` | 低噪声（600 iter）|
| `/tmp/train_stage9.log` | stage9 训练日志 |
| `/tmp/train_stage7_lvl4.log` | stage7 训练日志（回归基线） |
| `logs/extreme-parkour/stage9-selfgeo-lvl0/` | stage9 产物（含 `env_cfg.pkl`） |
| `logs/terrain_cache/r10xc80_hs0.01_eeb68991a573.pkl` | lvl0 地形缓存 |
| `logs/terrain_cache/r10xc80_hs0.01_7c7648d26b43.pkl` | lvl4+ 地形缓存 |
