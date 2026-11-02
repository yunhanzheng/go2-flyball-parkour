# new-run-v3 训练方案

> 唯一改动：`dr_ramp_full_at_mean_level` 从 **6.0** 改为 **9.0**。
> 目标：让"地形倾角"和"域随机化强度"两条难度轴同步爬升，把课程从 6.9 级的僵持中解放出来。

---

## 0. TL;DR

| 结论 | 依据 |
|---|---|
| 6 级不再是天花板 | new-run-v2 把 `terrain_level` 推到 **6.99**，且全程无崩盘 |
| 能力已经到 8 级 | 同一个 ckpt，干净动力学下 8 级（69°）成功率 **0.909~0.938** |
| 卡住的是 DR，不是技巧 | 满额 DR 下 8 级只剩 **0.238**，7 级 **0.450** |
| DR 满额太早 | `dr_ramp_full_at_mean_level = 6.0` → 60° 墙时就按最难档打 |
| 修法 | 把它推到 9.0：ceiling 变成 0.67 / 0.78 / 0.89 / 1.00 @ 6/7/8/9 级 |

---

## 1. 实验线

| run | 关键配置 | 结果 | 诊断 |
|---|---|---|---|
| **new-run** | 延迟 latch@6 + 姿态豁免@6 + DR 满额@6 | 5124 崩盘，10000 轮后 `timeout 0.94` / `reward −10` | 三重难度在 6 级叠加，value 崩塌 |
| **new-run-v2** | 豁免关、`--nodelay`、DR 仍满额@6 | **4000 轮，level 6.99，reward +5.3，vfl 0.06** | 前两把刀确认是主因；剩下 DR |
| **new-run-v3**（本方案） | 上述 + DR 满额@9 | 待跑 | — |

v2 从 `new-run/model_5000` warm start，`--max_iterations 4000`，实测 3.9 s/轮，总 14066 s。

---

## 2. new-run-v2 训练曲线

| iter | level | reach | base_contact | stage1_ground | s1_hold | vfl | reward |
|---|---|---|---|---|---|---|---|
| 0 | 1.66 | 0.000 | 0.000 | 0.000 | 0.000 | 0.23 | −2.43 |
| 200 | 5.09 | 0.530 | 0.207 | 0.142 | 0.309 | 0.14 | +2.43 |
| 400 | 6.27 | 0.393 | 0.200 | 0.144 | 0.244 | 0.08 | −0.80 |
| 1000 | 6.15 | 0.367 | 0.143 | 0.104 | 0.294 | 0.08 | +1.55 |
| 1800 | 6.43 | 0.582 | 0.138 | 0.053 | 0.393 | 0.07 | +1.36 |
| 2400 | 6.83 | 0.530 | 0.086 | 0.101 | 0.369 | 0.06 | +3.75 |
| 3000 | 6.82 | 0.542 | 0.163 | 0.066 | 0.366 | 0.06 | +3.58 |
| 3400 | 6.84 | 0.546 | 0.064 | 0.114 | 0.355 | 0.05 | +5.67 |
| 3800 | 6.92 | **0.595** | 0.057 | 0.153 | 0.383 | 0.07 | **+5.32** |

要点：

- **无崩盘**：`vfl` 全程 0.05~0.09（new-run 崩到 1.85）
- **reward 首次转正并稳定在 +4~5.7**（new-run 峰值 +0.03）
- 两把刀确认未开：`action_delay/latched = 0`、`armed_term_exempt_armed = 0`
- `terrain_level` 峰值 6.99 @ iter 3871，**最后 1600 轮在 6.8~6.99 之间震荡**
- `s0_roll_ok` 仍只有 0.08（不侧倾），但不影响 6~8 级通过 → 这不是当前瓶颈

---

## 3. 根因：DR 满额早于地形难度

`model_3800` 分级评测，同一权重、两种动力学，64 envs，每级 ~35~40 回合：

| level | 墙 | reach（clean） | reach（full DR） | 差值 |
|---|---|---|---|---|
| 5 | 54.6° | 0.683 | 0.725 | +0.04（噪声内） |
| 6 | 58.3° | 0.882 | **0.767** | −0.12 |
| 7 | 66.3° | 0.757 | **0.450** | −0.31 |
| 8 | 69.0° | **0.909** | **0.238** | **−0.67** |

*（clean 145 ep / full-DR 152 ep；L5 的反向差值是 ±0.05 量级的抽样噪声）*

满额 DR 下的死因构成：

| level | roll | base_contact | stage1_ground | 主要问题 |
|---|---|---|---|---|
| 7 | 0.075 | 0.075 | **0.350** | 后脚 plant 时踩回地面 |
| 8 | **0.238** | **0.333** | 0.167 | 打滑侧倾 / 躯干压墙 |

三点排除：

- ❌ 不是 plant 能力（clean 69° 有 91~94%）
- ❌ 不是 roll 对齐不足（clean 下 `s0_roll_ok = 0.08` 也能 94%）
- ❌ 不是 Head 撞墙（clean 下 `base_contact = 0.000`；满 DR 才 0.333，属于"被 DR 打乱后才撞"）

### 为什么表现为"震荡在 6.9"

晋级门槛 `promote_goal_frac = 0.375`，而降级 `demote_goal_frac = 0.25`。满额 DR 下 7 级 reach 只有 **0.450**，勉强过线：

```
6 级 0.767 → 升 7 级 → 0.450 压线 → 波动即降回 6 → 再升 → …
```

而 8 级 0.238 远低于门槛，根本进不去。

### DR 强度公式

[legged_robot.py:1227-1235](file:///root/autodl-tmp/go2-flyball-parkour/legged_gym/legged_gym/envs/base/legged_robot.py#L1227-L1235)：

```python
ceiling  = clamp(mean_terrain_level / dr_ramp_full_at_mean_level, 0, 1)
dr_scale = uniform(0, ceiling)          # dr_scale_sample_per_env = True
```

`dr_scale = 0` 是标称物理，`dr_scale = 1` 是配置范围全宽（摩擦 0.6~2.0、附加质量 0~3 kg、质心 ±0.2 m、电机力矩 ±20% …）。

---

## 4. v3 改动

`legged_gym/legged_gym/envs/go2/go2_teacher_config.py`：

```diff
-        dr_ramp_full_at_mean_level = 6.0
+        dr_ramp_full_at_mean_level = 9.0
```

| level | 6.0（旧） | 9.0（新） |
|---|---|---|
| 6 | 1.00 | **0.67** |
| 7 | 1.00 | **0.78** |
| 8 | 1.00 | **0.89** |
| 9 | 1.00 | **1.00** |

9 级（75°，最陡）仍然会拿到满额 DR，sim2real 覆盖不减；只是不再让"最难物理"跑在"中等地形"前面。

保持不动：豁免关闭、`--nodelay`、`roll_term_deg 95`、`terminate_after_contacts_on = ["base","Head"]`、奖励权重。

---

## 5. 起跑

```bash
cd /root/autodl-tmp/go2-flyball-parkour/legged_gym/legged_gym/scripts
conda activate parkour
python train.py --task go2_teacher \
  --exptid new-run-v3 --proj_name extreme-parkour \
  --device cuda:0 --headless \
  --resume --resumeid new-run-v2 --checkpoint 3800 \
  --nodelay --max_iterations 4000
```

注意：

- `--max_iterations` 是**增量**（`current_learning_iteration` 的恢复被注释掉了），4000 = 再训 4000 轮
- `--exptid` 必须换名，否则会覆盖 `new-run-v2` 的文件
- 地形课程不继承，从 `max_init_terrain_level = 5` 内随机起步，约 200~400 轮回到 6 级
- warm start 会自动 pre-arm DR，第 0 轮起就按新 ceiling 计算

---

## 6. 监控 / 验收 / 止损

| 检查点 | 期望 | 不达标时 |
|---|---|---|
| 本炉 500 轮 | `terrain_level` ≥ 7.0 且不再回落 | 查 `dr_ramp_armed` 是否生效 |
| 本炉 1500 轮 | ≥ 8.0，`reach_goal` ≥ 0.55 | 见下方"Plan B" |
| 本炉 3500 轮 | ≥ 8.5，或 `reach_goal ≥ 0.65` | 见下方"Plan B" |
| 全程 | `vfl < 0.3`、`lr > 3e-5` | 崩盘信号，立即停 |

**Plan B（若 1000 轮后仍在 6.8~7.0 震荡）**，按顺序单变量试：

1. `wococo_stage1_ground_max_z: 0.05 → 0.02`（满 DR 下 7 级第一大死因 0.350）
2. `dr_scale_range: [0.0, 1.0] → [0.0, 0.8]`（连最坏档也一起收敛）
3. `wococo_base_orient` 权重 0.25 → 0.35（治 8 级的 roll 0.238）

---

## 7. 遗留问题（本方案不处理）

| 问题 | 现状 | 计划 |
|---|---|---|
| 20 ms 动作延迟 | 训练关闭（`--nodelay`） | 等 9 级稳定后开 stage2：`--delay` + 从 4 级重爬，或改 `action_delay_latch_mean_level` 为 8.0 |
| 8~9 级 roll 窗口 | `roll_target −75°` vs `roll_term 95°`，只剩 20° 余量 | 若 9 级批量 `term_roll`，把 `roll_term_deg` 提到 110 |
| `s0_roll_ok` 只有 0.08 | 一直不侧倾 | 若 8 级出现批量 `base_contact`，再考虑加强 `wococo_base_orient` |
| 5 级 `stage1_ground = 0.25`（满 DR） | 低等级反而最严重 | 与 Plan B 第 1 条一起改 |

---

## 8. 复现评测

```bash
# 干净动力学（s = 0）
python play.py --exptid new-run-v2 --resume --resumeid new-run-v2 --checkpoint 3800 \
  --device cuda:0 --headless --no_video --num_envs 64 \
  --eval_min_level 5 --eval_max_level 8 --max_steps 2500

# 训练时的 DR（s ~ U(0, ceiling)）
... 同上，追加 --full-dr
```

输出中 `[eval] TERMINATION MODE x LEVEL` 就是要看的表。

指标口径：

- `reach_goal`：回合以"完成整个赛程"结束的比例，等价成功率
- `term_base_contact`：躯干或 Head 触地/触墙导致的终止占比
- `term_stage1_ground`：后脚 plant 阶段脚踩回地面导致的终止占比
- `s0_roll_ok`：APPROACH 阶段 |roll − roll_target| ≤ 45° 的时间占比

数据位置：

- 训练日志：`legged_gym/logs/wandb/<run-id>/files/output.log`
- 权重：`legged_gym/logs/extreme-parkour/<exptid>/model_*.pt`
- 评测输出：本文件第 8 节命令的 stdout
