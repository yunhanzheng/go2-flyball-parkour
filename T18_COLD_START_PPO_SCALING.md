# t18 冷启动课程卡死诊断 & PPO 规模重配（2026-11-01）

承接 `T18_ONE_STAGE_DEPTH_BC.md`。本文记录 t18-run1/run1b 课程卡死 row 0 的完整诊断链、
从 wandb 二进制里挖出的 teacher 血脉真相、以及 run3→run4 的 PPO 参数规模重配决策。

## TL;DR

1. **课程卡死不是链路 bug**：地形映射、课程规则、resume 逻辑全部验证无误；
   eval 实测 model_7999 会跑会压墙（27% 完成 FRONT_PLANT），但 REAR_PLANT 保持
   成功率 0%，走不完 4 航点回路 → `max_goal_idx < 3` → 永不晋升。
2. **本仓库历史上没有任何成功 teacher 是从零训的**。wandb metadata 证实血脉是
   flat_baseline → new-run → new-run-v2 的接力 warm-start。t18 冷启动是在走一条
   从未有人走通的路，参照系（new-run-v2）的 lr 带是在大 batch + 热启动下测出来的。
3. **480 env 下 adaptive KL 控制器失灵**是冷启动失败的主因之一：minibatch KL 测量
   噪声大，run1b 被误刹车压到地板（lr 7.6e-5、est 0.02），run3 反向失守冲顶
   （iter 3 到 1e-2 天花板 → critic 打崩 value 0.05→1.54）。
4. run4 = 冷启动 + C/D/E 三项修复（steps 96、epochs 2、KL 检查改 per-iteration），
   开局 lr 爬升速率降为每 iter ×1.5（run3 的 1/5），critic 未爆。

## 1. 症状

t18-run1b（从 run1/model_7999 热重启）544 iter 内 480 env 全员 `LVL mean=0.00 max=0`，
零晋升；run1 原始 8000 iter 同样从未离开 row 0（est=0.02 为间接旁证）。
课程参数：promote = `max_goal_idx_ep ≥ int(0.375×8) = 3`，demote = `< 2`，
仅对 env_class ∈ {15,19} 生效（goal 规则）。

## 2. 根因排除过程（三个假设）

| 假设 | 内容 | 结论 |
|---|---|---|
| A | helpers.py 在 use_camera 下重写地形 mix/grid，训练地形错配 | **排除**。mix/grid 重写只在 `use_camera AND headless` 分支触发（helpers.py L157），t18 不传 `--headless`（日志 num_cols=80 为证）；非 headless 分支仅改 `terrain.y_range=[-0.1,0.1]`，`flyball_wall_terrain` 不接收该参数，无影响 |
| C | terrain.py 里 grep 不到 flyball_box，类型映射可能断 | **排除**。映射是纯位置式：`terrain_dict` 键序 → 累积 proportions 阈值 → terrain.py L876 `elif choice < proportions[18]: idx = 19` → `flyball_wall_terrain()`。100% 占比下所有 tile env_class=19，goal 规则确实生效 |
| B | policy 本身不够强 | **证实**（见下 eval） |

### eval 实测（model_7999，teacher 模式，96 env × 3000 步，row 0 无栏）

```
frac past g0 (goal_idx>=1) : 0.6489      mean base x: 9.95 m
frac wococo_armed          : 0.5181
s0_active / s0_press       : 1.00 / 0.47   stage_1_unlocked: 0.275
s1_hold_ok                 : 0.0000  ← 卡点：REAR_PLANT 保持一次都没成功
max goal_idx reached       : 1  (< 晋升阈值 3)
```

结论：助跑→压墙→前掌 plant 链路已会，后掌撑墙→回程从未完成。**链路通、能力不够**。

## 3. 与 new-run-v2 的对比（env_cfg.pkl + git 存档）

new-run-v2 生效配置（`logs/extreme-parkour/new-run-v2/env_cfg.pkl`）：
**num_envs=6144、use_camera=False（纯 scan PPO）**、10×80 课程、promote/demote 同现在。
它当时两处未提交的本地改动（`only_positive_rewards=False`、
`enable_wococo_armed_term_exempt=False`）现已入库（go2_teacher_config.py L280/L425），
**配置文本两边一致，差异全在执行规格**：

| | new-run-v2 | t18 |
|---|---|---|
| num_envs | 6144 | 480（相机+深度缓冲显存上限，实测 18.1/24.5GB） |
| 每 iter 样本 | 147k | 11.5k（旧）→ 23k（run3）→ 46k（run4） |
| minibatch | 36864 | 2880（旧）→ 23040（run3/4，minib=1） |
| use_camera | False | True |
| 入口 | train.py + logger | /tmp/train_t18.py 裸脚本 |

## 4. 重大发现：teacher 血脉全是 warm-start + 健康 lr 带（wandb 二进制解析）

`legged_gym/logs/wandb/` 下保留着历代 run 的 `.wandb` 文件，用
`wandb.sdk.internal.datastore` + `wandb_internal_pb2` 可离线解析
（history 记录字段：`h.key` 或 `'/'.join(h.nested_key)`，值在 `h.value_json`）。

**启动参数（wandb-metadata.json）**：
- `new-run`：`--resume --resumeid flat_baseline`（10000 iter）
- `new-run-v2`：`--resume --resumeid new-run --checkpoint 5000`（4000 iter，lvl 7）
- → **从零冷启动在本仓库没有成功先例**

**`Loss/learning_rate` 全历史统计**：

| run | lr 范围 | lr 均值 | 备注 |
|---|---|---|---|
| new-run | 1e-5 ~ 8.65e-4 | 1.7e-4 | 26% 时间触地板但快速反弹 |
| new-run-v2 | 1e-5 ~ 3.84e-4 | 1.4e-4 | bang-bang 正常带 |
| t18-run1b（480旧参数） | 7.6e-5 → 地板 | — | KL 噪声误刹车，瘫死 |
| t18-run3（480，minib=1+kl0.03） | **顶到 1e-2 天花板**（iter 3）→ 地板 | — | 油门空踩，critic 打崩 |

两个健康 run 的 lr 从未超过 9e-4 —— `max_learning_rate=1e-2` 在成功先例里
从未被触碰，小 fleet 下 KL 测量失真才会撞上去。

**learned std 轨迹**（ckpt 提取）：new-run-v2 从 0.567 → 0.647（iter500 峰）→ 0.566
（iter3800，学会收缩=收敛特征）；t18-run1 停在 0.67-0.69 平台（只探索不收敛）。
std 达峰后回落 = 健康冷启动的判据之一。

## 5. run3 事故复盘（为什么 minib=1 + desired_kl=0.03 还不够）

run3 开局：KL 早期读数小（随机策略动作变化在 KL 公式下不显著）→ adaptive 油门
**每 minibatch ×1.5、每 iter 检查 5 次** → iter 3 即触顶 1e-2 → value_loss
0.05→1.54、est 0.34→0.065 并趴住 → 之后刹车沉底振荡（iter 225 起 lr≈1e-5）。
教训：**控制器检查频率本身是小 batch 噪声的放大器**，光放宽阈值治不了。

## 6. run4 参数集（C+D+E，用户拍板；A 封顶 max_lr 暂未采纳）

| 项 | 值 | 目的 |
|---|---|---|
| C: `num_steps_per_env` | 24→48→**96** | 每 iter 46k 样本（6144 原设计的 1/3），KL 噪声 ∝1/N 下降，GAE 视界翻倍 |
| D: `num_learning_epochs` | 5→**2** | 小 batch 少嚼，防 iter 内累积 KL 过冲 |
| E: KL 控制器 | **per-minibatch → per-iteration 平均，每 iter 最多 ×1.5/÷1.5 一次** | 直接消灭噪声放大器；改动在 `rsl_rl/rsl_rl/algorithms/ppo.py update()`（editable 安装，源码即生效） |
| 保持 | adaptive、desired_kl=0.03、minib=1、std 1.0、entropy 0.01、clip 0.2 | — |
| 种子 | 默认 1（与历代相同，A/B 可比） | — |

启动：`EP_T18_MINIB=1 EP_T18_STEPS=96 EP_T18_EPOCHS=2 EP_T18_DESIRED_KL=0.03
EP_T18_SAVEINT=2000 python -u /tmp/train_t18.py --task go2_teacher --use_camera
--num_envs 480 --max_iterations 100000 --exptid t18-run4 --no_wandb`

**开局实测**（iter 0-28）：5.1s/iter、18.4GB、lr 以每 iter ×1.5 上爬，**iter 24 触顶
1e-2**（run3 是 iter 3 触顶，E 把失控时间拉长 8 倍但没改变方向——早期平均 KL 就是
低于 0.06/2，油门没有反作用力）。与 run3 的关键差别：**critic 目前没崩**
（value 稳在 0.14-0.18、est 0.265，run3 同期 value 已冲到 1.5+）；lat_err
0.08→0.39 是 actor 在顶格 lr 下移动过快、encoder 追不上，属 BC 侧正常滞后。
**决策点已到**：天花板 1e-2 对 46k batch 无刹车兜底（健康先例从未超过 8.65e-4），
若 est/value 开始恶化即采纳 A（`max_learning_rate=1e-3`）重启 run5；若 LVL 在顶格
lr 下反而开始爬坡，则维持现状观察。

## 7. 工具与命令备忘

```bash
# teacher 行为评估（必须先停训练，双 IsaacGym 并行 init 会 CUDA crash）
EP_EVAL_MODE=teacher EP_EVAL_CKPT=<ckpt> EP_EVAL_STEPS=3000 \
python -u /tmp/eval_policy3.py --task go2_teacher --num_envs 96

# ckpt learned std 提取
torch.load(ckpt, map_location='cpu')['model_state_dict']['std'].mean()

# TB（借 AutoDL 平台 6007，logdir=/root/tf-logs，已 symlink logs/extreme-parkour）
```

## 8. 决策记录

- 冷启动继续，不热启动（用户拍板；时间不是约束，验证链路可行性优先）
- adaptive schedule 保留；minib 4→1、steps 24→48、desired_kl 0.01→0.03（run3，部分生效）
- run3 观察后仍恶化 → C(96)+D(2)+E(per-iter KL) 三连（run4，当前）
- A（max_lr 封顶 1e-3）作为 run4 失败时的第一 fallback
- 所有 run 必须开 TB（memory 30bfa20b）；TB x 轴 = iter（resume 时从 ckpt 编号+1 起算，
  run4 起为全新 run，从 0 画）

## 9. 判据（下一步看什么）

1. **lr 是否受控**：触顶 1e-2 前平均 KL 是否先越过 0.06 触发自然回落；
2. **est 回到 0.3+ 且 value_loss < 1** 持续（critic 活着）；
3. **std 达峰后回落**（对照 new-run-v2 的 0.65→0.57 形态）；
4. **LVL mean 离开 0**：样本量等效 new-run-v2 的 iter~150 时（≈35M 步，
   即 run4 的 iter ~760）应看到第一批晋升；iter 2000（≈92M 步）仍全员 row 0
   则冷启动可行性存疑，回头重议热启动或 flat_baseline 血脉的价值。

## 10. 后续（2026-11-01 晚）：run5/run6 全部回退，改走路 B

run4 触顶后先试 A（`EP_T18_MAXLR=1e-3`，run5），随后按用户指令**取消全部 PPO 改动**
（ppo.py 的 E 改动逐字还原、覆盖参数一律不设），run6 起即在初始化阶段被叫停。
结论是病根在规模而非超参：480 env 是"大 PPO + 相机"被绑成一个 run 的妥协产物。
最终方案与实测见 `T18_PATH_B_PARTIAL_FLEET.md`（部分机队深度采集，6144 env + 480 子集）。
