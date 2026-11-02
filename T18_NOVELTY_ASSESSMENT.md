# 新颖性判定 & "cos→swap 损耗"判据的实验设计（2026-11-01）

> **结论先行：方法层不是首创。** 路 B 的三个组成部分（大机队+少量相机渲染、师生同一炉、
> latent 对齐蒸馏）在公开文献/官方框架里都有明确先例。可主张的只有三样，分两档强弱：
> **实现层（属于我，但不是方法学 novelty，只在"本仓库首创"这个口径上成立）** 和
> **诊断量（cos→swap 损耗的定量对应关系，最像贡献，但目前只有 n=2，不足以支撑主张）**。
> 本文同时给出把第二样做成可信贡献的实验设计。

---

## 1. 先例清单（为什么不算首创）

| 我的做法 | 已有先例 | 出处 |
|---|---|---|
| 6144 env 物理 + 只有 480 env 渲染深度 | Isaac Lab **TiledCamera**：把所有并行 env 的相机视口拼成一张大纹理、一次 render 出全部图像，专为"感知 RL 的渲染开销"设计 | [arXiv 2511.04831](https://arxiv.org/html/2511.04831v1) |
| 带相机的 env 数远小于物理 env 数是常态 | Aerial Gym 报"210 个机器人 + 270×480 深度相机 = 1800 fps"；DiffAero 的深度渲染 FPS 表同样是按并行 env 数单独列的 | Aerial Gym (2023)、[arXiv 2509.10247](https://arxiv.org/html/2509.10247v1) |
| teacher 边变强、encoder 每步拿**当前** teacher 标签做 BC | EgoPush 的学生就是"在线 DAgger 式：每次迭代在线查询 teacher 生成动作标签 + 监督更新" | [arXiv 2602.18071](https://hyper.ai/cn/papers/2602.18071) |
| student latent 对齐 teacher 的高程图/MLP encoder 输出，部署时直接换感知头 | "Learning End-to-End Humanoid Locomotion from Raw Pixels"：蒸馏时让 student CNN 的 latent 匹配 teacher MLP encoder 输出 | [arXiv 2602.06382](https://arxiv.org/html/2602.06382v2) |
| "单阶段优于两阶段"的动机 | 社区已把理由写成结论：两阶段的痛点是"上限被 teacher 卡死 + 模仿误差累积"，PIE 明确主张 one-stage end-to-end；RMA/UPENG/PAIR/AMSI 是这条谱系上的既有工作（**这一行我只看了摘要，若要写进 related work 必须逐篇核对**） | eefocus PIE/WMP 综述、zhihu 运动控制 Know-How |

同规模无相机 baseline 的实测也在提示"这不是魔法"：路 B 与 new-run-v2 的 iter 时间只差 8%，
说明"大机队 + 小渲染子集"本来就是可扩展的工程做法，不是新范式。

---

## 2. 三档归属（诚实版）

### 2.1 实现层 —— 属于我，但不构成方法学 novelty；"本仓库首创"口径成立

`cfg.depth.num_depth_envs` 的**前缀切片**实现（详见 `T18_PATH_B_PARTIAL_FLEET.md` §1）。
值得记一笔的不是"只渲染一部分 env"，而是**为什么只能切前缀、不能任意散布**：

- `process_depth_batch` 的每 env 时序退化状态全部按**行号**索引：
  `self.pw_bs_k[:N]`（盲区宽度随机游走）、`self._pw_low`（孔洞漂移场，shape 含 N）、
  `self.depth_noise_scale[:N]`（噪声 scale）；
- 换成任意行号索引，这些状态会被写到**错误的 env** 上（写回也是按位置切片的），
  退化时序就错乱了 —— 要支持跨列散布，得给 `process_depth_batch` 加行索引参数并让
  状态读写都做 gather/scatter，那就重新引入 per-step 的索引开销；
- 前缀还顺带满足了 on-demand 采集的相位语义：`active = (due | fresh) & (row < nd)`，
  `due` 本身是按 per-env `depth_latency` 算的，子集只是把掩码按行截断，不改 lag 分布。

代价（已知妥协，别掩盖）：`terrain_types = arange(N) // (N/num_cols)`，6144/80 = 76.8，
所以 env 0..479 只落在 **7 个地形列**上，BC 每炉最多见到约 70 个 tile 变体。

**仓库内首创的证据**：原链路只有 `depth.camera_num_envs = 280` 这种**整机队切换**
（helpers 在 `--use_camera --headless` 时把 num_envs 从 6144 改成 280），
两种规格从未在同一个 run 里共存。

### 2.2 可辩护的卖点 —— cos（latent 对齐度）预测 swap 损耗

现有全部证据（数字取自 `T18_ONE_STAGE_DEPTH_BC.md` §1.1/§1.2，脚本
`/tmp/enc_loss_measure.py`、`/tmp/swap_enc_eval.py`）：

| encoder | cos(深度 latent, teacher scan latent) | swap 后 st1 | 相对 ctrl(0.6423) 的损耗 |
|---|---|---|---|
| t16/model_200 | **0.779** | **0.6419** | ≈ 0（零损耗） |
| v3-student/model_5000 | 0.459 | 0.5540 | −14%（plant 0.642→0.431，−33%） |
| stage3-v3/model_0 | 0.338 | 未测 | — |
| REF（teacher 自轨 + v3 enc） | 0.457 | — | — |

**为什么还不能当贡献卖（三个硬伤）**

1. **n=2**。两组数据点不足以支撑"定量对应关系"，只能叫经验观察。
2. **cos 的口径不统一**：REF 那行是在 **teacher 自己滚的轨迹**上测的，学生行是在
   **学生自己滚的轨迹**上测的（见 `enc_loss_measure.py` 的两个 measure 分支）。
   latent 目标本身依赖状态分布，所以不同 rollout 策略下的 cos **不可直接横比**——
   而 swap 损耗恰恰是在特定状态分布上发生的。这是最需要先修掉的方法学漏洞。
3. **只看 32 维 tanh latent 的整体均值 cos**：没有逐维分解（可能只有一两维对齐），
   没有 p10/p90（尾部不对齐可能才是致命的），也没有 swap eval 的方差（env 数/seed 数未知）。

### 2.3 任务层 —— 环境资产，与路 B 无关

飞球墙 30–75° 随机倾角 + P4L 的 FRONT/REAR/LANDING FSM + goal-based 课程（promote≥3/8、
demote<2/8）。这部分才是这个仓库真正的差异化内容，但它是**任务/环境**贡献，不是训练方法贡献，
别和方法 novelty 混着写。

---

## 3. 要卖就卖这个：把 cos→swap 判据做成可信贡献

### 3.1 先统一测量协议（修掉硬伤 2/3）

- **固定状态分布**：所有 cos 一律在**同一批 teacher 自滚轨迹**上测（`obs` 由冻结 teacher 驱动），
  encoder 只做前向；这样 cos 是"encoder 在 teacher 会遇到的状态上对齐得多好"的纯净度量。
- **多口径同报**：整体均值 cos + 逐维 cos（32 个）+ p10/p50/p90 + `rel = MSE/E|s|²`
  + `E|z|²/E|s|²`（尺度失配会伪装成方向失配）。
- **swap eval 给方差**：96 env × 3 seed，报 st1/plant 的均值±极差，损耗用相对值。

### 3.2 数据网格（目标 ≥12 个 (cos, Δ) 样本点）

| 轴 | 取值 | 来源 |
|---|---|---|
| 训练阶段 | `t18-pathb-6144` 的 `model_1999/3999/5999/7999/...` | 每存一档顺手测一次（~5–8 min，**必须停训练**，双 IsaacGym 会 CUDA crash） |
| 子集大小 nd | 480 vs 960 | 覆盖"BC 多样性/样本量"这一轴 |
| 深度退化强度 | `depth_noise_scale` 钉 0 / 0.5 / 1.0 | `EP_EVAL_CFG_OVERRIDE` / pinned scale |
| 深度分辨率 | 106×60（本 run 原生）vs 480×270（部署镜像） | 这就是 §5.5 sim2real 警告的量化版 |

分析：散点 + 单调拟合，报 **Spearman ρ**；判据成立的最低门槛 = ρ 显著且 0.6–0.8 区间有可辨拐点。
达不到就把措辞降回"经验观察"。

### 3.3 三方对比：单炉 vs 两阶段

同一个 teacher actor ckpt，两条路线走到同等通过率，比三件事：

1. **总样本量**（env-steps）；
2. **墙钟时间**（GPU-秒，路 B 之后两阶段也能用 6144，所以**不能只按 iter 数比**）；
3. **通过率**（st1 / plant / timeout，lvl4 与满 DR 各一套）。

预期结论形态：单炉省掉"冻结 teacher + 二次采集"的整段成本，但 BC 在追动靶
（`act_err` 随 actor 变强而抬升），所以要同时报"encoder 收敛滞后多少个 iter"。

---

## 4. 表述红线

**不能写**：首创子集渲染 / 首次一阶段视觉跑酷 / 提出 latent 蒸馏 / 优于两阶段（未经 3.3 实测）。

**可以写**：
- "在单张 24GB GPU 上把 6144-env PPO 与 480-env 深度 BC 放进同一个 run：前缀渲染子集 +
  on-demand per-env latency，默认关闭时与原链路逐字一致"；
- "给出 latent 对齐度与感知替换损耗之间的定量对应关系（N= 组，ρ= ）"；
- "任务侧：飞球墙随机倾角 + P4L plant FSM + goal 课程下的 teacher-student 单一 ckpt 部署链路"。

**必须挂的前提**：① 本 run 深度区间是 teacher 原生 106×60，**不等于**真机 D435 的
480×270 部署镜像（`T18_ONE_STAGE_DEPTH_BC.md` §5.5）；② 6144 冷启动可行性尚未验证
（硬判据点 iter 1000，见 `T18_PATH_B_PARTIAL_FLEET.md` §6）。

## 5. 待办

- [ ] `enc_loss_measure.py` 加"统一 teacher 自轨 + 逐维 cos + p10/p90"的测量分支；
- [ ] 写 `cos_vs_swap.py`：输入 ckpt 列表，输出 (cos, Δst1, Δplant) 表；
- [ ] 每 2000 iter 收一个样本点（与 `EP_T18_SAVEINT=2000` 对齐）；
- [ ] related work 逐篇核对 RMA / UPENG / PAIR / AMSI / DVL / PIE（本文 §1 只到摘要级）。
