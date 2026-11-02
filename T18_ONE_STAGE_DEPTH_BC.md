# T18：一阶段 "Teacher PPO + 深度 Encoder 在线 BC"（拔高程图、插相机部署）

> 结论先行：**swap 实验已证明"固定 teacher actor、只把感知输入从高程图 scan 换成深度 encoder latent"在
> latent 对齐度足够高时几乎零损耗**（t16-enc cos=0.78 → st1 0.6419 ≈ 原生 ctrl 0.6423）。
> t18 就是把这件事做成一阶段流程：**teacher 从 0 开始 PPO，同一个 run 里外挂深度 encoder 全程 BC 对齐，
> 部署时直接换感知，不需要学生、不需要第二阶段。**
> 脚本：`/tmp/train_t18.py`（冒烟通过：3 iter 训练+保存+ckpt 结构完整）。

---

## 1. 为什么可行 —— 三条证据链

### 1.1 latent 对齐度测量（`/tmp/enc_loss_measure.py`）
目标 = teacher actor 的 scan_encoder 输出（stop-grad）；预测 = 各 checkpoint 的深度 encoder latent。
rel = MSE / E|s|²（rel≈1 相当于"输出全零一样远"）；cos 是方向对齐度。

| checkpoint | MSE | rel | **cos** |
|---|---|---|---|
| REF：teacher 自轨（天花板参照线） | 0.1394 | 1.097 | 0.457 |
| v3-student/model_5000 | 0.1425 | 1.079 | 0.459 |
| stage3-v3/model_0 | 0.1542 | 1.456 | 0.338 |
| **t16/model_200** | 0.0509 | 0.501 | **0.779** |

### 1.2 swap 部署实验（`/tmp/swap_enc_eval.py`）—— 核心证据
teacher actor（new-run-v2/model_3800）完全冻结，只换感知输入：

| 配置 | st1 | plant | s0ever | s0_press | term_lateral | timeout |
|---|---|---|---|---|---|---|
| ctrl：自带 scan（高程图） | 0.6423 | 0.642 | 0.816 | 0.177 | 0.121 | 0.251 |
| swap-A：+ v3-student enc | 0.5540 | 0.431 | 0.850 | 0.248 | 0.005 | 0.135 |
| **swap-B：+ t16 model_200 enc** | **0.6419** | 0.582 | 0.830 | 0.230 | 0.017 | 0.089 |

读数：
- **latent 对齐度直接预测换 encoder 的损耗**：cos 0.78 → 零损耗；cos 0.46 → 掉 14% st1 / 33% plant。
- plant 瓶颈在 teacher actor 本身（ctrl 也只有 0.642），不在 encoder。
- 意外收获：换相机通道后 term_lateral 从 0.121 暴跌到 0.005–0.017 —— 深度输入治好了 teacher 的侧向出界病。

### 1.3 t16 的教训（为什么 BC 要"全程在线"）
t16（1000 iter 蒸馏 run）呈倒 U 形：iter200 全场最佳（st1 0.6775 / plant 0.6180），
iter400 起单调崩到 iter999 的 0.0456；崩溃签名 = 原地站桩等 timeout（s0_press 0.29→0.45、
term_lateral 0.49→0.001）。说明**离线/滞后对齐会过拟合晚期的坏 teacher**；
t18 每个 rollout 用**当前** actor 现算 a_t / scan_lat 作目标，teacher 变好目标跟着变好。

### 1.4 谱系勘误
v3-student 的真实 teacher 是 **new-run-v2/model_3800**（不是 extreme-parkour/teacher 的 model_3200，
两者 35/35 张量全不同，是两次独立训练）。此前 §6.3 "stage3-v3 ≈ teacher-3200" 的比较不成立。

---

## 2. t18 训练流程设计

一个进程、一阶段，两件事同时发生：

```
每个 iteration（24 个 50 Hz env step）:
  ├─ PPO 主循环 = learn_RL 原样复刻（go2_teacher，scan 高程图通道，privileged critic）
  │    每 IV=5 步（部署的 10 Hz encoder 节奏）快照 (obs, depth frame)
  └─ rollout 结束后，对每个快照做 BC tick：
       scan_lat = actor.scan_encoder(obs_scan)          # stop-grad 目标
       a_t      = actor(obs, scandots_latent=None)      # teacher 动作
       z        = depth_encoder(frame, obs[:53])        # 深度分支（heading 槽 6:8 清零）
       a_s      = actor(obs, scandots_latent=z)         # 同一个 actor，换感知
       L        = ACT_W·‖a_s−a_t‖₂ + LAT_W·‖z−scan_lat‖₂
       Adam(encoder, L)，grad clip 1.0；actor 参数冻结仅 backward 期间
```

要点：
- **只有 encoder 收 BC 梯度**；PPO 对 encoder 无梯度（teacher 配置 train_with_ppo=False）。
- 目标是"当前 actor 自己"，actor 进步 → 目标同步进步（课程自举，不需要外部 teacher）。
- 两个损失项都保留：动作 BC 保证"换了输入动作不变"，latent MSE 保证表征本身对齐（swap 实验证明对齐度决定损耗）。
- hidden state 按 IV 节奏跨 tick 延续，episode done 清零（与部署 10 Hz 递归一致）。
- 训练时 actor 看高程图（特权信息，学得快），部署时拔高程图插相机 —— 与蒸馏同构，但全程一个 run、零第二阶段。

## 3. 部署路径

ckpt 同时含 `model_state_dict`（actor/critic，scan encoder 仍在里面但不再使用）
和 `depth_encoder_state_dict`，直接走已验证的 swap 评估路径（`/tmp/swap_enc_eval.py`）：
`actor(obs, scandots_latent = depth_encoder(frame, proprio))`。

冒烟验证（t18-smoke8b，3 iter × 128 env）：训练循环 + `model_{it}.pt` 保存 +
key 结构（model 35 张量 / encoder 18 张量）全部通过。

## 4. 启动方法

```bash
export PATH=/root/miniconda3/envs/parkour/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/parkour/lib
cd /root/autodl-tmp/go2-flyball-parkour/legged_gym

python -u /tmp/train_t18.py --task go2_teacher --use_camera \
    --num_envs 480 --max_iterations 100000 --exptid t18-run1 --no_wandb
# EP_T18_SAVEINT=2000 控制存盘频率（默认 400；100000 iter 下建议 ≥2000 以免写爆磁盘）
```

可调环境变量：

| 变量 | 默认 | 含义 |
|---|---|---|
| `EP_T18_ACTW` | 1.0 | 动作 BC 损失权重 |
| `EP_T18_LATW` | 1.0 | latent MSE 损失权重 |
| `EP_T18_ENCLR` | 1e-3 | encoder Adam 学习率 |
| `EP_T18_SAVEINT` | 400 | 每多少 iter 存一次 ckpt |
| `EP_T18_OD` | 0 | 1=深度渲染改 on-demand（实测收益 <4%，见 §5） |
| `EP_T18_TF32` | 1 | matmul TF32 开关（torch 2.0 默认关，脚本已默认开） |

输出目录：`legged_gym/logs/extreme-parkour/<exptid>/model_{it}.pt`。
建议 **每 200 iter 用 swap 评估挑点**（t16 教训：最佳点未必是最后一个）。

## 5. 训练提速（2026-11-01 实测，脚本已内置分段计时 roll/bc/upd）

### 实测分解

| 配置 | t/iter | roll | bc | upd | 吞吐 (env steps/s) |
|---|---|---|---|---|---|
| 128 env | 1.15s | 0.98 (85%) | 0.03 (3%) | 0.15 (13%) | ~2.7k |
| 128 env + on-demand 渲染 | 1.10s | 0.95 | 0.03 | 0.16 | ~2.8k |
| **480 env** | **1.45s** | 1.25 | 0.06 | 0.16 | **~7.9k** |

### 结论与建议（按收益排序）

1. **直接把 env 数拉满到 480（已实测）**：env 数 ×3.75，iter 时间只 ×1.26 —— GPU PhysX 主要吃
   launch 开销，批量扩展性极好，吞吐近 3 倍提升（~8.3k steps/s）。t18-run1 实测 480 env @1.4s/iter、
   显存 **18.1/24.5 GB（余 6.3 GB）**。正式 run 3200 iter ≈ **75 分钟**。
   **num_envs 建以 480 为最佳点，最多 560（~20 GB）；不建议 ≥640**——t18 额外扛 5 个 BC 快照缓冲
   （obs+深度帧，随 env 线性增长）+ 100000 iter 的 Adam state，容易 OOM ；且 480 后吞吐增益已趋平，
   硬顶高会把后续单卡并行 eval 的余量挤没。
2. **BC 外挂本身几乎免费**（bc 0.06s = iter 的 4%）——"提高训练速度"不需要砍 BC tick 频率，
   保持 IV=5 全量快照，别为省时间降低监督密度。
3. **TF32 已默认打开**（`torch.backends.cuda.matmul.allow_tf32`，torch 2.0 默认是关的）：
   对 encoder 首层 Linear(62400→128) 和 PPO 更新有加速；数值差异在训练可接受范围，出问题用
   `EP_T18_TF32=0` 回退。
4. **on-demand 渲染在 106×60 下收益很小**（~0.05s/iter，<4%）：teacher cast 分辨率低，渲染不是瓶颈，
   物理才是 → `render_on_demand` 保持关。注意：若将来把深度链路切到 student 480×270（§5.5），
   渲染会变成大头，那时 `render_on_demand=True` 才是正解（student 注释：帧/延迟相同、光线 ~2.5×↓）。
5. **省掉重复启动开销**：每次进程启动 ~3.5-4 分钟（sim 创建 + warp BVH）；地形已命中缓存(0.6s)。
   挑点评估攒着批量跑，别每 100 iter 手动起一次 eval。
6. 不建议动的：`update_interval 5→8`（配置注释明确 8 会更差）、PPO epochs/minibatches（稳定性）、
   `capture_interval 1→5`（会改 pw_noise 时间演化，训练分布漂移）。
7. **深度分辨率：t18 全程锁 106×60（cast），与评估一致**——但这是低保真近似，非部署镜像（见 §5.5）。
8. 单卡 24 GB 硬约束：训练与 eval 不能并行，挑点评测安排在训练间歇或短 run 结束后。

### 5.5 深度分辨率：`original`(cast) vs `resized`(encoder 输入) —— 两个不同的东西

| 配置 | cast 原始 `original` | encoder 实际输入 `resized` | 延迟/自遮挡/hole-fill |
|---|---|---|---|
| base / **teacher（t18 用）** | **(106, 60)** | (87, 58) | 无（干净帧） |
| student（部署镜像，swap 验证的 enc 出处） | **(480, 270)** | (87, 58) | latency[4,5] + self_geom + hole_fill |

- **`resized=(87,58)` 是 encoder 骨干输入维，硬编码在 `DepthOnlyFCBackbone58x87` 的 `Linear(64*25*39,128)`，绝不能动。**
- 可动的是 `original`（warp 光线 cast 分辨率，高 cast→下采样更清晰）。
- **决策（2026-11-01，用户拍板）**：t18 训练 + swap 评估都用 teacher 原生 **106×60**，不引入 student 的
  480×270 配置。好处：train/eval 分布自洽 + 渲染便宜（`render_on_demand` 都不用开）。
- ⚠️ **sim2real 警告**：106×60 干净帧 ≠ 真机 D435（480×270 + 80ms 延迟 + 自遮挡 + hole-fill）。
  t18 在此区间的 encoder 会**偏乐观**——只验证"一阶段流水线可跑通、latent 可对齐"，
  **不等于可直接上真机**。要真部署，最终须把深度链路切到 student 480×270 区间重训/重评。

## 6. 已知边界与风险

- t18 冒烟只有 3 iter，**encoder 在随机 actor 上的早期 BC 目标质量很差**（actor 还没学会东西时
  a_t≈无意义）——设计上可接受（目标随 actor 变好），但若 lat_err 长期不降可考虑前 200 iter 只加 latent 项。
- 480 env 全量 run（~19.7 GB 显存）不能与 swap 评估并行；挑点评测需等训练间歇或降 env。
- `EP_USE_YAW=1` 与 depth 通道不兼容（encoder 头 34 维无法当 32 维 scan latent 用），脚本已在 runner 层拦截。
- **本 run 的深度区间 = teacher 106×60 干净帧，与 §1.2 swap 验证用的 student 480×270 部署镜像不是同一分布**：
  故 t18 的 swap 数字只证明流水线在低分辨率下自洽可行，**真机部署前必须在 480×270 区间重验**（见 §5.5）。
