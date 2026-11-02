# 路 B：部分机队深度采集（6144 PPO + 480 渲染子集）

承接 `T18_COLD_START_PPO_SCALING.md` §10。那一篇的结论是：课程卡死的病根不在 PPO 超参，
而在**规模**——t18 把"大 batch PPO"和"带相机的 BC"绑进同一个 env batch，24GB 显存逼着
num_envs 降到 480（new-run-v2 的 1/13），于是 KL 测不准、adaptive 控制器失灵、课程爬不出
row 0。本篇记录解绑方案、实测数字与回退方式。

## TL;DR

1. **原链路本来就是分离的**：`camera_num_envs = 280` 与 `num_envs = 6144` 两个规格同时存在
   于 [legged_robot_config.py L107-109](legged_gym/legged_gym/envs/base/legged_robot_config.py)，
   helpers 在 `--use_camera --headless` 时把机队切到 280。绑成一炉是 t18 的设计改动。
2. **新增 `cfg.depth.num_depth_envs`**：只渲染前 nd 个 env（前缀子集），PPO 机队回到 6144。
   默认不设时行为逐字不变。
3. **实测 3.8s/iter = 38.8k steps/s**，是同规模无相机 new-run-v2（3.51s / 42.0k steps/s）的
   **92%**，是 480 env 版（7.9k）的 **4.9 倍**。3800 iter 等效样本量从 19.3 小时降到 4.0 小时。
4. **num_envs 建议：保持 6144**（显存 22.0/24.5GB，已无上升空间；想加料应该加 nd 而不是 envs）。

## 1. 实现

### 1.1 环境侧（`envs/base/legged_robot.py`，commit `1f3121e`，分支 `t18-path-b`）

- `_update_depth_buffer_on_demand()`：`nd = cfg.depth.num_depth_envs`，当 `0 < nd < num_envs`
  时把 `active` 掩码在 `[nd:]` 全部置 False（warp 侧的 gating 本来就是 sync-free 的设计），
  然后 **后处理与环形缓冲都只切前 nd 行**：

  ```python
  m = nd if partial else depth.shape[0]
  new_frame = self.process_depth_batch(depth[:m])
  buf = self.depth_buffer[:m]
  ```

  切前缀而不是任意索引是刻意的：`process_depth_batch` 内部的每 env 时序状态
  （`pw_bs_k[:N]` 盲区随机游走、`_pw_low` 孔洞场、`depth_noise_scale[:N]`）都按行号取，
  用前缀它们天然对齐；换任意行号就会把状态写到错误的 env 上。
- `__init__` 的深度分配处加了校验：`num_depth_envs < num_envs` 必须配
  `render_on_demand=True`（eager 路径每个 capture step  cast 整机队，子集省不下任何光线），
  否则直接 `ValueError`，并打印 `[depth] partial-fleet capture: envs 0..{nd-1} of {N}`。

### 1.2 训练侧（`/tmp/train_t18.py`，环境变量 `EP_T18_DEPTH_ENVS`）

```bash
EP_T18_DEPTH_ENVS=480 EP_T18_SAVEINT=2000 \
python -u /tmp/train_t18.py --task go2_teacher --use_camera \
       --num_envs 6144 --max_iterations 100000 --exptid t18-pathb-6144 --no_wandb
```

- `NBC = min(_NDE, num_envs)`：BC 快照只取 `obs[:NBC]` / `get_depth_obs()[:NBC]`，
  encoder 的 GRU hidden state 也按 `NBC` 建（`[1, NBC, H]`），done 清零同样切片。
  不设该变量时 `NBC = num_envs`，等价于旧行为。
- 强制 `render_on_demand = True`（子集采集的前提）。
- **顺带修掉一个隐性问题**：`override_terrain_mix_for_camera = False` +
  `override_terrain_grid_for_camera = False`。helpers 只要看到 `use_camera` 就会把
  `terrain.y_range` 从 teacher 的 `[-0.4, 0.4]` 改写成 `[-0.1, 0.1]`
  （[helpers.py L177-179](legged_gym/legged_gym/utils/helpers.py)，**非 headless 也生效**），
  headless 下还会进一步把机队切到 280、地形网格切到 10×20、地形 mix 换成标准 parkour 集。
  student 配置早就为同样的原因把这两个开关关掉了。run1~run6 都吃到了 y_range 的改写。

## 2. 速度实测

### 2.1 对照表

| 配置 | iter 时间 | 样本/iter | steps/s | 显存 |
|---|---|---|---|---|
| new-run-v2（6144 env，**无相机**，wandb 4000 iter 中位） | 3.26s 采集 + 0.26s 学习 = **3.51s** | 147k | **42.0k**（p10 40.7k / p90 43.2k） | — |
| **路 B（6144 env + 480 渲染子集）** | **3.8s**（roll 3.5 / bc 0.07 / upd 0.23） | 147k | **38.8k** | 22.0 GB |
| 旧 t18（480 env 全相机，24 步） | 1.45s | 11.5k | 7.9k | 18.1 GB |

**相机子集的净代价只有 8%**。注意 new-run-v2 的 `Perf/*` 是从
`logs/wandb/run-20261028_112754-*/run-*.wandb` 二进制里解析出来的（该 run 没开 TB）。

### 2.2 iter 时间是一条瞬态曲线，别按开局数字下结论

```
iter   0 →  8.5s      随机 episode 相位 + 全员站立的接触峰值
iter   6 → 14.0s      峰值
iter  32 → 10.2s
iter  55 →  4.1s      大规模 reset 之后机器狗普遍倒地/休眠
iter 83-90 → 3.6-3.9s 平台
iter 272 → 5.3-5.7s    课程回到 row 0、能站住的 env 变多，接触量回升
```

GPU sm 利用率全程只有 6–34%、CPU ~1.6 核 → **不是算力受限**，是 launch/同步串行。

### 2.3 还能榨的（都记着没做）

1. **warp 的 launch 规模仍按整机队**：[warp_cam.py](legged_gym/legged_gym/utils/warp_cam.py)
   `dim = self.num_envs * self.num_pixels` = 6144×6360 = **39.1M 线程/次 cast**，子集靠 kernel
   内部早退；stereo 恢复段的 `torch.rand_like(d)` 与三次 elementwise 同样跑满 39M 像素；
   `get_depth_obs()` 每 `update_interval` 步 clone 一次 `[6144,58,87]`（124MB）。
   改成"相机按 nd 建 + 输入切 `[:nd]`"预计省 0.2–0.3s/iter（5–8%）。**没做**：8% 不值一次重启。
2. `nd 480 → 240` 再省 ~0.15s/iter，但 BC 样本减半——不划算。
3. 每次 render 一个 `wp.synchronize()`（24 次/iter）是 sm 利用率低的根因，要动就得改异步/双缓冲，
   风险远大于收益。

## 3. num_envs 建议：保持 6144

- **显存**：22.0/24.5GB。固定开销大头是 trimesh 地形（1004 万顶点 / 2008 万三角面）+ warp BVH +
  PhysX 接触缓冲（`max_gpu_contact_pairs=2**24`，teacher 还把 `default_buffer_size_multiplier`
  提到 10 专供飞球墙）。8192 env 要把 obs/动作/刚体状态线性 ×1.33，余量不够；7168 也换不来吞吐
  （roll 等比例拉长，3.8s→4.4s，steps/s 基本不变）。
- **标定**：`num_mini_batches=4` / `desired_kl=0.01` / adaptive lr 全是按 6144 调的，改规模等于
  重新引入 `T18_COLD_START_PPO_SCALING.md` 里那一整套噪声问题。
- **真要加料就加 nd**：480→960 约 +0.1s/iter、+0.3GB，BC 样本翻倍，且缓解下面第 4 条的多样性问题。

## 4. 已知妥协：子集是"前 nd 行"

`terrain_types = arange(N) // (N/num_cols)`，6144/80 = 76.8，所以 env 0..479 只落在
**7 个列（tile 变体）** 上；乘以自己的课程行数，BC 每炉最多见到约 70 个不同的墙角/栏高组合。
这是"前缀切片才能让每 env 时序状态对齐"的代价。要全 80 列覆盖，得把子集改成跨列散布并让
`process_depth_batch` 接受行索引——留待 nd 扩容时一并处理。

## 5. git / 推送状态

| 项 | 值 |
|---|---|
| 分支 | `t18-path-b`（路 B 改动只在这里） |
| 路 B 提交 | `1f3121e` legged_robot.py（+35/−9） |
| 回退锚点 | tag `pre-pathb` = `69a7f34` = 本地 main |
| 远端 | `69a7f34` 已推到 `origin/main`（经 ghfast 镜像；github.com 直连在本机超时） |
| 撤销方式 | `git checkout pre-pathb -- legged_gym/legged_gym/envs/base/legged_robot.py` 或 `git checkout main` |

推送注意：`git remote get-url origin` 是 `https://yunhanzheng@github.com/...`，不带 token 时会
**卡在凭据交互**上（表现为命令超时）；本机能用的通路是
`git push https://x-access-token:<TOKEN>@ghfast.top/https://github.com/<owner>/<repo>.git main:main`。
凭据不落盘、不写进 `.git/config`。**不要用 `git checkout .` 回退**：`logs/extreme-parkour/*/model_*.pt`
是被 git 跟踪的，会连带把 ckpt 覆盖掉。

## 6. 当前 run 与判据

`t18-pathb-6144`（pid 358296，TB 借平台 6007，`logs/extreme-parkour/t18-pathb-6144/tb/`）。
本 run 与 new-run-v2 **同机队、同 24 步/iter**，所以 iter 编号在样本量上可直接对标。

1. **首次晋升**：iter 272 仍是 `LVL mean=0.00 max=0`。40M 步（=v2 的 iter 272 位置）还没有晋升，
   但本仓库**没有任何 6144 冷启动先例**（血脉是 flat_baseline → new-run → new-run-v2 的接力
   warm-start），所以没有可引用的基线；把 **iter 1000**（≈147M 步）设为硬判据点，届时若仍
   `max=0`，则冷启动在这条规格上也不成立，回头重议热启动。
2. **lr 带**：iter 272 为 8.78e-4，已落在健康先例的范围内（new-run 峰值 8.65e-4、v2 峰值 3.84e-4），
   **没有** run3/run4 那种顶到 1e-2 天花板的失控——这是路 B 有效性的第一个信号。
3. **critic 健康**：value 0.027、est 0.043（est 偏低是任务回报本身难，待与晋升一起看）。
4. **BC**：`lat_err` 0.10–0.26、`act_err` 随 actor 变强而抬升（0.64），符合"encoder 追动靶"预期。

## 7. 踩坑记录

- 编辑工具对 `legged_robot.py` 报 `save file ... failed, reason: unknown`，但**内容实际已写入**，
  重试导致 init 块的校验代码被写了两遍（表现为启动日志打印两条 partial-fleet 提示）。
  改这个文件一律用 `python - <<'EOF'` 的字符串替换 + `assert count==1`，改完必须
  `grep -c` 复核，不要相信工具返回码。
- 大改 `num_envs` 前先跑 `--max_iterations 3` 冒烟：路 B 的显存、iter 时间、BC 切片正确性
  都是在这 3 个 iter 里暴露的（启动仍需 ~4 分钟）。
