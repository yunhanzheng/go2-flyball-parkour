"""Microbenchmark: cv2 hole_fill vs _hole_fill_gpu at the teacher's real resolution."""
import isaacgym  # noqa: F401  must precede torch
import sys
import time

sys.path.insert(0, "/home/ubuntu22/data/go2-flyball-parkour/rsl_rl")
sys.path.insert(0, "/home/ubuntu22/data/go2-flyball-parkour/legged_gym")

import numpy as np
import torch

import legged_gym.envs  # noqa: F401  initialises the task registry in the usual order
from legged_gym.utils.depth_hole_fill import hole_fill

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
FAR, NEAR = 2.0, 0.0


def make_batch(n, h, w, hole_frac=0.30, seed=0):
    """Negative-in-front depth, with `hole_frac` of pixels marked invalid (== -far)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.rand(n, h, w, generator=g) * 1.8 + 0.2
    mask = torch.rand(n, h, w, generator=g) < hole_frac
    pos[mask] = FAR
    return (-pos).to(DEV)


def cpu_path(depth):
    positive = (-depth).detach().cpu().numpy()
    invalid = positive >= FAR - 1e-4
    positive[invalid] = 0.0
    filled = [hole_fill(f, 1, 8) for f in positive]
    return -torch.from_numpy(np.stack(filled)).to(device=depth.device, dtype=depth.dtype)


def gpu_path(depth):
    from torch.nn import functional as F
    pos = -depth
    pos = torch.where(pos >= FAR - 1e-4, torch.zeros_like(pos), pos)
    for _ in range(8):
        inv = pos <= 0
        cand = F.max_pool2d(pos.unsqueeze(1), 3, 1, 1).squeeze(1)
        pos = torch.where(inv & (cand > 0), cand, pos)
    return -pos


def bench(fn, depth, iters=10, warm=3):
    for _ in range(warm):
        fn(depth)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(depth)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, out


print(f"device = {DEV}")
for n, h, w, tag in [(280, 60, 106, "warp 原生 106x60（teacher 实际分辨率）"),
                     (280, 270, 480, "deploy 上采样 480x270")]:
    depth = make_batch(n, h, w)
    t_cpu, out_cpu = bench(cpu_path, depth)
    t_gpu, out_gpu = bench(gpu_path, depth)
    same = torch.equal(out_cpu, out_gpu)
    print(f"\n{tag}   N={n} H={h} W={w}")
    print(f"  cpu(cv2)  : {t_cpu*1000:8.1f} ms/step")
    print(f"  gpu(pool) : {t_gpu*1000:8.1f} ms/step   ({t_cpu/max(t_gpu,1e-9):6.1f}x)")
    print(f"  逐位相同  : {same}")
