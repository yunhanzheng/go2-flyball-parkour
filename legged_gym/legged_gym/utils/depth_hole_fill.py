"""Deployment-compatible RealSense depth hole filling.

This is the implementation from Deploy-an-RL-policy-on-the-Unitree-Go2-robot
``src/deploy_rl_policy/scripts/distill_common.py``. Keep the algorithm aligned
with that file: the real camera supplies positive metres and invalid pixels as
zero, while the simulator adapter converts to/from that convention.
"""

import numpy as np


def hole_fill_batch(d, mode: int = 1, max_iter: int = 8):
    """Batched, device-agnostic equivalent of :func:`hole_fill` on [N,H,W].

    The numpy version runs the 3x3 iteration once per frame on the CPU; with
    1024 camera envs that was ~15 ms/frame, i.e. ~16 s per capture batch and
    99% of the training iteration time (382 s/iter, collection 382.2 s).

    Bit-identical to the per-frame loop:
      * same iteration count and the same 3x3 neighbourhood,
      * the same rule `out = where(invalid & cand > 0, cand, out)`,
      * mode 2's erosion is the exact dual `-max_pool(-x)`,
      * max_pool2d pads with -inf just as cv2.dilate uses
        morphologyDefaultBorderValue (so border pixels never win),
      * once a frame stops changing it keeps returning the same values, so the
        reference's `break`s only save time and are skipped here on purpose
        (no host syncs).

    `d` is positive-metres depth with 0 marking invalid pixels.
    """
    import torch
    import torch.nn.functional as F

    out = d.clone()
    if mode == 0:
        valid = out > 0
        width = out.shape[-1]
        ar = torch.arange(width, device=out.device, dtype=torch.int64)
        ar = ar.view(1, 1, -1).expand_as(out)
        idx = torch.where(valid, ar, torch.full_like(ar, -1))
        idx = torch.cummax(idx, dim=-1).values          # nearest valid column to the left
        filled = torch.gather(out, -1, idx.clamp_min(0))
        return torch.where(idx >= 0, filled, out)

    if mode == 1:
        for _ in range(max_iter):
            inv = out <= 0
            cand = F.max_pool2d(out.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
            out = torch.where(inv & (cand > 0), cand, out)
        return out

    # mode 2: nearest_from_around
    for _ in range(max_iter):
        inv = out <= 0
        tmp = torch.where(out > 0, out, torch.full_like(out, np.float32(1e6)))
        cand = -F.max_pool2d(-tmp.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        cand = torch.where(cand >= np.float32(1e6) - 1, torch.zeros_like(cand), cand)
        out = torch.where(inv & (cand > 0), cand, out)
    return out


def hole_fill(d: np.ndarray, mode: int = 1, max_iter: int = 8) -> np.ndarray:
    """Fill invalid (<=0) pixels in a positive-metres depth frame.

    Mirrors librealsense's rs2_hole_filling_filter:
      0 fill_from_left
      1 farthest_from_around (deployment default)
      2 nearest_from_around
    """
    out = np.asarray(d).astype(np.float32, copy=True)
    if mode == 0:
        valid = out > 0
        idx = np.where(valid, np.arange(out.shape[1], dtype=np.int64)[None, :], -1)
        np.maximum.accumulate(idx, axis=1, out=idx)
        rows = np.arange(out.shape[0])[:, None]
        filled = out[rows, np.where(idx < 0, 0, idx)]
        return np.where(idx >= 0, filled, out).astype(np.float32)

    import cv2
    k = np.ones((3, 3), np.uint8)
    for _ in range(max_iter):
        inv = out <= 0
        if not inv.any():
            break
        if mode == 1:
            cand = cv2.dilate(out, k)
        else:
            tmp = np.where(out > 0, out, np.float32(1e6))
            cand = cv2.erode(tmp, k)
            cand = np.where(cand >= 1e6 - 1, np.float32(0.0), cand)
        nxt = np.where(inv & (cand > 0), cand, out)
        if np.array_equal(nxt, out):
            break
        out = nxt
    return out
