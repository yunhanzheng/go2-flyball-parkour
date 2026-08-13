"""Deployment-compatible RealSense depth hole filling.

This is the implementation from Deploy-an-RL-policy-on-the-Unitree-Go2-robot
``src/deploy_rl_policy/scripts/distill_common.py``. Keep the algorithm aligned
with that file: the real camera supplies positive metres and invalid pixels as
zero, while the simulator adapter converts to/from that convention.
"""

import numpy as np


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
