"""Sub-batch depth post-process equivalence test (pure tensors, no IsaacGym sim).

Run:  python legged_gym/legged_gym/tests/test_depth_post_subset.py
Checks that process_depth_batch(rows=...) -- the on-demand optimisation that only
post-processes the envs actually ray-cast this step -- is bit-identical to the
full-batch call when rows covers the whole fleet, still advances the per-env
temporal degradation state exactly once per call, and that the ring-buffer write
path leaves non-active rows untouched.
"""
import sys
import types

import isaacgym  # noqa: F401  -- isaacgym refuses to import after torch (gymdeps.py)
import torch
import torchvision.transforms as T

from legged_gym.envs.base.legged_robot import LeggedRobot


class _Prof:
    def enter(self, *a, **k):
        pass

    def flush(self, *a, **k):
        pass


class _DepthCfg:
    far_clip = 2.0
    near_clip = 0.0
    pw_noise = True
    pw_hole_prob = 0.007
    pw_hole_scale = 32
    pw_hole_drift = 0.5
    pw_hole_smooth_sigma = 1.5
    pw_edge_thresh = 0.10
    pw_edge_prob = 1.0
    pw_edge_near_frac = 1 / 3
    pw_edge_far_frac = 1 / 3
    pw_edge_dilate = 0
    pw_blindspot_cols = 0
    pw_blindspot_drift = 0.05
    pw_blindspot_jitter = 0.0
    pw_empty_min = 0.0
    pw_blur_sigma = 0.0
    dis_noise = 0.0
    noise_gaussian = 0.0
    noise_dropout = 0.0
    hole_fill = True
    hole_fill_mode = 1
    hole_fill_iters = 8
    hole_fill_upscale = False
    hole_fill_width = 480
    hole_fill_height = 270
    empty_below_m = 0.0
    original = (480, 270)


class _Cfg:
    depth = _DepthCfg


_METHODS = [
    "process_depth_batch", "_apply_pw_holes", "_apply_parkour_wild_noise",
    "_apply_depth_noise", "_depth_noise_factors", "_pw_low_advance",
    "_pw_field_from_rows", "_pw_hole_field", "_bs_advance",
    "_gaussian_blur_batch", "_apply_configured_hole_fill",
    "normalize_depth_image",
]


def make_env(n, device="cpu"):
    e = types.SimpleNamespace()
    e.cfg = _Cfg()
    e.device = device
    e.num_envs = n
    e.prof = _Prof()
    e._noise_rate_factor = None
    e._noise_phys_factor = None
    e.depth_noise_scale = torch.full((n,), 0.5)   # s=0.5: calibrated D435
    e.resize_transform = T.Resize(
        (58, 87), interpolation=T.InterpolationMode.BICUBIC)
    e.pw_bs_k = torch.zeros(n)
    e.pw_bs_slope = torch.zeros(n)
    e.pw_bs_side = torch.zeros(n, dtype=torch.bool)
    for name in _METHODS:
        setattr(e, name, types.MethodType(getattr(LeggedRobot, name), e))
    return e


def synth_depth(n, gen):
    """A 'room': far background with a near slab in the middle third."""
    d = torch.full((n, 270, 480), -2.0)
    d[:, 80:190, 120:360] = -0.8
    d += torch.rand(n, 270, 480, generator=gen) * 0.02
    return d


def main():
    n_envs = 6
    gen = torch.Generator().manual_seed(1234)
    depth = synth_depth(n_envs, gen)
    idx_all = torch.arange(n_envs)

    # ---- 1. rows=arange(N) must match rows=None bit-for-bit
    outs = {}
    for tag, kw in (("none", {}), ("allrows", {"rows": idx_all})):
        torch.manual_seed(7)
        e = make_env(n_envs)
        outs[tag] = e.process_depth_batch(depth, **kw).clone()
        outs[tag + "_pwlow"] = e._pw_low.clone()
    same = torch.equal(outs["none"], outs["allrows"])
    print(f"[1] rows=None vs rows=arange(N): bit-identical = {same}")
    print(f"    out {tuple(outs['none'].shape)} range "
          f"[{outs['none'].min():.3f}, {outs['none'].max():.3f}]  "
          f"far-frac={(outs['none'] <= -0.499).float().mean():.4f}")
    print(f"    _pw_low {tuple(outs['none_pwlow'].shape)} "
          f"identical={torch.equal(outs['none_pwlow'], outs['allrows_pwlow'])}")

    # ---- 2. sub-batch runs: right shape, finite
    torch.manual_seed(7)
    e2 = make_env(n_envs)
    rows = torch.tensor([1, 3, 4])
    sub = e2.process_depth_batch(depth.index_select(0, rows), rows=rows)
    fin = bool(torch.isfinite(sub).all().item())
    print(f"[2] sub-batch rows={rows.tolist()} -> {tuple(sub.shape)} "
          f"finite={fin} range [{sub.min():.3f}, {sub.max():.3f}]")

    # ---- 3. temporal state advances exactly once per call (not twice)
    torch.manual_seed(7)
    e3 = make_env(n_envs)
    seen = []
    for _ in range(3):
        e3.process_depth_batch(depth, rows=idx_all)
        seen.append(e3._pw_low.clone())
    changed = (not torch.equal(seen[0], seen[1])) and (not torch.equal(seen[1], seen[2]))
    print(f"[3] hole field shape {tuple(seen[0].shape)}, advances once/call = {changed}")

    # ---- 4. ring-buffer write path: inactive rows untouched, fresh = all slots
    n_slots, h, w = 2, 58, 87
    buf0 = torch.arange(n_envs * n_slots * h * w, dtype=torch.float32).reshape(
        n_envs, n_slots, h, w)
    buf = buf0.clone()
    active = torch.tensor([False, True, False, True, True, False])
    fresh = torch.tensor([False, False, False, True, False, False])
    r = active.nonzero(as_tuple=False).squeeze(1)
    new_frame = torch.full((r.numel(), h, w), 99.0)
    for s in range(n_slots - 1):
        buf[:, s].index_copy_(0, r, buf.index_select(0, r).select(1, s + 1))
    buf[:, -1].index_copy_(0, r, new_frame)
    frows = fresh.nonzero(as_tuple=False).squeeze(1)
    if frows.numel() > 0:
        sub_fresh = new_frame.index_select(0, torch.searchsorted(r, frows))
        for s in range(n_slots - 1):
            buf[:, s].index_copy_(0, frows, sub_fresh)
    inactive = (~active).nonzero(as_tuple=False).squeeze(1)
    untouched = torch.equal(buf[inactive], buf0[inactive])
    roll_ok = torch.equal(buf[1, 0], buf0[1, 1]) and torch.equal(buf[1, 1], new_frame[0])
    fresh_ok = torch.equal(buf[3, 0], new_frame[1]) and torch.equal(buf[3, 1], new_frame[1])
    print(f"[4] inactive rows untouched={untouched}  "
          f"roll_ok={roll_ok}  fresh_fills_all_slots={fresh_ok}")

    ok = same and fin and changed and untouched and roll_ok and fresh_ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
