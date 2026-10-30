"""The GPU hole fill must reproduce the numpy reference bit for bit.

Both backends are legal deployment reconstructions only if they are the same
function, so this pins equality over the shapes and hole patterns the student
actually produces (480x270, pw_holes + blind band + random dropouts).
"""

import importlib.util
from pathlib import Path
import types

import pytest

# isaacgym must come before torch; it refuses to load otherwise.
from isaacgym import gymapi  # noqa: F401
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "depth_hole_fill.py"
_SPEC = importlib.util.spec_from_file_location("depth_hole_fill", _MODULE_PATH)
dhf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dhf)


def _fake_env(far=5.0, mode=1, iters=8, upscale=False, w=480, h=270):
    cfg = types.SimpleNamespace(
        depth=types.SimpleNamespace(
            far_clip=far, hole_fill_mode=mode, hole_fill_iters=iters,
            hole_fill_upscale=upscale, hole_fill_width=w, hole_fill_height=h))
    return types.SimpleNamespace(cfg=cfg)


def _frames(n=8, h=270, w=480, far=5.0, hole_frac=0.06, seed=0):
    """Negative-in-front depth with invalid pixels logged as far_clip."""
    g = torch.Generator().manual_seed(seed)
    z = 0.4 + 1.6 * torch.rand(n, h, w, generator=g)
    invalid = torch.rand(n, h, w, generator=g) < hole_frac
    invalid[:, :, :6] = True          # left blind band, as the real sensor has
    invalid[:, :40, :] = True         # floor band that leaves the min range
    z = torch.where(invalid, torch.full_like(z, far), z)
    return -z


def _cpu_reference(depth, env):
    """The numpy path of _apply_configured_hole_fill, minus the cfg guards."""
    positive = (-depth).detach().cpu().numpy().copy()
    invalid = positive >= float(env.cfg.depth.far_clip) - 1e-4
    positive[invalid] = 0.0
    filled = [dhf.hole_fill(f, env.cfg.depth.hole_fill_mode,
                            env.cfg.depth.hole_fill_iters) for f in positive]
    return -torch.from_numpy(__import__("numpy").stack(filled))


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("iters", [1, 3, 8])
def test_gpu_matches_cpu(mode, iters):
    depth = _frames()
    env = _fake_env(mode=mode, iters=iters)
    assert torch.equal(LeggedRobot._hole_fill_gpu(env, depth),
                       _cpu_reference(depth, env))


def test_nothing_to_fill_is_identity():
    depth = -torch.full((2, 270, 480), 1.5)
    env = _fake_env()
    assert torch.equal(LeggedRobot._hole_fill_gpu(env, depth), depth)


def test_all_invalid_collapses_to_zero_in_both_backends():
    # The adapter logs invalid as far_clip and the filter wants 0, so a frame with
    # nothing to fill comes back as 0 -- not as far_clip. Both backends must agree
    # on that, and process_depth_batch's empty_below_m pass is what turns it back
    # into -far before the policy sees it.
    depth = -torch.full((2, 270, 480), 5.0)
    env = _fake_env()
    gpu = LeggedRobot._hole_fill_gpu(env, depth)
    assert torch.equal(gpu, _cpu_reference(depth, env))
    assert torch.all(gpu == 0)
