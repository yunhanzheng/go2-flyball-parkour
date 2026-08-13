"""Focused tests for exact collision geometry vs hurdle clearance parabola."""

import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "parabolic_clearance.py"
_SPEC = importlib.util.spec_from_file_location("parabolic_clearance", _MODULE_PATH)
pc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pc)


def _bounds(count=1):
    return torch.tensor(
        [[[-0.15, -0.55, 0.0], [0.15, 0.55, 0.30]]],
        dtype=torch.float32).repeat(count, 1, 1)


def test_parabola_support_has_exact_apex_and_floor_roots():
    bounds = _bounds(3)
    directions = torch.tensor([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    points = pc._support_parabolic_prism(directions, bounds)
    assert torch.allclose(points[0, [0, 2]], torch.tensor([0.0, 0.30]))
    assert torch.allclose(points[1, [0, 2]], torch.tensor([-0.15, 0.0]))
    assert torch.allclose(points[2, [0, 2]], torch.tensor([0.15, 0.0]))


def test_sphere_overlap_respects_curved_boundary_and_padded_width():
    centers = torch.tensor([
        [0.00, 0.00, 0.34],   # radius reaches below 30 cm apex
        [0.00, 0.00, 0.36],   # remains above apex
        [0.20, 0.00, 0.01],   # reaches through the curved shoulder
        [0.30, 0.00, 0.10],   # too far from the x root
        [0.00, 0.60, 0.20],   # reaches through padded y face
        [0.00, 0.70, 0.20],   # remains outside padded y face
    ])
    radii = torch.tensor([0.05, 0.05, 0.06, 0.05, 0.06, 0.10])
    sizes = torch.zeros((len(centers), 3))
    sizes[:, 0] = radii
    hit = pc._gjk_intersects(
        centers,
        torch.eye(3).repeat(len(centers), 1, 1),
        torch.full((len(centers),), pc.SPHERE, dtype=torch.long),
        sizes,
        _bounds(len(centers)))
    assert hit.tolist() == [True, False, True, False, True, False]


def test_full_collision_primitive_counts_when_body_origin_is_outside():
    # The box origin is beyond the +15 cm root, but its actual collision volume
    # crosses into the parabola. An origin-only reward would incorrectly miss it.
    pos = torch.tensor([[[0.20, 0.0, 0.08]]])
    rot = torch.eye(3).reshape(1, 1, 3, 3)
    bounds = _bounds().unsqueeze(0)
    hit = pc.primitive_parabolic_prism_overlaps(
        pos, rot,
        torch.tensor([pc.BOX]),
        torch.tensor([[0.08, 0.03, 0.03]]),
        bounds)
    assert hit.shape == (1, 1, 1)
    assert hit.item()


def test_degenerate_hurdle_slot_never_collides():
    pos = torch.zeros((1, 1, 3))
    rot = torch.eye(3).reshape(1, 1, 3, 3)
    hit = pc.primitive_parabolic_prism_overlaps(
        pos, rot,
        torch.tensor([pc.SPHERE]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.zeros((1, 1, 2, 3)))
    assert not hit.item()
