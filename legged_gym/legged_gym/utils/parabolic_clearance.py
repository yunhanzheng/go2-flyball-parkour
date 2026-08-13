"""Exact convex-overlap tests for hurdle clearance parabolic prisms.

The keep-out volume is the parabola hypograph extruded along world Y::

    z_floor <= z <= z_floor + h * (1 - ((x - x_center) / a) ** 2)
    |x - x_center| <= a
    y_lo <= y <= y_hi

It is convex, so collision against the robot's convex URDF collision
primitives can be determined with GJK using analytical support mappings.  The
test is performed against the actual Isaac collision primitives (boxes,
spheres, and cylinders converted to capsules), not their rigid-body origins.
"""

import torch


BOX, CAPSULE, SPHERE = 0, 1, 2


def quat_to_matrix(quat):
    """Convert xyzw quaternions to rotation matrices."""
    x, y, z, w = quat.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack((
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
    ), dim=-1).reshape(quat.shape[:-1] + (3, 3))


def collision_primitive_world_poses(body_pos, body_quat, body_indices,
                                    offset_pos, offset_rot):
    """Return [N,K] world poses for collision primitives attached to bodies."""
    pos = body_pos[:, body_indices]
    body_rot = quat_to_matrix(body_quat[:, body_indices])
    prim_pos = pos + torch.matmul(
        body_rot, offset_pos.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
    prim_rot = torch.matmul(body_rot, offset_rot.unsqueeze(0))
    return prim_pos, prim_rot


def _safe_normalize(v):
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1.0e-12)


def _support_primitive(direction, pos, rot, primitive_type, size):
    """Analytical support points for world-space boxes/capsules/spheres."""
    local_d = torch.matmul(rot.transpose(-1, -2), direction.unsqueeze(-1)).squeeze(-1)
    local_n = _safe_normalize(local_d)

    box_point = torch.where(local_d >= 0.0, size, -size)

    capsule_point = local_n * size[:, 0:1]
    capsule_point[:, 2] += torch.where(
        local_d[:, 2] >= 0.0, size[:, 1], -size[:, 1])

    sphere_point = local_n * size[:, 0:1]

    local = torch.where(
        (primitive_type == BOX).unsqueeze(1), box_point,
        torch.where((primitive_type == CAPSULE).unsqueeze(1),
                    capsule_point, sphere_point))
    return pos + torch.matmul(rot, local.unsqueeze(-1)).squeeze(-1)


def _support_parabolic_prism(direction, bounds):
    """Analytical support point for the convex parabolic prism in ``bounds``."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    half_span = 0.5 * (hi[:, 0] - lo[:, 0])
    center_x = 0.5 * (hi[:, 0] + lo[:, 0])
    height = hi[:, 2] - lo[:, 2]
    dx, dy, dz = direction.unbind(1)

    # For an upward support direction, maximize
    # dx*x + dz*h*(1-x^2/a^2).  For a horizontal/downward direction the
    # floor edge is the maximizer.  All candidate pairs have positive a/h.
    upward = dz > 1.0e-12
    stationary_x = dx * half_span.square() / (
        2.0 * dz.clamp_min(1.0e-12) * height)
    edge_x = torch.where(dx >= 0.0, half_span, -half_span)
    rel_x = torch.where(
        upward, torch.clamp(stationary_x, min=-half_span, max=half_span), edge_x)
    rel_z = torch.where(
        upward, height * (1.0 - (rel_x / half_span).square()),
        torch.zeros_like(height))
    y = torch.where(dy >= 0.0, hi[:, 1], lo[:, 1])
    return torch.stack((center_x + rel_x, y, lo[:, 2] + rel_z), dim=1)


def _triple_cross(a, b, c):
    return torch.cross(torch.cross(a, b, dim=1), c, dim=1)


def _gjk_intersects(primitive_pos, primitive_rot, primitive_type, primitive_size,
                    bounds, max_iterations=32):
    """Batched GJK intersection for matched primitive/prism pairs."""
    count_pairs = primitive_pos.shape[0]
    if count_pairs == 0:
        return torch.zeros(0, dtype=torch.bool, device=primitive_pos.device)

    def support(direction, ids=None):
        if ids is None:
            pos = primitive_pos
            rot = primitive_rot
            ptype = primitive_type
            size = primitive_size
            pair_bounds = bounds
        else:
            pos = primitive_pos[ids]
            rot = primitive_rot[ids]
            ptype = primitive_type[ids]
            size = primitive_size[ids]
            pair_bounds = bounds[ids]
        return (_support_primitive(direction, pos, rot, ptype, size)
                - _support_parabolic_prism(-direction, pair_bounds))

    prism_center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    direction = prism_center - primitive_pos
    coincident = direction.square().sum(1) < 1.0e-16
    direction[coincident] = direction.new_tensor((1.0, 0.0, 0.0))

    simplex = primitive_pos.new_zeros((count_pairs, 4, 3))
    simplex[:, 0] = support(direction)
    simplex_size = torch.ones(count_pairs, dtype=torch.long, device=primitive_pos.device)
    direction = -simplex[:, 0]
    active = torch.ones(count_pairs, dtype=torch.bool, device=primitive_pos.device)
    intersects = torch.zeros_like(active)

    def mark_zero_direction(ids):
        zero = direction[ids].square().sum(1) <= 1.0e-16
        if zero.any():
            hit_ids = ids[zero]
            intersects[hit_ids] = True
            active[hit_ids] = False
        return ids[~zero]

    def handle_line(ids):
        if ids.numel() == 0:
            return
        a = simplex[ids, 0]
        b = simplex[ids, 1]
        ao = -a
        ab = b - a
        toward = (ab * ao).sum(1) > 0.0

        point_ids = ids[~toward]
        if point_ids.numel():
            simplex_size[point_ids] = 1
            direction[point_ids] = -simplex[point_ids, 0]

        line_ids = ids[toward]
        if line_ids.numel():
            la = simplex[line_ids, 0]
            lab = simplex[line_ids, 1] - la
            lao = -la
            direction[line_ids] = _triple_cross(lab, lao, lab)
            mark_zero_direction(line_ids)

    def handle_triangle(ids):
        if ids.numel() == 0:
            return
        a = simplex[ids, 0]
        b = simplex[ids, 1]
        c = simplex[ids, 2]
        ao = -a
        ab, ac = b - a, c - a
        abc = torch.cross(ab, ac, dim=1)

        outside_ac = (torch.cross(abc, ac, dim=1) * ao).sum(1) > 0.0
        ac_ids = ids[outside_ac]
        if ac_ids.numel():
            aa = simplex[ac_ids, 0]
            aac = simplex[ac_ids, 2] - aa
            aao = -aa
            toward_ac = (aac * aao).sum(1) > 0.0
            keep_ac = ac_ids[toward_ac]
            if keep_ac.numel():
                simplex[keep_ac, 1] = simplex[keep_ac, 2]
                simplex_size[keep_ac] = 2
                edge = simplex[keep_ac, 1] - simplex[keep_ac, 0]
                direction[keep_ac] = _triple_cross(edge, -simplex[keep_ac, 0], edge)
                mark_zero_direction(keep_ac)
            check_ab = ac_ids[~toward_ac]
            if check_ab.numel():
                simplex_size[check_ab] = 2
                handle_line(check_ab)

        remaining = ids[~outside_ac]
        if remaining.numel() == 0:
            return
        ra = simplex[remaining, 0]
        rab = simplex[remaining, 1] - ra
        rac = simplex[remaining, 2] - ra
        rao = -ra
        rabc = torch.cross(rab, rac, dim=1)
        outside_ab = (torch.cross(rab, rabc, dim=1) * rao).sum(1) > 0.0
        ab_ids = remaining[outside_ab]
        if ab_ids.numel():
            simplex_size[ab_ids] = 2
            handle_line(ab_ids)

        face_ids = remaining[~outside_ab]
        if face_ids.numel():
            fa = simplex[face_ids, 0]
            fab = simplex[face_ids, 1] - fa
            fac = simplex[face_ids, 2] - fa
            fao = -fa
            fabc = torch.cross(fab, fac, dim=1)
            above = (fabc * fao).sum(1) > 0.0
            direction[face_ids] = torch.where(
                above.unsqueeze(1), fabc, -fabc)
            flip_ids = face_ids[~above]
            if flip_ids.numel():
                old_b = simplex[flip_ids, 1].clone()
                simplex[flip_ids, 1] = simplex[flip_ids, 2]
                simplex[flip_ids, 2] = old_b
            mark_zero_direction(face_ids)

    def handle_tetrahedron(ids):
        if ids.numel() == 0:
            return
        a, b = simplex[ids, 0], simplex[ids, 1]
        c, d = simplex[ids, 2], simplex[ids, 3]
        ao = -a
        ab, ac, ad = b - a, c - a, d - a
        abc = torch.cross(ab, ac, dim=1)
        acd = torch.cross(ac, ad, dim=1)
        adb = torch.cross(ad, ab, dim=1)

        outside_abc = (abc * ao).sum(1) > 0.0
        abc_ids = ids[outside_abc]
        if abc_ids.numel():
            simplex_size[abc_ids] = 3
            handle_triangle(abc_ids)

        remaining = ids[~outside_abc]
        if remaining.numel():
            ri = torch.nonzero(~outside_abc, as_tuple=False).squeeze(1)
            outside_acd = (acd[ri] * ao[ri]).sum(1) > 0.0
            acd_ids = remaining[outside_acd]
            if acd_ids.numel():
                simplex[acd_ids, 1] = simplex[acd_ids, 2].clone()
                simplex[acd_ids, 2] = simplex[acd_ids, 3].clone()
                simplex_size[acd_ids] = 3
                handle_triangle(acd_ids)

            last = remaining[~outside_acd]
            if last.numel():
                rri = ri[~outside_acd]
                outside_adb = (adb[rri] * ao[rri]).sum(1) > 0.0
                adb_ids = last[outside_adb]
                if adb_ids.numel():
                    old_b = simplex[adb_ids, 1].clone()
                    simplex[adb_ids, 1] = simplex[adb_ids, 3]
                    simplex[adb_ids, 2] = old_b
                    simplex_size[adb_ids] = 3
                    handle_triangle(adb_ids)
                inside_ids = last[~outside_adb]
                if inside_ids.numel():
                    intersects[inside_ids] = True
                    active[inside_ids] = False

    for _ in range(max_iterations):
        ids = torch.nonzero(active, as_tuple=False).squeeze(1)
        if ids.numel() == 0:
            break
        ids = mark_zero_direction(ids)
        if ids.numel() == 0:
            continue

        new_point = support(direction[ids], ids)
        # A negative support projection is a separating plane.  Keep a small
        # numerical cushion so floating-point noise cannot create a false gap.
        separated = (new_point * direction[ids]).sum(1) < -1.0e-10
        sep_ids = ids[separated]
        active[sep_ids] = False
        add_ids = ids[~separated]
        if add_ids.numel() == 0:
            continue

        simplex[add_ids, 1:] = simplex[add_ids, :3].clone()
        simplex[add_ids, 0] = new_point[~separated]
        simplex_size[add_ids] += 1

        line_ids = add_ids[simplex_size[add_ids] == 2]
        tri_ids = add_ids[simplex_size[add_ids] == 3]
        tetra_ids = add_ids[simplex_size[add_ids] == 4]
        handle_line(line_ids)
        handle_triangle(tri_ids)
        handle_tetrahedron(tetra_ids)

    # Failure to find a separating plane after the fixed iteration budget is
    # conservatively classified as overlap: clearance penalties must not have
    # false negatives.  In ordinary configurations GJK resolves in <8 steps.
    intersects[active] = True
    return intersects


def primitive_parabolic_prism_overlaps(primitive_pos, primitive_rot,
                                       primitive_type, primitive_size, bounds):
    """Return ``[N,K,S]`` overlap flags for collision primitives and prisms.

    ``bounds`` is ``[N,S,2,3]`` and stores each parabola's exact x roots,
    padded y limits, floor z, and apex z. Degenerate slots remain sentinels.
    A conservative bounding-sphere broad phase is followed by exact GJK only
    for candidates, keeping the per-step cost proportional to nearby hurdles.
    """
    n_env, n_primitive = primitive_pos.shape[:2]
    n_slot = bounds.shape[1]
    lo, hi = bounds[:, :, 0], bounds[:, :, 1]
    real = (hi[..., 0] > lo[..., 0]) & (hi[..., 2] > lo[..., 2])

    box_radius = torch.linalg.vector_norm(primitive_size, dim=1)
    capsule_radius = primitive_size[:, 0] + primitive_size[:, 1]
    sphere_radius = primitive_size[:, 0]
    radius = torch.where(
        primitive_type == BOX, box_radius,
        torch.where(primitive_type == CAPSULE, capsule_radius, sphere_radius))

    pos = primitive_pos.unsqueeze(2)
    rad = radius.view(1, n_primitive, 1, 1)
    candidate = (
        (pos >= lo.unsqueeze(1) - rad) &
        (pos <= hi.unsqueeze(1) + rad)
    ).all(dim=-1) & real.unsqueeze(1)

    result = torch.zeros(
        (n_env, n_primitive, n_slot), dtype=torch.bool, device=primitive_pos.device)
    candidate_idx = torch.nonzero(candidate, as_tuple=False)
    if candidate_idx.numel() == 0:
        return result

    env_idx, primitive_idx, slot_idx = candidate_idx.unbind(1)
    hit = _gjk_intersects(
        primitive_pos[env_idx, primitive_idx],
        primitive_rot[env_idx, primitive_idx],
        primitive_type[primitive_idx],
        primitive_size[primitive_idx],
        bounds[env_idx, slot_idx])
    result[env_idx, primitive_idx, slot_idx] = hit
    return result
