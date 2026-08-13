"""Robot self-geometry (the robot's own body) for the Warp depth camera.

The depth student is trained on a ray-cast against the terrain mesh only, so it
never sees its own legs -- while the real D435 does. This module extracts the
robot's COLLISION primitives from its URDF and expresses each one in the frame of
the rigid body that survives Isaac Gym's fixed-joint collapse. That lets the
depth ray-caster intersect the robot's own body using nothing but the per-step
rigid body state tensor (already on the GPU), with no mesh, no BVH and no refit.

Why collision primitives and not the visual meshes: the Go2 URDF's collision
geometry is 100% boxes / cylinders / spheres, so every primitive has a closed-form
ray intersection that costs nothing next to a BVH traversal (measured: 15
primitives add < 0.1 ms to a 32 ms / 1024-env frame -- the ray-cast is
memory-latency bound and the primitive math hides in the stalls). The visual
meshes would be more faithful to what the camera actually sees, but they cost a
per-env dynamic mesh + per-step BVH refit. The collision shells hug the visual
ones closely enough for a 106x58 depth image whose noise model already erodes
silhouettes by 1-2 px.

Two Isaac Gym asset options change what the primitives mean, and both are
honoured here:
  * collapse_fixed_joints (default True) -- links joined by a fixed joint are
    merged into their parent unless the joint carries dont_collapse="true", so
    they have no rigid body state of their own. Their primitives are composed
    through the fixed joints onto the nearest surviving ancestor.
  * replace_cylinder_with_capsule (default True) -- Isaac swaps collision
    cylinders for capsules, so modelling URDF cylinders as capsules here matches
    the simulated collision shape exactly. With the flag off, a capsule
    over-estimates the true cylinder by at most one radius at each cap.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

# Primitive type codes -- must match the constants in utils/warp_cam.py.
BOX, CAPSULE, SPHERE = 0, 1, 2
_TYPE_NAMES = {BOX: "box", CAPSULE: "capsule", SPHERE: "sphere"}


def _rpy_to_matrix(rpy):
    """URDF roll-pitch-yaw (fixed-axis xyz) -> 3x3, i.e. R = Rz(y) Ry(p) Rx(r)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def _origin_of(elem):
    """<origin xyz rpy> child -> (R, t). Missing origin = identity."""
    o = elem.find("origin") if elem is not None else None
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if o is not None:
        if o.get("xyz"):
            xyz = np.array([float(v) for v in o.get("xyz").split()])
        if o.get("rpy"):
            rpy = np.array([float(v) for v in o.get("rpy").split()])
    return _rpy_to_matrix(rpy), xyz


class SelfGeometry:
    """Flat table of K collision primitives rigidly attached to K rigid bodies.

    Primitive k rides on rigid body `body_names[k]`; its pose in that body's
    frame is (offset_rot[k], offset_pos[k]). World pose per env is therefore
        R = R_body @ offset_rot[k],  t = t_body + R_body @ offset_pos[k].

    sizes[k] is packed per type:
        BOX     -> (hx, hy, hz)       half extents
        CAPSULE -> (radius, half_len, 0)   axis is the primitive's local +z
        SPHERE  -> (radius, 0, 0)
    """

    def __init__(self, body_names, types, sizes, offset_pos, offset_rot, link_names):
        self.body_names = list(body_names)          # [K] surviving body per primitive
        self.link_names = list(link_names)          # [K] URDF link the primitive came from
        self.types = np.asarray(types, dtype=np.int32)
        self.sizes = np.asarray(sizes, dtype=np.float32).reshape(-1, 3)
        self.offset_pos = np.asarray(offset_pos, dtype=np.float32).reshape(-1, 3)
        self.offset_rot = np.asarray(offset_rot, dtype=np.float32).reshape(-1, 3, 3)

    def __len__(self):
        return len(self.body_names)

    def body_indices(self, name_to_index):
        """Map each primitive onto its index in the rigid body state tensor.

        name_to_index maps a rigid body name to its index within one env, i.e.
        what gym.find_actor_rigid_body_handle returns -- the same convention the
        env already uses for feet_indices. Raises if a body is missing or
        unresolved: silently dropping geometry would leave a hole in the depth
        image that is very hard to notice.
        """
        missing = sorted({n for n in self.body_names
                          if name_to_index.get(n, -1) is None or name_to_index.get(n, -1) < 0})
        if missing:
            raise KeyError(
                f"self-geometry references rigid bodies the simulated actor does not "
                f"have (or that failed to resolve): {missing}. "
                f"Resolved bodies: {sorted(name_to_index)}")
        return np.array([name_to_index[n] for n in self.body_names], dtype=np.int64)

    def summary(self):
        counts = {}
        for t in self.types:
            counts[_TYPE_NAMES[int(t)]] = counts.get(_TYPE_NAMES[int(t)], 0) + 1
        parts = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        return f"{len(self)} primitives ({parts}) on {len(set(self.body_names))} rigid bodies"


def load_urdf_self_geometry(urdf_path, collapse_fixed_joints=True,
                            replace_cylinder_with_capsule=True, link_filter=None,
                            strict=False):
    """Parse `urdf_path` into a SelfGeometry table.

    collapse_fixed_joints / replace_cylinder_with_capsule should mirror the
    matching cfg.asset options so the rendered shells match the simulated ones.
    link_filter, if given, is a list of substrings; only links whose name
    contains one of them contribute (e.g. ["FL_", "FR_"] for the front legs).
    strict=True rejects unsupported collision shapes instead of omitting them;
    use it when the table drives a collision/penalty calculation rather than a
    best-effort camera rendering.
    """
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found for self-geometry: {urdf_path}")
    root = ET.parse(urdf_path).getroot()

    # joint topology: child link -> (parent link, is_collapsed, R, t)
    parent_of = {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        R, t = _origin_of(j)
        collapsed = (collapse_fixed_joints and j.get("type") == "fixed"
                     and (j.get("dont_collapse") or "").lower() != "true")
        parent_of[child] = (j.find("parent").get("link"), collapsed, R, t)

    prims = dict(body_names=[], types=[], sizes=[], offset_pos=[], offset_rot=[], link_names=[])
    skipped = []
    for link in root.findall("link"):
        name = link.get("name")
        collisions = link.findall("collision")
        if not collisions:
            continue
        if link_filter is not None and not any(f in name for f in link_filter):
            continue

        # Walk up through collapsed fixed joints to the surviving ancestor,
        # accumulating the transform link->ancestor.
        host = name
        R_acc = np.eye(3)
        t_acc = np.zeros(3)
        while host in parent_of and parent_of[host][1]:
            parent, _, R_j, t_j = parent_of[host]
            t_acc = t_j + R_j @ t_acc
            R_acc = R_j @ R_acc
            host = parent

        for col in collisions:
            R_c, t_c = _origin_of(col)
            geom = col.find("geometry")
            if geom is None or len(geom) == 0:
                continue
            shape = geom[0]
            if shape.tag == "box":
                ptype = BOX
                size = np.array([float(v) for v in shape.get("size").split()]) / 2.0
            elif shape.tag == "sphere":
                ptype = SPHERE
                size = np.array([float(shape.get("radius")), 0.0, 0.0])
            elif shape.tag == "cylinder":
                # Capsule about the primitive's local +z (URDF cylinder axis).
                ptype = CAPSULE
                size = np.array([float(shape.get("radius")),
                                 float(shape.get("length")) / 2.0, 0.0])
            else:
                skipped.append(f"{name}:{shape.tag}")
                continue
            prims["body_names"].append(host)
            prims["link_names"].append(name)
            prims["types"].append(ptype)
            prims["sizes"].append(size)
            prims["offset_pos"].append(t_acc + R_acc @ t_c)
            prims["offset_rot"].append(R_acc @ R_c)

    if skipped:
        if strict:
            raise ValueError(
                "unsupported URDF collision shapes prevent an exact primitive "
                f"collision test: {sorted(set(skipped))}")
        # Meshes/planes have no closed-form ray test; loudly refuse to pretend
        # the body is fully represented.
        print(f"[self-geom] WARNING: skipped unsupported collision shapes "
              f"(no closed-form ray test): {sorted(set(skipped))}")
    if not prims["body_names"]:
        raise ValueError(f"no supported collision primitives found in {urdf_path}"
                         + (f" matching link_filter={link_filter}" if link_filter else ""))
    if not replace_cylinder_with_capsule and CAPSULE in prims["types"]:
        print("[self-geom] note: cfg.asset.replace_cylinder_with_capsule=False, but "
              "cylinders are rendered as capsules (over-estimates by <= 1 radius per cap)")
    return SelfGeometry(**prims)
