"""NVIDIA Warp depth camera for fast distillation rendering.

Replaces the Isaac Gym per-env camera loop (step_graphics +
render_all_camera_sensors + a 6144-iteration get_camera_image_gpu_tensor
loop) with a single GPU ray-cast against the terrain triangle mesh. One
thread per (env, pixel); the whole depth batch comes back as a torch tensor
that aliases Warp memory (no host round-trip).

Output convention is chosen to match Isaac Gym's IMAGE_DEPTH so the existing
LeggedRobot.process_depth_image pipeline (crop -> noise -> clip to
[-far_clip, -near_clip] -> resize -> normalize) works unchanged:
  * depth is the perpendicular distance along the camera's forward (+X) axis,
  * stored NEGATIVE (a surface 1 m in front -> -1.0),
  * ray misses -> -far_clip.

Camera frame: x forward, y left, z up (right-handed). Forward ~ robot body
+X, pitched down about body-Y by a per-env angle (matching attach_camera's
gymapi.Quat.from_euler_zyx(0, pitch, 0)). Left/right and up/down image
mirroring is self-consistent between training and sim-eval since both use
this renderer; match it to the real sensor at hardware-deploy time.

Optional stereo-shadow mode (cfg.depth.stereo_shadow): models the D435's
stereo geometry instead of treating depth as a single ideal pinhole. The
real camera registers depth to the LEFT IR imager and needs the RIGHT
imager (stereo_baseline to the right, camera-frame -Y) to see the same
surface point, otherwise matching fails and the pixel reads invalid (-> far
here). Two geometric failure modes are ray-cast exactly:
  1. half-occlusion — a 2nd ray from the right imager toward the hit point;
     any nearer geometry blocks the match (shadow band hugging the LEFT
     side of near obstacles),
  2. right-FOV bound — the point projects into the right image at
     u_R = u_L - focal*baseline/z; u_R < 0 has nothing to match against
     (depth-dependent blind band at the LEFT image edge; the geometric
     version of the pw_blindspot_* heuristic).

Both stereo failure modes can be thinned PER ENV via render(shadow_keep=...):
the kernel then stashes each shadowed pixel's ideal depth as a POSITIVE value
(valid depths are always <= 0) and render() keeps or recovers it with that
probability. This is how the per-env depth-noise curriculum ramps the shadow
from none (s=0) to full (s>=0.5), and it doubles as the honest phenomenology —
the geometric model invalidates every occluded pixel while real block matching
recovers most of them.

Optional self-geometry mode (cfg.depth.self_geometry): also intersect the
robot's OWN body, so the student sees its legs the way the real D435 does. The
body is represented by the URDF collision primitives (see utils/self_geometry.py)
transformed by the per-step rigid body states -- analytic ray tests, no mesh and
no BVH refit. Measured cost is nil: the ray-cast is memory-latency bound on BVH
traversal and the primitive math hides inside the stalls. With nprim=0 the
kernels are bit-identical to the terrain-only versions.

Requires `warp-lang` (pip install warp-lang). Tested against the warp 1.x
mesh_query_ray struct API.
"""

import numpy as np
import torch

try:
    import warp as wp
    wp.init()
    _WARP_AVAILABLE = True
except Exception as e:  # pragma: no cover - import guard
    wp = None
    _WARP_AVAILABLE = False
    _WARP_IMPORT_ERROR = e


if _WARP_AVAILABLE:

    @wp.func
    def _safe_inv(x: float) -> float:
        """1/x with the zero guarded, sign preserved (slab test needs +-inf)."""
        if x >= 0.0:
            return 1.0 / wp.max(x, 1.0e-8)
        return 1.0 / wp.min(x, -1.0e-8)


    @wp.func
    def _ray_box(o: wp.vec3, d: wp.vec3, e: wp.vec3, tmax: float) -> float:
        """Ray vs origin-centred box, half extents e, in the box's own frame."""
        t0 = 0.0
        t1 = tmax
        i = _safe_inv(d[0])
        a = (-e[0] - o[0]) * i
        b = (e[0] - o[0]) * i
        t0 = wp.max(t0, wp.min(a, b))
        t1 = wp.min(t1, wp.max(a, b))
        i = _safe_inv(d[1])
        a = (-e[1] - o[1]) * i
        b = (e[1] - o[1]) * i
        t0 = wp.max(t0, wp.min(a, b))
        t1 = wp.min(t1, wp.max(a, b))
        i = _safe_inv(d[2])
        a = (-e[2] - o[2]) * i
        b = (e[2] - o[2]) * i
        t0 = wp.max(t0, wp.min(a, b))
        t1 = wp.min(t1, wp.max(a, b))
        if t1 < t0:
            return tmax
        return t0


    @wp.func
    def _ray_sphere(o: wp.vec3, d: wp.vec3, r: float, tmax: float) -> float:
        """Ray vs origin-centred sphere (d normalized, so a == 1)."""
        b = 2.0 * wp.dot(o, d)
        c = wp.dot(o, o) - r * r
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return tmax
        sq = wp.sqrt(disc)
        t = 0.5 * (-b - sq)
        if t <= 0.0:
            t = 0.5 * (-b + sq)   # origin inside: take the exit point
        if t <= 0.0 or t >= tmax:
            return tmax
        return t


    @wp.func
    def _ray_capsule(o: wp.vec3, d: wp.vec3, r: float, h: float, tmax: float) -> float:
        """Ray vs capsule about the local +z axis, half-length h, radius r."""
        best = tmax
        a = d[0] * d[0] + d[1] * d[1]
        if a > 1.0e-12:                        # not parallel to the axis
            b = 2.0 * (o[0] * d[0] + o[1] * d[1])
            c = o[0] * o[0] + o[1] * o[1] - r * r
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sq = wp.sqrt(disc)
                t = (-b - sq) / (2.0 * a)
                if t > 0.0 and t < best and wp.abs(o[2] + t * d[2]) <= h:
                    best = t
                t = (-b + sq) / (2.0 * a)
                if t > 0.0 and t < best and wp.abs(o[2] + t * d[2]) <= h:
                    best = t
        t = _ray_sphere(o - wp.vec3(0.0, 0.0, h), d, r, best)      # +z cap
        if t < best:
            best = t
        t = _ray_sphere(o - wp.vec3(0.0, 0.0, -h), d, r, best)     # -z cap
        if t < best:
            best = t
        return best


    @wp.func
    def _self_hit(ro: wp.vec3, rd: wp.vec3, env: int, nprim: int,
                  prim_pos: wp.array2d(dtype=wp.vec3),
                  prim_rot: wp.array2d(dtype=wp.mat33),
                  prim_size: wp.array(dtype=wp.vec3),
                  prim_type: wp.array(dtype=wp.int32),
                  tmax: float) -> float:
        """Nearest hit on this env's own body within tmax, else tmax.

        Unconditional sweep -- no bounding-volume reject is possible because the
        camera mount sits INSIDE the robot's own bounding sphere.
        """
        best = tmax
        for k in range(nprim):
            R = prim_rot[env, k]
            o = wp.transpose(R) * (ro - prim_pos[env, k])
            d = wp.transpose(R) * rd
            s = prim_size[k]
            ty = prim_type[k]
            t = best
            if ty == 0:                        # BOX
                t = _ray_box(o, d, s, best)
            elif ty == 1:                      # CAPSULE
                t = _ray_capsule(o, d, s[0], s[1], best)
            else:                              # SPHERE
                t = _ray_sphere(o, d, s[0], best)
            if t < best:
                best = t
        return best


    @wp.kernel
    def _cast_depth_kernel(
        mesh: wp.uint64,
        cam_pos: wp.array(dtype=wp.vec3),       # [num_envs] camera world position
        cam_rot: wp.array(dtype=wp.mat33),      # [num_envs] world<-cam rotation
        ray_dir_cam: wp.array(dtype=wp.vec3),   # [num_pixels] normalized dir in cam frame
        num_pixels: int,
        far: float,
        nprim: int,                             # 0 = terrain only (bit-identical)
        prim_pos: wp.array2d(dtype=wp.vec3),    # [num_envs, nprim] world position
        prim_rot: wp.array2d(dtype=wp.mat33),   # [num_envs, nprim] world<-prim rotation
        prim_size: wp.array(dtype=wp.vec3),     # [nprim] packed per type
        prim_type: wp.array(dtype=wp.int32),    # [nprim] BOX / CAPSULE / SPHERE
        active: wp.array(dtype=wp.int32),       # [num_envs] 0 = skip this env
        depth_out: wp.array(dtype=wp.float32),  # [num_envs * num_pixels]
    ):
        tid = wp.tid()
        cam = tid // num_pixels
        pix = tid - cam * num_pixels

        # Per-env gate: rays of a skipped env return before any BVH traversal, so
        # a subset of the fleet can be cast without resizing the launch (which
        # would need a host-side count, i.e. a sync in the per-step hot loop).
        # Their output rows keep their previous contents. All-ones == full render.
        if active[cam] == 0:
            return

        ro = cam_pos[cam]
        rot = cam_rot[cam]
        rd = wp.normalize(rot * ray_dir_cam[pix])
        fwd = wp.normalize(rot * wp.vec3(1.0, 0.0, 0.0))
        max_t = far / wp.dot(rd, fwd)

        hit = False
        t_best = max_t
        query = wp.mesh_query_ray(mesh, ro, rd, max_t)
        if query.result:
            hit = True
            t_best = query.t

        # The robot's own body, bounded by the terrain hit so occluded parts are
        # skipped. No-op when nprim == 0.
        t_self = _self_hit(ro, rd, cam, nprim, prim_pos, prim_rot, prim_size, prim_type, t_best)
        if t_self < t_best:
            hit = True
            t_best = t_self

        if hit:
            # Perpendicular depth = range * cos(angle to optical axis).
            depth_out[tid] = -t_best * wp.dot(rd, fwd)
        else:
            depth_out[tid] = -far


    @wp.kernel
    def _cast_depth_stereo_kernel(
        mesh: wp.uint64,
        cam_pos: wp.array(dtype=wp.vec3),       # [num_envs] LEFT (reference) imager position
        cam_rot: wp.array(dtype=wp.mat33),      # [num_envs] world<-cam rotation
        ray_dir_cam: wp.array(dtype=wp.vec3),   # [num_pixels] normalized dir in cam frame
        num_pixels: int,
        img_w: int,
        far: float,
        baseline: float,                        # right imager at camera-frame -Y * baseline
        focal: float,                           # px, from horizontal_fov
        eps: float,                             # m, endpoint clearance for the shadow ray
        nprim: int,                             # 0 = terrain only (bit-identical)
        prim_pos: wp.array2d(dtype=wp.vec3),    # [num_envs, nprim] world position
        prim_rot: wp.array2d(dtype=wp.mat33),   # [num_envs, nprim] world<-prim rotation
        prim_size: wp.array(dtype=wp.vec3),     # [nprim] packed per type
        prim_type: wp.array(dtype=wp.int32),    # [nprim] BOX / CAPSULE / SPHERE
        shadow_recover: int,                    # 1 = stash mono depth as +z on shadowed px
        active: wp.array(dtype=wp.int32),       # [num_envs] 0 = skip this env
        depth_out: wp.array(dtype=wp.float32),  # [num_envs * num_pixels]
    ):
        tid = wp.tid()
        cam = tid // num_pixels
        pix = tid - cam * num_pixels

        # Per-env gate -- see _cast_depth_kernel.
        if active[cam] == 0:
            return

        ro = cam_pos[cam]
        rot = cam_rot[cam]
        rd = wp.normalize(rot * ray_dir_cam[pix])
        fwd = wp.normalize(rot * wp.vec3(1.0, 0.0, 0.0))
        max_t = far / wp.dot(rd, fwd)

        depth_out[tid] = -far  # default: ray miss / stereo-match failure
        hit = False
        t_best = max_t
        query = wp.mesh_query_ray(mesh, ro, rd, max_t)
        if query.result:
            hit = True
            t_best = query.t

        # The robot's own body. No-op when nprim == 0.
        t_self = _self_hit(ro, rd, cam, nprim, prim_pos, prim_rot, prim_size, prim_type, t_best)
        if t_self < t_best:
            hit = True
            t_best = t_self

        if hit:
            z = t_best * wp.dot(rd, fwd)        # perpendicular depth

            # Right-FOV / disparity bound: rectified pair with identical FOV,
            # so the point lands in the right image at u_R = u_L - f*b/z.
            # u_R < 0 -> nothing to match against -> invalid.
            row = pix // img_w
            u = float(pix - row * img_w)
            blocked = u - focal * baseline / z < 0.0
            if not blocked:
                # Half-occlusion: the point must also be visible from the
                # right imager. Camera-frame y is LEFT, so the right imager
                # sits at -Y * baseline. Any strictly-nearer hit along that
                # line of sight blocks the stereo match. The robot's own body
                # occludes the match exactly like terrain does, so a leg casts a
                # real stereo shadow instead of a clean silhouette.
                p = ro + rd * t_best
                r_ro = ro + rot * wp.vec3(0.0, -baseline, 0.0)
                to_p = p - r_ro
                dist = wp.length(to_p)
                reach = wp.max(dist - eps, 0.0)
                occ = wp.mesh_query_ray(mesh, r_ro, to_p / dist, reach)
                blocked = occ.result
                if not blocked:
                    ts = _self_hit(r_ro, to_p / dist, cam, nprim, prim_pos, prim_rot,
                                   prim_size, prim_type, reach)
                    if ts < reach:
                        blocked = True
            if blocked:
                # shadow_recover: stash the mono (pinhole) depth as a POSITIVE
                # value instead of writing -far. Valid depths are always <= 0,
                # so the sign alone tells the torch side "this pixel is
                # stereo-shadowed and its ideal reading is +z" — it can then
                # keep or recover the pixel per env with no second output array
                # and no RNG in the kernel. Covers BOTH failure modes
                # (right-FOV bound and half-occlusion).
                if shadow_recover != 0:
                    depth_out[tid] = z
                else:
                    depth_out[tid] = -far
            else:
                depth_out[tid] = -z


def _quat_to_matrix(quat):
    """xyzw quaternion -> [N,3,3] rotation matrix (torch)."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    R = torch.empty(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
    R[:, 0, 0] = 1.0 - (tyy + tzz); R[:, 0, 1] = txy - twz;        R[:, 0, 2] = txz + twy
    R[:, 1, 0] = txy + twz;         R[:, 1, 1] = 1.0 - (txx + tzz); R[:, 1, 2] = tyz - twx
    R[:, 2, 0] = txz - twy;         R[:, 2, 1] = tyz + twx;        R[:, 2, 2] = 1.0 - (txx + tyy)
    return R


class WarpDepthCamera:
    """Per-env pinhole depth camera rendered with Warp ray-casting."""

    def __init__(self, num_envs, depth_cfg, device):
        if not _WARP_AVAILABLE:
            raise ImportError(
                "warp-lang is required for the Warp depth camera "
                "(pip install warp-lang). Original import error: "
                f"{_WARP_IMPORT_ERROR}"
            )
        self.num_envs = num_envs
        self.device = device
        self.wp_device = wp.device_from_torch(device)
        self.cfg = depth_cfg

        self.W, self.H = int(depth_cfg.original[0]), int(depth_cfg.original[1])
        self.num_pixels = self.W * self.H
        self.far = float(depth_cfg.far_clip)

        # Geometric stereo-shadow model (see module docstring).
        self.stereo_shadow = bool(getattr(depth_cfg, "stereo_shadow", False))
        self.stereo_baseline = float(getattr(depth_cfg, "stereo_baseline", 0.050))
        self.stereo_eps = float(getattr(depth_cfg, "stereo_eps", 0.015))

        # Body-frame mount offset and per-env pitch (radians, down-positive).
        self.offset_body = torch.tensor(depth_cfg.position, device=device, dtype=torch.float32)
        lo, hi = float(depth_cfg.angle[0]), float(depth_cfg.angle[1])
        self.cam_pitch = (torch.empty(num_envs, device=device).uniform_(lo, hi)) * np.pi / 180.0
        cosp, sinp = torch.cos(self.cam_pitch), torch.sin(self.cam_pitch)
        # R_mount = R_y(pitch): rotates forward +X downward (toward -Z, z up).
        R_mount = torch.zeros(num_envs, 3, 3, device=device)
        R_mount[:, 0, 0] = cosp;  R_mount[:, 0, 2] = sinp
        R_mount[:, 1, 1] = 1.0
        R_mount[:, 2, 0] = -sinp; R_mount[:, 2, 2] = cosp
        self.R_mount = R_mount

        # Pinhole ray directions in the camera frame (normalized), one per
        # pixel, row-major [H*W]. focal from the horizontal FOV.
        hfov = np.radians(float(depth_cfg.horizontal_fov))
        focal = (self.W / 2.0) / np.tan(hfov / 2.0)
        self.focal = float(focal)
        cx, cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        vv, uu = torch.meshgrid(
            torch.arange(self.H, device=device, dtype=torch.float32),
            torch.arange(self.W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        a = (uu - cx) / focal      # column: +right
        b = (vv - cy) / focal      # row: +down
        dirs = torch.stack([torch.ones_like(a), -a, -b], dim=-1)  # x fwd, y left, z up
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        self._ray_dir_cam_t = dirs.reshape(-1, 3).contiguous()
        self._ray_dir_cam = wp.from_torch(self._ray_dir_cam_t, dtype=wp.vec3)

        # Persistent output buffer (torch-owned, aliased by warp each render).
        self._depth_flat = torch.zeros(num_envs * self.num_pixels, device=device, dtype=torch.float32)
        self.depth = self._depth_flat.view(num_envs, self.H, self.W)

        # All-envs-active gate for the full render (see render()).
        self._active_all_t = torch.ones(num_envs, device=device, dtype=torch.int32)
        self._active_all = wp.from_torch(self._active_all_t, dtype=wp.int32)
        self._active_t = torch.ones(num_envs, device=device, dtype=torch.int32)
        self._active_wp = wp.from_torch(self._active_t, dtype=wp.int32)

        self.wp_mesh = None

        # Self-geometry (own body) -- off until set_self_geometry() is called.
        # nprim=0 makes the kernels bit-identical to the terrain-only versions;
        # the dummy 1-element arrays exist only because wp.launch needs valid
        # array handles for every parameter.
        self.n_self_prim = 0
        self._self_body_idx = None
        self._self_off_pos = None
        self._self_off_rot = None
        self._prim_size_wp = wp.zeros(1, dtype=wp.vec3, device=self.wp_device)
        self._prim_type_wp = wp.zeros(1, dtype=wp.int32, device=self.wp_device)
        self._prim_pos_dummy = wp.zeros((1, 1), dtype=wp.vec3, device=self.wp_device)
        self._prim_rot_dummy = wp.zeros((1, 1), dtype=wp.mat33, device=self.wp_device)

    def set_self_geometry(self, self_geom, body_index_map):
        """Enable self-body rendering from a utils.self_geometry.SelfGeometry table.

        body_index_map maps rigid body name -> index within one env, i.e. what
        gym.find_actor_rigid_body_handle returns. After this call, render()
        REQUIRES body_pos/body_quat.
        """
        # Imported here, not at module scope: legged_gym.utils.__init__ pulls in
        # isaacgym, which must be imported before torch, and this module is also
        # used standalone by the depth verification scripts.
        from legged_gym.utils.self_geometry import BOX, CAPSULE, SPHERE

        body_idx = self_geom.body_indices(body_index_map)
        # The kernel branches on these codes as literals 0/1/2 (_self_hit).
        bad = sorted({int(t) for t in self_geom.types} - {BOX, CAPSULE, SPHERE})
        if bad:
            raise ValueError(f"unknown self-geometry primitive type codes: {bad}")
        assert (BOX, CAPSULE, SPHERE) == (0, 1, 2), \
            "primitive type codes must match the literals in _self_hit"

        self.n_self_prim = len(self_geom)
        self._self_body_idx = torch.as_tensor(body_idx, dtype=torch.long, device=self.device)
        self._self_off_pos = torch.as_tensor(self_geom.offset_pos, dtype=torch.float32,
                                             device=self.device)                    # [K,3]
        self._self_off_rot = torch.as_tensor(self_geom.offset_rot, dtype=torch.float32,
                                             device=self.device)                    # [K,3,3]
        # Static per-primitive shape data: keep the torch tensors alive, since the
        # warp arrays alias their storage.
        self._prim_size_t = torch.as_tensor(self_geom.sizes, dtype=torch.float32,
                                            device=self.device).contiguous()
        self._prim_type_t = torch.as_tensor(self_geom.types, dtype=torch.int32,
                                            device=self.device).contiguous()
        self._prim_size_wp = wp.from_torch(self._prim_size_t, dtype=wp.vec3)
        self._prim_type_wp = wp.from_torch(self._prim_type_t, dtype=wp.int32)

    def _self_prim_poses(self, body_pos, body_quat):
        """World poses of every self primitive: [N,K] positions and rotations."""
        K = self.n_self_prim
        bp = body_pos[:, self._self_body_idx]                       # [N,K,3]
        bq = body_quat[:, self._self_body_idx]                      # [N,K,4]
        R_body = _quat_to_matrix(bq.reshape(-1, 4)).view(self.num_envs, K, 3, 3)
        prim_rot = torch.matmul(R_body, self._self_off_rot).contiguous()
        prim_pos = bp + torch.matmul(R_body, self._self_off_pos.unsqueeze(-1)).squeeze(-1)
        return prim_pos.contiguous(), prim_rot

    def set_mesh(self, vertices, triangles, offset_xyz):
        """Build the Warp mesh once. vertices [V,3] float, triangles [F,3] int.
        offset_xyz is added to every vertex to match the Isaac Gym trimesh
        transform (typically (-border_size, -border_size, 0))."""
        verts = np.asarray(vertices, dtype=np.float32).copy()
        verts += np.asarray(offset_xyz, dtype=np.float32)
        tris = np.asarray(triangles, dtype=np.int32).reshape(-1)
        self.wp_mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=self.wp_device),
            indices=wp.array(tris, dtype=wp.int32, device=self.wp_device),
        )

    def render(self, root_pos, base_quat, body_pos=None, body_quat=None,
               shadow_keep=None, self_on=None, active=None):
        """root_pos [N,3] world position, base_quat [N,4] xyzw. Returns the
        depth tensor [N,H,W] (also stored as self.depth).

        active [N] bool casts ONLY the envs it marks: rays of the rest return
        before any BVH traversal and their output rows keep whatever they held
        before, so the caller must consume only the rows it asked for. Gating (as
        opposed to compacting into a smaller launch) keeps the call sync-free —
        a variable launch size would need the active COUNT on the host.
        None = cast every env.

        body_pos [N,B,3] / body_quat [N,B,4] are the per-link world poses from
        the rigid body state tensor; required only when self-geometry is on.

        shadow_keep [N] in [0,1] thins the stereo shadow per env: each
        half-occluded / out-of-right-FOV pixel stays invalid with this
        probability and otherwise recovers its ideal pinhole depth. The
        geometric model invalidates EVERY occluded pixel while real block
        matching recovers most of them, so partial recovery is both the
        phenomenology and the per-env ramp (1.0 = today's full shadow).

        self_on [N] bool gates own-body rendering per env; False banishes that
        env's self primitives out of every ray's reach. Both default to None =
        no per-env gating (bit-identical to the un-gated kernels)."""
        assert self.wp_mesh is not None, "WarpDepthCamera.set_mesh must be called before render"
        if active is None:
            active_wp = self._active_all
        else:
            # Copy into the persistent int32 gate (the warp array aliases its
            # storage, so it must outlive the launch).
            self._active_t.copy_(active)
            active_wp = self._active_wp
        R_base = _quat_to_matrix(base_quat)
        R_world_cam = torch.bmm(R_base, self.R_mount).contiguous()        # [N,3,3]
        cam_pos = root_pos + torch.bmm(R_base, self.offset_body.expand(self.num_envs, 3).unsqueeze(-1)).squeeze(-1)
        cam_pos = cam_pos.contiguous()

        cam_pos_wp = wp.from_torch(cam_pos, dtype=wp.vec3)
        cam_rot_wp = wp.from_torch(R_world_cam, dtype=wp.mat33)
        depth_wp = wp.from_torch(self._depth_flat, dtype=wp.float32)

        if self.n_self_prim > 0:
            if body_pos is None or body_quat is None:
                raise ValueError("self-geometry is enabled: render() needs body_pos and "
                                 "body_quat (rigid_body_states[..., 0:3] and [..., 3:7])")
            prim_pos_t, prim_rot_t = self._self_prim_poses(body_pos, body_quat)
            if self_on is not None:
                # Per-env own-body gate: translate the primitives of "no legs"
                # envs far outside the ray range instead of branching in the
                # kernel — a miss costs the same analytic test either way.
                prim_pos_t = (prim_pos_t
                              + (~self_on).view(-1, 1, 1).to(prim_pos_t.dtype) * 1.0e6
                              ).contiguous()
            prim_pos_wp = wp.from_torch(prim_pos_t, dtype=wp.vec3)
            prim_rot_wp = wp.from_torch(prim_rot_t, dtype=wp.mat33)
        else:
            prim_pos_wp = self._prim_pos_dummy
            prim_rot_wp = self._prim_rot_dummy

        if self.stereo_shadow:
            recover = 1 if shadow_keep is not None else 0
            wp.launch(
                kernel=_cast_depth_stereo_kernel,
                dim=self.num_envs * self.num_pixels,
                inputs=[self.wp_mesh.id, cam_pos_wp, cam_rot_wp, self._ray_dir_cam,
                        self.num_pixels, self.W, self.far,
                        self.stereo_baseline, self.focal, self.stereo_eps,
                        self.n_self_prim, prim_pos_wp, prim_rot_wp,
                        self._prim_size_wp, self._prim_type_wp, recover,
                        active_wp, depth_wp],
                device=self.wp_device,
            )
            if recover:
                wp.synchronize()
                # Positive values mark the stereo-shadowed pixels and carry
                # their ideal depth (see the kernel). Keep each with
                # probability shadow_keep, else hand back the pinhole reading.
                # Rows that were NOT cast are excluded: they still hold their own
                # already-resolved values from an earlier call, and re-running the
                # sign flip on them would corrupt them.
                d = self.depth
                shadow = d > 0.0
                if active is not None:
                    shadow = shadow & active.view(-1, 1, 1)
                keep = torch.rand_like(d) < shadow_keep.view(-1, 1, 1)
                far_t = torch.full_like(d, -self.far)
                d.copy_(torch.where(shadow, torch.where(keep, far_t, -d), d))
                return self.depth
        else:
            wp.launch(
                kernel=_cast_depth_kernel,
                dim=self.num_envs * self.num_pixels,
                inputs=[self.wp_mesh.id, cam_pos_wp, cam_rot_wp, self._ray_dir_cam,
                        self.num_pixels, self.far,
                        self.n_self_prim, prim_pos_wp, prim_rot_wp,
                        self._prim_size_wp, self._prim_type_wp,
                        active_wp, depth_wp],
                device=self.wp_device,
            )
        wp.synchronize()
        return self.depth
