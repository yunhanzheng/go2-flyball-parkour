"""Flyball-wall terrain and the per-foot foothold goals it exposes.

The sequential per-foot contact formulation used here (the ``wococo_*`` rewards and
the staged plant schedule) follows WoCoCo: "Learning Whole-Body Humanoid Control with
Sequential Contacts", Zhang, Xiao, He and Shi (arXiv:2406.06005). See the
Acknowledgements section of the README for the full list of methods this repository
builds on.
"""
import os
import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation


# Foothold-goal cache, keyed by path so each table is read from disk once.
_FOOTHOLD_TABLE_CACHE = {}
_FOOTHOLD_MISSING_WARNED = set()


def _load_foothold_table(path):
    if path not in _FOOTHOLD_TABLE_CACHE:
        z = np.load(path, allow_pickle=True)
        tbl = {k: z[k] for k in z.files}
        _FOOTHOLD_TABLE_CACHE[path] = tbl
        final = bool(np.asarray(tbl['final_on_face']).item()) if 'final_on_face' in tbl else False
        order = [str(f) for f in tbl['foot_order']] if 'foot_order' in tbl else ['FL', 'FR', 'RL', 'RR']
        print(f"[flyball] FOOTHOLD TABLE: {os.path.basename(path)}")
        print(f"[flyball]   path   {path}")
        print(f"[flyball]   angles {[float(t) for t in tbl['theta_deg']]} deg"
              f"  (theta outside this span clamps to the nearest row)")
        print(f"[flyball]   mode   {'FINAL on-face targets — z-offset + rear y-shift SKIPPED' if final else 'foothold goals — z-offset + rear y-shift APPLIED'}")
        # s is in metres for the authored plate; the terrain rescales via s_frac.
        plate_l = float(np.asarray(tbl['plate_length_m']).item()) if 'plate_length_m' in tbl else 1.10
        print(f"[flyball]   s in metres along-slope for plate_length {plate_l:.2f} m, u lateral from lane centre")
        for i, t in enumerate(tbl['theta_deg']):
            row = "  ".join(
                f"{order[f]} s={tbl['s_frac_mean'][i, f] * plate_l:.3f} u={tbl['u_mean'][i, f]:+.3f}"
                for f in range(len(order)))
            print(f"[flyball]   {float(t):5.1f}° | {row}")
    return _FOOTHOLD_TABLE_CACHE[path]


def _hurdle_clear_boxes(footprints_px, hs, floor_z_m,
                        pad_x, pad_y, pad_z, x_max_m=None,
                        box_height_m=None, parabola_half_span_m=0.15):
    """Build bounds for per-hurdle extruded parabolic clearance volumes.

    The exact keep-out volume has its apex at the hurdle's x centre and is
    extruded across the hurdle footprint plus ``pad_y`` on both lateral faces.
    Its x/z cross-section is

        z = floor_z + h * (1 - ((x - x_center) / half_span) ** 2)

    so the default roots are exactly 15 cm before/after the hurdle and the
    default apex is exactly 30 cm above the floor. The returned AABB stores the
    roots, padded y limits, floor, and apex; `_reward_hurdle_clearance` performs
    the exact convex collision test rather than treating the AABB as the shape.

    ``pad_x`` and ``pad_z`` remain in the signature for saved-config/caller
    compatibility but no longer define the clearance shape. ``box_height_m``
    is retained as the configured apex height; None selects 0.30 m.

    footprints_px: list indexed BY HURDLE SLOT, each entry either None (no real
        bar in that slot -> all-zero sentinel row) or
        (x_a_px, x_b_px, y_a_px, y_b_px, height_m). Pixel spans are half-open
        [x_a, x_b) exactly as the height-field writes them.
    floor_z_m: local z of the ground the bar stands on (the tile's platform_height).
    x_max_m: retained for caller compatibility; roots are deliberately not
        clamped so they remain exactly symmetric about the hurdle centre.
    box_height_m: apex height above the floor; defaults to 0.30 m.

    Returns (n_slots, 2, 3) float32 in LOCAL metres: [[x_lo,y_lo,z_lo],[x_hi,y_hi,z_hi]].
    """
    out = np.zeros((max(1, len(footprints_px)), 2, 3), dtype=np.float32)
    for s, fp in enumerate(footprints_px):
        if fp is None:
            continue
        x_a_px, x_b_px, y_a_px, y_b_px, _h_m = fp
        # A zero-height slot would otherwise emit a full-height keep-out prism over
        # flat ground; the all-zero sentinel is skipped by `real_slot`.
        if float(_h_m) <= 1e-6:
            continue
        x_center = 0.5 * (float(x_a_px) + float(x_b_px)) * hs
        half_span = float(parabola_half_span_m)
        height = 0.30 if box_height_m is None else float(box_height_m)
        if half_span <= 0.0 or height <= 0.0:
            raise ValueError("parabolic hurdle clearance span/height must be positive")
        x_lo = x_center - half_span
        x_hi = x_center + half_span
        out[s, 0] = [x_lo,
                     float(y_a_px) * hs - pad_y,
                     float(floor_z_m)]
        z_hi = float(floor_z_m) + height
        out[s, 1] = [x_hi,
                     float(y_b_px) * hs + pad_y,
                     z_hi]
    return out


def flyball_wall_terrain(
    terrain,
    difficulty: float,
    platform_len: float = 2.5,
    platform_height: float = 0.0,
    wall_height: float = 1.0,
    wall_width: float = 1.5,
    # When set, the plate has a FIXED along-slope length of `plate_length` metres
    # and wall_height = plate_length * sin(theta). None = legacy fixed wall_height.
    plate_length: float = None,
    wall_thickness: float = 0.35,
    wall_angle_deg_range: tuple = (50.0, 90.0),
    # Per-env jitter (deg) on the level-lerped wall angle, so same-level tiles do
    # not share one angle. Foothold goals follow via np.interp(theta_deg).
    wall_angle_jitter_deg: float = 0.0,
    runup_dist_range: tuple = (4.0, 5.5),
    runup_jitter: float = 0.5,
    front_goal_offset: float = 1.0,
    lateral_offset: float = 0.0,
    landing_extra_lateral_offset: float = -0.10,  # g3 sits (lateral_offset + this) off centre. -0.10 shifts landing 15 cm toward centre vs the prior +0.05 default.
    landing_back_extra: float = 1.5,              # extra distance (m) to push g3 further back from the wall (subtracts from g_land_x).
    landing_centerline: bool = False,             # when True, override g_land_y to the corridor centerline (mid_y), ignoring lateral offsets.
    return_offset: float = 0.7,
    wall_goal_height: float = 0.26,
    pit_width_range: tuple = (0.10, 0.30),
    pit_height_offset_range: tuple = (0.05, 0.15),
    pit_depth: float = 2.0,
    roughness_height: tuple = (0.02, 0.06),
    roughness_downsampled_scale: float = 0.075,
    roughness_zero_prob: float = 0.0,
    # Contact regions (x offsets relative to wall_front_x; y relative to corridor centre)
    liftoff_x_range: tuple = (-0.9500, -0.7500),
    liftoff_y_mid: float   = -0.5600,
    liftoff_y_half: float  =  0.5300,
    liftoff_z_above: float =  0.0500,
    wall_y_half: float     =  0.8200,
    wall_z_min_above: float =  0.1600,
    wall_x_depth_scale: float = 2.0,
    landing_x_range: tuple = (-0.9600, -0.4700),
    landing_y_mid: float   =  0.6500,
    landing_y_half: float  =  0.6300,
    landing_z_above: float =  0.0500,
    # Front-foot LANDING footholds (P4L): FL/FR straddle (wall_front_x - standoff,
    # centre + y_mid) by +/- stance_half (FR at higher y) and +/- x_stagger.
    landing_foot_standoff: float    = 0.35,
    landing_foot_y_mid: float       = 0.15,
    landing_foot_stance_half: float = 0.10,
    landing_foot_x_stagger: float   = 0.10,
    air_x_range: tuple     = (-0.8500, -0.1000),
    air_y_mid: float       =  0.6500,
    air_y_half: float      =  1.0000,
    air_z_min_above: float =  0.1000,
    air_z_max_above: float =  0.8000,
    # Raise a thin obstacle at (platform_len + hurdle_offset_from_platform) so the
    # policy must clear it before reaching the wall takeoff.
    add_hurdle: bool                   = False,
    hurdle_height_range: tuple         = (0.05, 0.20),
    # Curriculum mode: first rows hurdle-free, height range interpolated over the rest.
    curriculum_level: int              = None,
    curriculum_max_level: int          = None,
    hurdle_zero_below_level: int       = 0,
    # PER-HURDLE jitter (m) on the level-lerped height, sampled independently.
    hurdle_height_jitter_m: float      = 0.0,
    # Bar/difficulty decorrelation: per slot, presence with probability
    # (1 - hurdle_zero_prob) and height uniform over [h_easy, h_hard], so bar
    # height is not a difficulty cue and a bar is either absent or full-size,
    # never a sliver below the deployment minimum. Level-lerp and jitter bypassed.
    hurdle_height_decorrelate: bool    = False,
    hurdle_zero_prob: float            = 0.0,
    hurdle_thickness: float            = 0.18,
    # Optional easy->hard thickness curriculum. None = fixed legacy thickness.
    hurdle_thickness_range: tuple      = None,
    hurdle_thickness_range_uniform: bool = False,
    hurdle_thickness_jitter_m: float   = 0.0,
    # Hard ceilings applied AFTER the difficulty lerp and the jitter, so jitter is
    # preserved but can never push a hurdle past the size the task allows.
    hurdle_height_max_m: float         = None,
    hurdle_thickness_max_m: float      = None,
    hurdle_width: float                = 2.0,
    # Lateral-width randomisation, uniform per tile and deliberately NOT tied to
    # difficulty (a narrow hurdle just moves the clearable span).
    hurdle_width_range: tuple          = None,
    # Hurdle-top roughness as (floor, ceiling) amplitude in metres, lerped with
    # difficulty: level 0 stays flat-topped.
    hurdle_roughness_height: tuple     = (0.0, 0.0),
    hurdle_offset_from_platform: float = 0.8,
    # Jitter (m) on the hurdle's x-centre; g0 tracks it via its hx_b anchor.
    hurdle_offset_jitter: float        = 0.3,
    # Fixed x-distance from the (last) hurdle's back face to g0, decoupling the
    # post-hurdle target from wall geometry so the command vector is stable.
    post_hurdle_goal_distance: float   = 0.6,
    # Second obstacle `second_hurdle_offset` metres past hurdle 1's rear
    # face, so the policy must clear two hurdles before the wall takeoff.
    add_second_hurdle: bool            = False,
    second_hurdle_offset: float        = 2.0,
    second_hurdle_offset_jitter: float = 0.3,
    # Compact-course placement: spawn->hurdle-front and hurdle-rear->wall-toe are
    # sampled directly, so both gaps keep independent jitter.
    spawn_x: float                         = 1.0,
    hurdle_front_from_spawn: float         = None,
    hurdle_front_from_spawn_jitter: float  = 0.0,
    wall_to_hurdle_rear: float             = None,
    wall_to_hurdle_rear_jitter: float      = 0.0,
    # Hard floor on the jittered last-hurdle-rear -> wall-toe gap [m]: that gap is
    # the whole run-up, so a low draw leaves no room to reach takeoff speed.
    wall_to_hurdle_rear_min: float         = None,
    # Scale hurdle jitters by difficulty (wall-angle jitter stays unscaled).
    hurdle_jitter_scale_with_difficulty: bool = False,
    # Per-hurdle AABBs. Slot 0 = forward (spawn → wall), slot 1 = backward.
    hurdle_liftoff_back: float    = 0.60,
    hurdle_liftoff_gap: float     = 0.20,
    hurdle_landing_forward: float = 0.60,
    hurdle_landing_gap: float     = 0.0,
    hurdle_aabb_y_half: float     = 0.50,
    hurdle_aabb_z_above: float    = 0.10,
    hurdle_air_z_above: float     = 0.30,
    # Parabolic clearance volume: pad_y expands the hurdle width laterally;
    # pad_x/pad_z are saved-config compatibility only.
    hurdle_clear_pad_x: float     = 0.05,
    hurdle_clear_pad_y: float     = 0.05,
    hurdle_clear_pad_z: float     = 0.05,
    hurdle_clear_box_height: float = None,
    hurdle_clear_half_span: float = 0.15,
    # Hurdle-only variant: omit the plate / backing wall and expose only the two
    # forward hurdle waypoints, so `terrain.goals` has exactly two entries.
    build_wall: bool                  = True,
    end_after_hurdles: bool           = False,
    # The base only counts as "in air" once it clears the hurdle by this margin.
    hurdle_air_z_min_above: float = 0.10,
    # Symmetric x-extension of the AIRBORNE AABB; 0 = matches hurdle thickness.
    hurdle_air_x_extra: float     = 0.0,
    # Extra x-extension toward each slot's landing direction (0 → wall, 1 → spawn).
    hurdle_air_x_extra_landing: float = 0.15,
    # Path to foothold_table.npz; None (or theta out of range) = analytic patch.
    foothold_table_path: str = None,
    # Constant z-lift (metres) for every per-foot target: the P11-27 table was
    # recorded with delta_z ≈ 0.176 m and its s_frac is relative to the wall base,
    # so a flat-base wall needs this to restore the recorded absolute z.
    foothold_z_offset_m: float = 0.0,
    # Kept before newly added options so positional callers keep their mapping.
    backing_wall_height: float = 2.0,
    backing_wall_thickness: float = 0.35,
    # Rear-pair world-y adjustment applied when theta exceeds the threshold below.
    rear_foothold_positive_y_shift_m: float = 0.0,
    rear_foothold_shift_min_angle_deg: float = 50.0,
    # Per-tile randomisation of the RETURN-leg goal x positions (indices >= 4): one
    # u per tile lerps each between g3's x and its nominal position; all share u,
    # so their ordering along -x is preserved.
    randomize_return_goal_x: bool = False,
    # Compact 4-waypoint layout (go2_teacher_swimmer), documented in the docstring.
    # Consumers of the wall / return goal indices must be told about the shift via
    # cfg.terrain.flyball_wall_goal_idx (=1) / flyball_return_goal_idx (=2).
    swimmer_goal_layout: bool = False,
):
    """
    Wall-turn terrain (no pit): robot sprints to an inclined plate, plants feet,
    pushes off, and returns. The plate leans against a taller backing wall that
    spans the full terrain-column width.

    Curriculum variables (all scale 0→1 with difficulty):
      - Alpha  (wall_angle_deg_range): wall incline angle, easy=50° → hard=90°
      - Delta_z (pit_height_offset_range): wall base height above approach floor (0.05 → 0.15 m)

    Goals (stored as (8, 3) in local metres — z is explicit for all goals):
      0) midpoint on the diagonal between spawn and wp_takeoff
      1) wp_takeoff: front_goal_offset m in front of wall, lateral_offset m off centre
      2) wp_contact: on wall surface at delta_z + wall_goal_height — z set analytically from wall angle
      3) wp_land: front_goal_offset m in front of wall, opposite lateral side
      4) wp_return: return_offset m behind start, centered
      5-7) padded with wp_return

    With ``swimmer_goal_layout`` (two-hurdle course only) the takeoff and landing
    waypoints are dropped and the circuit becomes:
      0) midpoint of the gap between the two hurdles
      1) wp_contact (the wall goal, formerly index 2)
      2) the same between-hurdles midpoint, on the way back
      3) final return goal; 4-7 padded with it
    """
    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale

    def m2px(m): return int(round(m / hs))
    def clamp(v, lo, hi): return max(lo, min(hi, v))
    def lerp(a, b, t): return a + (b - a) * t

    d = clamp(float(difficulty), 0.0, 1.0)
    hurdle_jitter_scale = d if hurdle_jitter_scale_with_difficulty else 1.0

    delta_x = lerp(pit_width_range[0], pit_width_range[1], d)   # kept for goal offset (same distance as when pit existed)
    delta_z = lerp(pit_height_offset_range[0], pit_height_offset_range[1], d)

    # Sampled before wall placement: the wall toe is anchored to the last bar's rear.
    def _sample_hurdle_thickness_m():
        if hurdle_thickness_range is None:
            th = float(hurdle_thickness)
        elif hurdle_thickness_range_uniform:
            th = float(np.random.uniform(
                float(hurdle_thickness_range[0]),
                float(hurdle_thickness_range[1])))
        else:
            th = lerp(
                float(hurdle_thickness_range[0]),
                float(hurdle_thickness_range[1]), d)
        if hurdle_thickness_jitter_m > 0.0:
            th += float(np.random.uniform(
                -hurdle_thickness_jitter_m,
                hurdle_thickness_jitter_m)) * hurdle_jitter_scale
        th = max(hs, th)
        if hurdle_thickness_max_m is not None:
            th = min(th, float(hurdle_thickness_max_m))
        return th

    hurdle_thickness_m = _sample_hurdle_thickness_m()
    hurdle2_thickness_m = (
        _sample_hurdle_thickness_m()
        if add_hurdle and add_second_hurdle
        else hurdle_thickness_m
    )

    base_h = int(round(platform_height / vs))
    terrain.height_field_raw[:] = base_h
    mid_y = terrain.length // 2
    p_len = clamp(m2px(platform_len), 1, terrain.width)

    runup_m = lerp(runup_dist_range[0], runup_dist_range[1], d)
    runup_m = max(0.4, runup_m + np.random.uniform(-runup_jitter, runup_jitter) * d)
    runup_px = max(1, m2px(runup_m))

    theta_deg = lerp(wall_angle_deg_range[0], wall_angle_deg_range[1], d)
    if wall_angle_jitter_deg > 0.0:
        theta_deg = theta_deg + float(np.random.uniform(-wall_angle_jitter_deg, wall_angle_jitter_deg))
    theta = np.deg2rad(theta_deg)
    tan_t = max(np.tan(theta), 1e-6)

    # Derive the vertical extent from the (jittered) angle so the plate is always
    # exactly `plate_length` m long; everything below reads this wall_height.
    if plate_length is not None:
        wall_height = float(plate_length) * np.sin(theta)

    wall_bottom_len_m = min(wall_height / tan_t, 6.0)
    dx_px = max(2, m2px(wall_bottom_len_m))
    depth_px = max(2, m2px(wall_thickness))
    thk_px = max(1, m2px(hurdle_thickness_m))
    thk2_px = max(1, m2px(hurdle2_thickness_m))

    plate_half_w = max(1, m2px(wall_width / 2.0))
    backing_depth_px = max(2, m2px(backing_wall_thickness))
    compact_layout = (
        add_hurdle
        and hurdle_front_from_spawn is not None
        and wall_to_hurdle_rear is not None
    )
    compact_hx_a = compact_hx_b = None
    compact_hx2_a = compact_hx2_b = None
    if compact_layout:
        front_dist_m = float(hurdle_front_from_spawn)
        if hurdle_front_from_spawn_jitter > 0.0:
            front_dist_m += float(np.random.uniform(
                -hurdle_front_from_spawn_jitter,
                hurdle_front_from_spawn_jitter)) * hurdle_jitter_scale
        compact_hx_a = clamp(m2px(float(spawn_x) + front_dist_m), 0, terrain.width - 1)
        compact_hx_b = clamp(compact_hx_a + thk_px, compact_hx_a + 1, terrain.width)

        last_hurdle_back_px = compact_hx_b
        if add_second_hurdle:
            gap_m = float(second_hurdle_offset)
            if second_hurdle_offset_jitter > 0.0:
                gap_m += float(np.random.uniform(
                    -second_hurdle_offset_jitter,
                    second_hurdle_offset_jitter)) * hurdle_jitter_scale
            gap_px = max(1, m2px(gap_m))
            compact_hx2_a = clamp(compact_hx_b + gap_px, compact_hx_b + 1, terrain.width - 1)
            compact_hx2_b = clamp(compact_hx2_a + thk2_px, compact_hx2_a + 1, terrain.width)
            last_hurdle_back_px = compact_hx2_b

        rear_gap_m = float(wall_to_hurdle_rear)
        if wall_to_hurdle_rear_jitter > 0.0:
            rear_gap_m += float(np.random.uniform(
                -wall_to_hurdle_rear_jitter,
                wall_to_hurdle_rear_jitter)) * hurdle_jitter_scale
        # Floor the JITTERED value, not the nominal: with base 2.0 and jitter
        # +-0.3 the raw draw reaches 1.70 m, not enough run-up to reach plant speed.
        if wall_to_hurdle_rear_min is not None:
            rear_gap_m = max(rear_gap_m, float(wall_to_hurdle_rear_min))
        rear_gap_px = max(1, m2px(rear_gap_m))
        desired_wall_front_px = last_hurdle_back_px + rear_gap_px
        desired_wall_x0 = desired_wall_front_px + (depth_px - 1)
        max_wall_x0 = terrain.width - dx_px - backing_depth_px
        if desired_wall_x0 > max_wall_x0:
            raise ValueError(
                "compact flyball layout does not fit terrain tile: "
                f"wall_x0={desired_wall_x0}, max={max_wall_x0}, "
                f"tile_length={terrain.width * hs:.3f} m")
        wall_x0 = clamp(desired_wall_x0, 0, max_wall_x0)
        wall_front_px = desired_wall_front_px
        runup_m = (wall_x0 - p_len) * hs
    else:
        wall_x0 = clamp(
            p_len + runup_px, 0,
            terrain.width - dx_px - backing_depth_px,
        )
        wall_front_px = wall_x0 - (depth_px - 1)

    # Roughness is applied globally, then ONLY the plate footprint is flattened —
    # required, not cosmetic: the incline is laid down via a height-field max, so a
    # bump above the ramp's low end would survive it and pit the contact surface.
    # Amplitude lerps roughness_height[0] (lvl0) -> roughness_height[1] (top), and
    # a per-tile draw leaves roughness_zero_prob of tiles completely smooth.
    roughness_zero_prob = min(max(float(roughness_zero_prob), 0.0), 1.0)
    if random.random() < roughness_zero_prob:
        rough_h = 0.0
    else:
        rough_h = (roughness_height[1] - roughness_height[0]) * d + roughness_height[0]
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=-rough_h, max_height=rough_h,
        step=0.005, downsampled_scale=roughness_downsampled_scale,
    )
    # The backing wall is raised via np.maximum, so it needs no pre-clearing.
    _plate_y_lo = clamp(mid_y - plate_half_w, 0, terrain.length)
    _plate_y_hi = clamp(mid_y + plate_half_w + 1, 0, terrain.length)
    _plate_x_hi = min(terrain.width, wall_x0 + dx_px)
    if build_wall:
        terrain.height_field_raw[wall_front_px:_plate_x_hi, _plate_y_lo:_plate_y_hi] = base_h

    wall_base_h = base_h + int(round(delta_z / vs))
    wall_top_h = wall_base_h + int(round(wall_height / vs))
    ramp = np.linspace(float(wall_base_h), float(wall_top_h), dx_px, endpoint=True)
    XI, YI = np.meshgrid(np.arange(terrain.width), np.arange(terrain.length), indexing='ij')
    if build_wall:
        for depth in range(depth_px):
            u = XI - wall_x0 + depth
            in_wall = (u >= 0) & (u < dx_px) & (np.abs(YI - mid_y) <= plate_half_w)
            u_c = np.clip(u, 0, dx_px - 1 - 1e-9).astype(np.float64)
            u_lo = np.floor(u_c).astype(np.int32)
            u_hi = np.minimum(u_lo + 1, dx_px - 1)
            h_interp = ramp[u_lo] * (1.0 - (u_c - u_lo)) + ramp[u_hi] * (u_c - u_lo)
            h_int = np.round(h_interp).astype(np.int16)
            mask = in_wall & (h_int > terrain.height_field_raw)
            terrain.height_field_raw[mask] = h_int[mask]

    # The backing block's front face is flush with the ramp's TOP edge so no flat
    # top strip remains: the thickness extrusion otherwise clips the ramp top over
    # the last depth_px columns, and starting there overwrites that plateau. The
    # wall sits behind the analytical contact surface, so goals/AABBs are unchanged.
    backing_front_px = wall_x0 + dx_px - depth_px
    backing_back_px = min(terrain.width, backing_front_px + backing_depth_px)
    backing_height_m = max(float(backing_wall_height), delta_z + wall_height)
    backing_top_h = base_h + int(round(backing_height_m / vs))
    if build_wall:
        backing = terrain.height_field_raw[backing_front_px:backing_back_px, :]
        np.maximum(backing, backing_top_h, out=backing)

    # Hurdles are raised after the wall so they don't interact with the wall area.
    hurdle_x_lo = -1.0
    hurdle_x_hi = -1.0
    hurdle2_x_lo = -1.0
    hurdle2_x_hi = -1.0
    last_hx_b = None  # back-face px of the last (rightmost) hurdle, for g0 anchoring
    # Realised footprint per bar, same slot order as flyball_hurdle_x_ranges_world:
    # (x_a_px, x_b_px, y_a_px, y_b_px, height_m), or None. Only bars above
    # _hurdle_min_h are filled, so the clearance volume self-disables at level 0.
    hurdle_footprints_px = [None, None]
    if add_hurdle:
        h_easy, h_hard = hurdle_height_range
        zero_height_row = (
            curriculum_level is not None
            and curriculum_level < int(hurdle_zero_below_level))
        if zero_height_row:
            h_m = 0.0
        elif curriculum_level is not None:
            # Rebase so the first nonzero row gets the configured minimum.
            lo = int(hurdle_zero_below_level)
            hi = int(curriculum_max_level if curriculum_max_level is not None
                     else lo)
            active_t = (curriculum_level - lo) / max(hi - lo, 1)
            h_m = lerp(h_easy, h_hard, clamp(active_t, 0.0, 1.0))
        else:
            h_m = lerp(h_easy, h_hard, d)   # randomized-terrain fallback
        # Each hurdle samples its own height around the level-lerped base.
        def _jittered_h_m():
            if zero_height_row:
                return 0.0
            if hurdle_height_decorrelate:
                if float(np.random.uniform()) < float(hurdle_zero_prob):
                    return 0.0
                return float(np.random.uniform(h_easy, h_hard))
            hh = h_m
            if hurdle_height_jitter_m > 0.0:
                hh = h_m + float(np.random.uniform(
                    -hurdle_height_jitter_m,
                    hurdle_height_jitter_m)) * hurdle_jitter_scale
            if hurdle_height_max_m is not None:
                hh = min(hh, float(hurdle_height_max_m))
            # h_easy is a hard floor once hurdles are enabled: jitter may vary
            # harder rows upward but cannot go below the requested minimum.
            return max(float(h_easy), hh)
        # A hurdle shorter than this is treated as flat ground: no x-range slot is
        # recorded, so _reward_hurdle_contact never penalises a foot there.
        _hurdle_min_h = 0.01
        def _sample_hurdle_y_span():
            if hurdle_width_range is None:
                width_m = float(hurdle_width)
            else:
                width_m = float(np.random.uniform(
                    float(hurdle_width_range[0]), float(hurdle_width_range[1])))
            half_w = max(1, m2px(width_m / 2.0))
            return (
                width_m,
                clamp(mid_y - half_w, 0, terrain.length - 1),
                clamp(mid_y + half_w, 0, terrain.length),
            )

        hurdle_width_m, hy_a, hy_b = _sample_hurdle_y_span()
        hurdle2_width_m, hy2_a, hy2_b = (
            _sample_hurdle_y_span()
            if add_second_hurdle
            else (hurdle_width_m, hy_a, hy_b)
        )
        # lerp(floor, ceiling, difficulty), then a per-tile draw in [floor, amp_max].
        _hr_lo, _hr_hi = float(hurdle_roughness_height[0]), float(hurdle_roughness_height[1])
        _hr_amp_max = lerp(_hr_lo, _hr_hi, d)
        _hr_amp = float(np.random.uniform(_hr_lo, _hr_amp_max)) if _hr_amp_max > 0.0 else 0.0
        _hr_cell_px = max(1, m2px(roughness_downsampled_scale))

        def _roughen_top(x_a, x_b, y_a, y_b):
            """Add coarse-cell noise (+-_hr_amp) to an already-flat hurdle top.

            Noise is drawn on `roughness_downsampled_scale` cells and repeated
            onto the 1 cm height field so the bumps match the scale of the
            approach roughness instead of being per-cell white noise.
            """
            # Amplitude quantises to vertical_scale (5 mm), so with a 1 cm ceiling
            # the reachable steps are 0 / +-5 mm / +-10 mm. Round without a floor
            # so sub-half-cell amplitudes stay flat (preserves low-level scaling).
            amp_i = int(round(_hr_amp / vs))
            if amp_i < 1 or x_b <= x_a or y_b <= y_a:
                return
            nx_c = int(np.ceil((x_b - x_a) / _hr_cell_px))
            ny_c = int(np.ceil((y_b - y_a) / _hr_cell_px))
            coarse = np.random.randint(-amp_i, amp_i + 1, size=(nx_c, ny_c))
            noise = np.kron(coarse, np.ones((_hr_cell_px, _hr_cell_px), dtype=int))
            terrain.height_field_raw[x_a:x_b, y_a:y_b] += noise[:x_b - x_a, :y_b - y_a]

        if compact_layout:
            hx_a, hx_b = compact_hx_a, compact_hx_b
            hx_center = (hx_a + hx_b) // 2
        else:
            offset_jitter = np.random.uniform(-hurdle_offset_jitter, hurdle_offset_jitter)
            offset_m = hurdle_offset_from_platform + offset_jitter * hurdle_jitter_scale
            offset_m = max(hurdle_thickness_m, offset_m)  # never collide with the spawn platform edge
            hx_center = p_len + m2px(offset_m)
            hx_a = clamp(hx_center - thk_px // 2, 0, terrain.width - 1)
            hx_b = clamp(hx_center + (thk_px - thk_px // 2), 1, terrain.width)
        h1_m = _jittered_h_m()
        terrain.height_field_raw[hx_a:hx_b, hy_a:hy_b] = base_h + int(round(h1_m / vs))
        if h1_m > _hurdle_min_h:
            _roughen_top(hx_a, hx_b, hy_a, hy_b)
        # Only arm the contact penalty for a hurdle that rises above the floor.
        if h1_m > _hurdle_min_h:
            hurdle_x_lo = float(hx_a) * hs
            hurdle_x_hi = float(hx_b) * hs
            hurdle_footprints_px[0] = (hx_a, hx_b, hy_a, hy_b, h1_m)
        last_hx_b = hx_b

        # Hurdle 2: own jitter; height, thickness and width sampled independently.
        if add_second_hurdle:
            if compact_layout:
                hx2_a, hx2_b = compact_hx2_a, compact_hx2_b
            else:
                gap_jitter = np.random.uniform(-second_hurdle_offset_jitter, second_hurdle_offset_jitter)
                gap_m = second_hurdle_offset + gap_jitter * hurdle_jitter_scale
                gap_px = max(1, m2px(gap_m))
                hx2_a = clamp(hx_b + gap_px, hx_b + 1, terrain.width - 1)
                hx2_b = clamp(hx2_a + thk2_px, hx2_a + 1, terrain.width)
            h2_m = _jittered_h_m()
            terrain.height_field_raw[hx2_a:hx2_b, hy2_a:hy2_b] = base_h + int(round(h2_m / vs))
            if h2_m > _hurdle_min_h:
                _roughen_top(hx2_a, hx2_b, hy2_a, hy2_b)
                hurdle2_x_lo = float(hx2_a) * hs
                hurdle2_x_hi = float(hx2_b) * hs
                hurdle_footprints_px[1] = (hx2_a, hx2_b, hy2_a, hy2_b, h2_m)
            last_hx_b = hx2_b

    if end_after_hurdles and not (add_hurdle and add_second_hurdle):
        raise ValueError("end_after_hurdles requires two enabled hurdles")
    if swimmer_goal_layout and (end_after_hurdles or not (add_hurdle and add_second_hurdle)):
        raise ValueError("swimmer_goal_layout requires the two-hurdle wall course")
    goals = np.zeros((2 if end_after_hurdles else 8, 3), dtype=np.float32)

    g_start_x = (clamp(p_len - m2px(0.5), 0, terrain.width - 1)) * hs
    g_start_y = mid_y * hs

    pit_width_px = max(1, m2px(delta_x))
    g_takeoff_x = clamp(wall_front_px - pit_width_px - m2px(front_goal_offset), 0, terrain.width - 1) * hs
    g_takeoff_y = clamp(mid_y - m2px(lateral_offset), 0, terrain.length - 1) * hs

    # g_contact: analytically on the wall's front surface at delta_z + wall_goal_height.
    wall_front_x = wall_front_px * hs
    contact_x = clamp(wall_front_x + wall_goal_height / tan_t, 0, (terrain.width - 1) * hs)
    contact_y = mid_y * hs

    # g_land: landing_back_extra farther from the wall than g_takeoff, pinned to the
    # centerline (the wall-turn diagonal lands back on centre).
    g_land_x = clamp(wall_front_px - pit_width_px - m2px(front_goal_offset + landing_back_extra), 0, terrain.width - 1) * hs
    g_land_y = mid_y * hs

    g_return_x = clamp(g_start_x - return_offset, 0.0, (terrain.width - 1) * hs)

    if compact_layout:
        # Extreme negative rear-gap jitter can put the takeoff point just behind the
        # hurdle; keep the circuit monotonic without overshooting the final goal.
        hurdle_rear_x = float(compact_hx_b) * hs
        g_takeoff_x = max(g_takeoff_x, hurdle_rear_x)
        g_land_x = max(g_land_x, g_return_x)

    if end_after_hurdles:
        # Reaching g1 advances cur_goal_idx to num_goals -> reach-goal termination.
        h1_back_x = float(hx_b) * hs
        h2_front_x = float(hx2_a) * hs
        between_x = 0.5 * (h1_back_x + h2_front_x)
        finish_x = clamp(
            float(hx2_b) * hs + post_hurdle_goal_distance,
            0.0, (terrain.width - 1) * hs)
        goals[0] = [between_x, mid_y * hs, platform_height]
        goals[1] = [finish_x,  mid_y * hs, platform_height]
    elif add_hurdle and add_second_hurdle:
        # Two-hurdle layout: g0 between-hurdles forward, g1 takeoff, g2 contact,
        # g3 landing, g4 between-hurdles backward, g5 return, g6/g7 padded to g5.
        # Goal-index gates (takeoff=1, contact=2, landing=3) match the legacy
        # layout, so cfg.terrain.flyball_*_goal_idx defaults apply unchanged.
        h1_back_x = float(hx_b) * hs           # spawn-side hurdle back face (forward direction)
        h2_front_x = float(hx2_a) * hs         # wall-side hurdle front face
        between_x = 0.5 * (h1_back_x + h2_front_x)
        between_y = mid_y * hs

        # Return path: g4 sits 50 cm on the spawn side of hurdle 1; g5+ are
        # farther toward spawn by a per-tile distance sampled around 1.7 m.
        g4_x = clamp(float(hx_a) * hs - 0.50, 0.0, (terrain.width - 1) * hs)
        g5_gap_m = 1.70 + float(np.random.uniform(-0.30, 0.30)) * hurdle_jitter_scale
        g5_x = clamp(g4_x - g5_gap_m, 0.0, (terrain.width - 1) * hs)

        if swimmer_goal_layout:
            # Both between-hurdles goals sit on the exact midpoint of the gap.
            goals[0] = [between_x, between_y, platform_height]
            goals[1] = [contact_x, contact_y, delta_z + wall_goal_height]
            goals[2] = [between_x, between_y, platform_height]
            goals[3] = [g5_x,      g_start_y, platform_height]
            goals[4] = goals[3]
            goals[5] = goals[3]
            goals[6] = goals[3]
            goals[7] = goals[3]
        else:
            goals[0] = [between_x,   between_y,   platform_height]
            goals[1] = [g_takeoff_x, g_takeoff_y, platform_height]
            goals[2] = [contact_x,   contact_y,   delta_z + wall_goal_height]
            goals[3] = [g_land_x,    g_land_y,    platform_height]

            goals[4] = [g4_x,       between_y,   platform_height]
            goals[5] = [g5_x,       g_start_y,   platform_height]
            goals[6] = goals[5]
            goals[7] = goals[5]
    else:
        # Legacy layout: g0 post-hurdle / midpoint(spawn, takeoff), g1 takeoff,
        # g2 contact, g3 landing, g4 return, g5..g7 padded.
        if add_hurdle:
            g0_x = clamp(float(last_hx_b) * hs + post_hurdle_goal_distance, 0.0, (terrain.width - 1) * hs)
        else:
            g0_x = 0.5 * (g_start_x + g_takeoff_x)
        g0_y = mid_y * hs

        goals[0] = [g0_x,        g0_y,        platform_height]
        goals[1] = [g_takeoff_x, g_takeoff_y, platform_height]
        goals[2] = [contact_x,   contact_y,   delta_z + wall_goal_height]
        goals[3] = [g_land_x,    g_land_y,    platform_height]
        goals[4] = [g_return_x,  g_start_y,   platform_height]
        goals[5] = goals[4]
        goals[6] = goals[4]
        goals[7] = goals[4]

    if not (add_hurdle and add_second_hurdle):
        # Push the final return goals (index >= 4) 2 m further in -x, past the start
        # platform, before the circuit ends.
        final_goal_shift_x = -2.0
        for gi in range(4, 8):
            goals[gi] = [
                clamp(goals[gi][0] + final_goal_shift_x, 0.0, (terrain.width - 1) * hs),
                goals[gi][1],
                goals[gi][2],
            ]

        # One per-tile u slides every return goal between g3's x and its nominal x.
        if randomize_return_goal_x:
            u = float(np.random.uniform(0.0, 1.0))
            g3_x = float(goals[3][0])
            for gi in range(4, 8):
                goals[gi] = [
                    clamp(lerp(g3_x, float(goals[gi][0]), u), 0.0, (terrain.width - 1) * hs),
                    goals[gi][1],
                    goals[gi][2],
                ]

    terrain.goals = goals

    # pit_x0/pit_x1 used by legged_robot.py for on-wall detection (feet >= pit_x1).
    # No pit exists; both set to wall_front so on-wall detection is unchanged.
    terrain.flyball_pit_x0_world = float(wall_front_px * hs)
    terrain.flyball_pit_x1_world = float(wall_front_px * hs)

    # Forbidden zone: same x-extent as original pit, enforced via contact penalty
    terrain.flyball_forbidden_x0_world = float((wall_front_px - pit_width_px) * hs)
    terrain.flyball_forbidden_x1_world = float(wall_front_px * hs)

    # Shape (2, 2) to match the global buffer `flyball_hurdle_x_ranges`. Slot 1 is
    # hurdle 2 when add_second_hurdle, else sentinel (-1, -1) so the in-range test
    # never fires there.
    terrain.flyball_hurdle_x_ranges_world = np.array(
        [[hurdle_x_lo,  hurdle_x_hi],
         [hurdle2_x_lo, hurdle2_x_hi]], dtype=np.float32
    )

    terrain.wall_theta_deg = float(theta_deg)
    terrain.flyball_wall_angle_deg = float(theta_deg)
    terrain.runup_m = float(runup_m)
    terrain.wall_goal_height = float(wall_goal_height)
    terrain.wall_delta_z = float(delta_z)
    terrain.wall_delta_x = float(delta_x)
    # Exact realised geometry (after heightfield quantisation), for debug tooling.
    terrain.hurdle_thickness_m = float(thk_px * hs)
    terrain.hurdle2_thickness_m = float(thk2_px * hs) if (add_hurdle and add_second_hurdle) else 0.0
    terrain.hurdle_height_m = float(h1_m) if add_hurdle else 0.0
    terrain.hurdle2_height_m = float(h2_m) if (add_hurdle and add_second_hurdle) else 0.0
    terrain.hurdle_width_m = float((hy_b - hy_a) * hs) if add_hurdle else 0.0
    terrain.hurdle2_width_m = float((hy2_b - hy2_a) * hs) if (add_hurdle and add_second_hurdle) else 0.0
    terrain.hurdle_front_from_spawn_m = (
        float(hx_a * hs - spawn_x) if add_hurdle else 0.0)
    terrain.wall_to_hurdle_rear_m = (
        float(wall_front_px * hs - hx_b * hs) if add_hurdle else 0.0)
    terrain.wall_toe_x_m = float(wall_front_px * hs)
    terrain.wall_back_x_m = float(backing_back_px * hs)

    # Contact regions for the WoCoCo reward, each an AABB [[x,y,z]lo, [x,y,z]hi] in
    # local metres; world offsets are added by Terrain.add_terrain_to_map.
    _cy = mid_y * hs
    _x_max = (terrain.width - 1) * hs

    terrain.flyball_liftoff_region = np.array([
        [clamp(wall_front_x + liftoff_x_range[0], 0.0, _x_max),
         _cy + liftoff_y_mid - liftoff_y_half,
         platform_height - 0.1],
        [clamp(wall_front_x + liftoff_x_range[1], 0.0, _x_max),
         _cy + liftoff_y_mid + liftoff_y_half,
         platform_height + liftoff_z_above],
    ], dtype=np.float32)

    wall_region_z_min = delta_z + wall_z_min_above
    wall_region_x_min = clamp(wall_front_x + wall_z_min_above / tan_t, 0.0, _x_max)
    wall_top_x = clamp(wall_front_x + wall_height / tan_t, 0.0, _x_max)
    wall_region_x_max = clamp(
        wall_region_x_min + wall_x_depth_scale * (wall_top_x - wall_region_x_min), 0.0, _x_max)
    wall_region_y_half = min(float(wall_y_half), wall_width / 2.0)
    terrain.flyball_wall_region = np.array([
        [wall_region_x_min, _cy - wall_region_y_half, wall_region_z_min],
        [wall_region_x_max, _cy + wall_region_y_half, delta_z + wall_height],
    ], dtype=np.float32)

    terrain.flyball_landing_region = np.array([
        [clamp(wall_front_x + landing_x_range[0], 0.0, _x_max),
         _cy + landing_y_mid - landing_y_half,
         platform_height - 0.1],
        [clamp(wall_front_x + landing_x_range[1], 0.0, _x_max),
         _cy + landing_y_mid + landing_y_half,
         platform_height + landing_z_above],
    ], dtype=np.float32)

    # Floor points landing_foot_standoff metres out from the wall bottom lip, on the
    # +y landing side. Index order FL, FR matches feet_indices[:2]; FR gets the
    # HIGHER world y and sits landing_foot_x_stagger m closer to the wall (+x).
    # Drives the P4L stage-2 landing pull, independent of flyball_landing_region.
    _lf_x  = clamp(wall_front_x - landing_foot_standoff, 0.0, _x_max)
    _lf_cy = _cy + landing_foot_y_mid
    _lf_x_fl = clamp(_lf_x - landing_foot_x_stagger, 0.0, _x_max)
    _lf_x_fr = clamp(_lf_x + landing_foot_x_stagger, 0.0, _x_max)
    terrain.flyball_landing_foot_target = np.array([
        [_lf_x_fl, _lf_cy - landing_foot_stance_half, platform_height],   # FL (lower y, -x)
        [_lf_x_fr, _lf_cy + landing_foot_stance_half, platform_height],   # FR (higher y, +x)
    ], dtype=np.float32)

    terrain.flyball_air_region = np.array([
        [clamp(wall_front_x + air_x_range[0], 0.0, _x_max),
         _cy + air_y_mid - air_y_half,
         platform_height + air_z_min_above],
        [clamp(wall_front_x + air_x_range[1], 0.0, _x_max),
         _cy + air_y_mid + air_y_half,
         platform_height + air_z_max_above],
    ], dtype=np.float32)

    # Per-foot targets, order matching feet_indices: FL, FR, RL, RR. Wall-local
    # frame: s = along-slope distance up the face, u = lateral offset from _cy.
    # World point on the ramp face: x = wall_front_x + s*cos(theta),
    #                               z = platform_height + delta_z + s*sin(theta),
    #                               y = _cy + u.  L(theta) = wall_height/sin(theta).
    _st = max(np.sin(theta), 1e-6)
    _L_theta = wall_height / _st

    def _foot_world(s, u, dz=None):
        # foothold_z_offset_m raises the target's HEIGHT while keeping it ON the
        # inclined face: shift up-slope by dz/sin(theta), clamped to the plate top
        # (_L_theta) so a large offset can't push past the ridge.
        dz = foothold_z_offset_m if dz is None else dz
        s_eff = min(s + dz / _st, _L_theta)
        return [clamp(wall_front_x + s_eff * np.cos(theta), 0.0, _x_max),
                _cy + u,
                platform_height + delta_z + s_eff * _st]

    foot_target = np.zeros((4, 3), dtype=np.float32)
    _tbl = None
    if foothold_table_path is not None and os.path.exists(foothold_table_path):
        _tbl = _load_foothold_table(foothold_table_path)
    elif foothold_table_path is not None and foothold_table_path not in _FOOTHOLD_MISSING_WARNED:
        # Silent degradation to the analytic patch looks like a working run.
        _FOOTHOLD_MISSING_WARNED.add(foothold_table_path)
        print(f"[flyball] WARNING: foothold table NOT FOUND at {foothold_table_path}\n"
              f"[flyball]          falling back to the crude ANALYTIC foot targets")
    # Tables flagged final_on_face (the hand-tuned 30-80 deg table from
    # scripts/make_foothold_table_manual.py) already bake in the z-offset and the
    # steep-wall rear +y shift, so both runtime adjustments are skipped.
    _tbl_final = bool(np.asarray(_tbl['final_on_face']).item()) \
        if (_tbl is not None and 'final_on_face' in _tbl) else False
    if _tbl is not None:
        th = _tbl['theta_deg']                      # (T,)
        # np.interp clamps to the end values outside [th0, thT] — i.e. theta>50
        # uses the 50-deg row (no upward extrapolation), as the plan requires.
        for f in range(4):
            sfrac = float(np.interp(theta_deg, th, _tbl['s_frac_mean'][:, f]))
            uu    = float(np.interp(theta_deg, th, _tbl['u_mean'][:, f]))
            foot_target[f] = _foot_world(sfrac * _L_theta, uu,
                                         dz=0.0 if _tbl_final else None)
    else:
        # Analytic fallback: front pair low on the slope at +u, rear pair higher at -u.
        _analytic = [(0.10, +0.10), (0.20, +0.05), (0.13, -0.28), (0.30, -0.30)]
        for f, (sfrac, uu) in enumerate(_analytic):
            foot_target[f] = _foot_world(sfrac * _L_theta, uu)
    if theta_deg > rear_foothold_shift_min_angle_deg and not _tbl_final:
        foot_target[2:, 1] += rear_foothold_positive_y_shift_m
    terrain.flyball_foot_target = foot_target

    # Per-hurdle AABBs (slot 0 = forward, slot 1 = backward), always shape (2, 2, 3)
    # to match the global buffer; with no hurdle, all slots collapse to zero-area
    # sentinels (lo == hi) so the in-region tests never fire.
    hurdle_liftoff = np.zeros((2, 2, 3), dtype=np.float32)
    hurdle_air     = np.zeros((2, 2, 3), dtype=np.float32)
    hurdle_landing = np.zeros((2, 2, 3), dtype=np.float32)
    if add_hurdle:
        h_top_m = float(h1_m)
        h_front_m = float(hx_a) * hs   # spawn-side face of hurdle
        h_back_m  = float(hx_b) * hs   # wall-side face of hurdle
        y_min = clamp(_cy - hurdle_aabb_y_half, 0.0, (terrain.length - 1) * hs)
        y_max = clamp(_cy + hurdle_aabb_y_half, 0.0, (terrain.length - 1) * hs)
        hurdle_liftoff[0, 0] = [clamp(h_front_m - hurdle_liftoff_gap - hurdle_liftoff_back, 0.0, _x_max),
                                y_min, platform_height - 0.1]
        hurdle_liftoff[0, 1] = [clamp(h_front_m - hurdle_liftoff_gap, 0.0, _x_max),
                                y_max, platform_height + hurdle_aabb_z_above]
        hurdle_air[0, 0]     = [clamp(h_front_m - hurdle_air_x_extra, 0.0, _x_max),
                                y_min, platform_height + h_top_m + hurdle_air_z_min_above]
        hurdle_air[0, 1]     = [clamp(h_back_m  + hurdle_air_x_extra + hurdle_air_x_extra_landing, 0.0, _x_max),
                                y_max, platform_height + h_top_m + hurdle_air_z_min_above + hurdle_air_z_above]
        hurdle_landing[0, 0] = [clamp(h_back_m + hurdle_landing_gap, 0.0, _x_max),
                                y_min, platform_height - 0.1]
        hurdle_landing[0, 1] = [clamp(h_back_m + hurdle_landing_gap + hurdle_landing_forward, 0.0, _x_max),
                                y_max, platform_height + hurdle_aabb_z_above]
        # Slot 1 — backward crossing, mirrored: pre-AABB wall side, post-AABB spawn side.
        hurdle_liftoff[1, 0] = [clamp(h_back_m + hurdle_liftoff_gap, 0.0, _x_max),
                                y_min, platform_height - 0.1]
        hurdle_liftoff[1, 1] = [clamp(h_back_m + hurdle_liftoff_gap + hurdle_liftoff_back, 0.0, _x_max),
                                y_max, platform_height + hurdle_aabb_z_above]
        hurdle_air[1, 0]     = [clamp(h_front_m - hurdle_air_x_extra - hurdle_air_x_extra_landing, 0.0, _x_max),
                                y_min, platform_height + h_top_m + hurdle_air_z_min_above]
        hurdle_air[1, 1]     = [clamp(h_back_m  + hurdle_air_x_extra, 0.0, _x_max),
                                y_max, platform_height + h_top_m + hurdle_air_z_min_above + hurdle_air_z_above]
        hurdle_landing[1, 0] = [clamp(h_front_m - hurdle_landing_gap - hurdle_landing_forward, 0.0, _x_max),
                                y_min, platform_height - 0.1]
        hurdle_landing[1, 1] = [clamp(h_front_m - hurdle_landing_gap, 0.0, _x_max),
                                y_max, platform_height + hurdle_aabb_z_above]
    terrain.flyball_hurdle_liftoff_region = hurdle_liftoff
    terrain.flyball_hurdle_air_region     = hurdle_air
    terrain.flyball_hurdle_landing_region = hurdle_landing

    # Bounds store roots/padded width/floor/apex for the exact collision test.
    terrain.flyball_hurdle_clear_region = _hurdle_clear_boxes(
        hurdle_footprints_px, hs, platform_height,
        hurdle_clear_pad_x, hurdle_clear_pad_y, hurdle_clear_pad_z,
        x_max_m=_x_max,
        box_height_m=hurdle_clear_box_height,
        parabola_half_span_m=hurdle_clear_half_span)

    return terrain


def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles
