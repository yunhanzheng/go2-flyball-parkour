"""go2_teacher — privileged (scandot) wall-plant teacher config.

The single configuration used by the release. It is the flattened form of what
used to be a two-level chain: the former ``Go2TeacherCfg`` body with the
``Go2TeacherCfg`` overrides folded in, now inheriting ``Go2RoughCfg``
directly. Sites where the swimmer layer overrode a teacher value are marked
``[swimmer]``.

The three families the swimmer layer changed, for the record:

  1. rewards.wococo_wall_press_force_threshold_by_level = [10, 10, 35, 45, 50]
     — a per-level ramp replacing the flat 10 N wall-press gate (harder to
     register a wall plant on the upper curriculum rows).

  2. terrain foothold-GOAL LOCATIONS at the e68094d ``go2_cliffside`` values:
     the hand-authored 30-80 deg foothold table, the 5 cm on-face z-lift (vs
     0.15 for the default foothold goals), and the cliffside landing-foot / rear-foothold
     geometry. The wall-angle range, plate, run-up, and pit offset already
     matched cliffside, so they were never overridden.

  3. a compact 4-waypoint goal circuit (``flyball_swimmer_goal_layout``):
     g0 between-hurdles midpoint -> g1 wall -> g2 between-hurdles midpoint ->
     g3 final return goal (g4..g7 padded). The takeoff and landing waypoints
     are dropped, so the wall goal moves from index 2 to 1 and the return phase
     from 4 to 2; both are declared via ``flyball_wall_goal_idx`` /
     ``flyball_return_goal_idx``, which every env19 goal gate in
     ``legged_robot.py`` reads. Goal promote/demote fractions are re-anchored to
     the shorter circuit.

The per-angle acceptance radii / kappas (``wococo_per_angle_radii_low``,
``wococo_foot_region_radius``, ``wococo_landing_foot_region_radius``, ...) were
INTENTIONALLY left at the teacher values by the swimmer layer — only the goal
*positions* moved. They are unchanged here.

    python train.py --exptid <id> --task go2_teacher --device cuda:0
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO


# ACTIVE [swimmer]. Hand-tuned, row every 5 deg 30-80. final_on_face=True, so
# the z-offset lift and rear +y shift are skipped for it.
_FOOTHOLD_TABLE_MANUAL_30_80 = (
    LEGGED_GYM_ROOT_DIR + "/resources/footholds/foothold_table_manual_30_80.npz"
)

# Opt-in alternative (the pre-swimmer default): rows at 20/30/40/50 deg, pre-lift
# (z-offset and rear +y shift apply on top), end-clamped above 50 deg. Selecting it
# means reverting flyball_foothold_z_offset_m to 0.15 and the rear +y shift to 0.00.
_FOOTHOLD_GOALS_DEFAULT = (
    LEGGED_GYM_ROOT_DIR + "/resources/footholds/foothold_table.npz"
)


class Go2TeacherCfg( Go2RoughCfg ):

    class sim( Go2RoughCfg.sim ):
        class physx( Go2RoughCfg.sim.physx ):
            default_buffer_size_multiplier = 10   # wall base spikes collision pairs

    class env( Go2RoughCfg.env ):
        episode_length_s = 20
        roll_term_deg = 95.0    # banked wall plant needs large roll tolerance
        next_goal_threshold       = 0.2
        terminate_on_lateral_deviation = True
        lateral_deviation_term_m       = 1.5
        # Must stay False to reproduce main's teacher. In main the teacher runs
        # WITHOUT a camera, and legged_robot.check_termination gates this cutoff on
        # `cfg.depth.use_camera`, so main's teacher effectively never terminated on
        # yaw error. The dual-encoder teacher DOES take a camera, which would arm
        # this cutoff and silently add an episode-end main never had. (The distil
        # tasks set False for the same reason.)
        terminate_on_yaw_tracking = False
        # No stall termination and no behind-the-wall termination: main's teacher
        # has neither. terminate_on_low_velocity is opt-in and defaults to False in
        # the code, so it is simply left unset. terminate_behind_wall defaults to
        # True in the SHARED base config (it was added for later flyball work), so it
        # has to be pinned off here to keep the teacher's episode ends identical to
        # main's -- otherwise walking around the plate would end the episode early.
        terminate_behind_wall = False
        # Start-pose randomization stays OFF to match main's teacher run: main never
        # jittered spawn xy/yaw, so the warm-start must see the same reset
        # distribution. rand_yaw_range keeps main's value (inert while
        # randomize_start_yaw is False).
        randomize_start_pos = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2

    class commands( Go2RoughCfg.commands ):
        # Per episode: STANDSTILL (random 0-4 s) -> RUN -> HARD STOP.
        wococo_bimodal_cmd   = True
        # Force the bimodal wall-command sampler ON regardless of the wococo_stage
        # reward scale (otherwise the nostage ablation falls through to the generic
        # continuous U[lin_vel_x] draw, e.g. 0.38 m/s). No-op for non-wall classes.
        wococo_bimodal_cmd_force = True
        wococo_bimodal_env_classes = [19, 15]
        initial_standstill_max_s = 4.0
        wococo_zero_cmd_prob     = 0.0
        resample_only_on_reset = True
        class ranges( Go2RoughCfg.commands.ranges ):
            lin_vel_x = [0.0, 1.0]
        class max_ranges( Go2RoughCfg.commands.max_ranges ):
            lin_vel_x = [1.0, 1.0]

    class control( Go2RoughCfg.control ):
        stiffness = {'joint': 35.}   # sim2real-validated
        damping   = {'joint': 0.6}

    class domain_rand( Go2RoughCfg.domain_rand ):
        # PD-gain mismatch randomized alongside the observed motor-strength mismatch.
        randomize_pd_gain          = True
        randomize_motor_offset     = True
        randomize_restitution      = True
        restitution_range          = [0.0, 0.5]
        randomize_friction         = True
        friction_range             = [0.6, 2.0]
        # Per-env, per-joint PD gain multipliers.
        kp_range                   = [0.95, 1.05]
        kd_range                   = [0.95, 1.05]
        randomize_base_mass        = True
        base_mass_range            = None
        added_mass_range           = [0.0, 3.0]
        randomize_base_com         = True
        added_com_range            = [-0.2, 0.2]

        # Disturbance settings (pushes remain disabled).
        push_robots                = False
        push_interval_s            = 4
        max_push_vel_xy            = 0.25
        max_push_vel_z             = 0.0
        max_push_ang_vel           = 0.25

        randomize_motor            = True
        motor_strength_range       = [0.8, 1.2]

        # Fixed-delay model: no per-env action/proprio latency DR. Pinned to main's
        # teacher run so the dual-encoder branch differs only in the actor input.
        randomize_action_latency   = False
        action_latency_range       = [0, 2]        # inert while the flag above is False
        randomize_proprio_latency  = False
        action_delay               = True
        action_delay_latch_mean_level = 6.0
        action_delay_latch_step       = None
        action_delay_ramp_steps       = 0
        # This task is a TEACHER whose camera is a live perception sensor, not a
        # stand-in for the height scan. train.py's distillation guard would otherwise
        # force --delay and latch the 1-step (20 ms) delay at step 0, arming it ~6
        # terrain levels earlier than main's teacher. Opting out keeps main's actual
        # behaviour: 0-delay episodes until the fleet mean level first reaches 6.0.
        # Students/distillation configs leave this at the default (True).
        action_delay_distillation_guard = False
        randomize_contact_threshold = False

        # Fleet-mean DR ramp: the ceiling reaches full DR at mean level 6, and each
        # reset samples a per-env scale below it. Resumes are pre-armed by helpers.py.
        dr_scale_range             = [0.0, 1.0]
        dr_lock_to_terrain_level   = True
        dr_ramp_use_mean_level     = True
        # main's per-env draw is kept: each reset samples U[s_lo, ceiling] per env, so
        # the fleet still sees a real spread of DR strength. Only the CEILING below is
        # made sticky -- the draw itself is untouched.
        dr_scale_sample_per_env    = True
        # The single deliberate deviation from main: the ceiling is latched at its
        # running max instead of being re-read from the current mean level. Otherwise a
        # curriculum demotion drags the randomisation down with it, and the low terrain
        # rows would never again expose the policy to the high-DR regime it already
        # unlocked -- each env would only meet high DR on the rows it happens to climb.
        dr_ramp_monotonic          = True
        dr_ramp_full_at_mean_level = 6.0
        dr_ramp_arm_below_mean_level = 0.5
        dr_ramp_start_armed        = False

    class depth(Go2RoughCfg.depth):
        # The dual-encoder teacher is the DEPTH-CONSUMING policy, so its image chain
        # is copied verbatim from main's Go2StudentCfg.depth -- main's teacher never
        # takes a camera, so the student is the only deployment-faithful reference.
        # Verified key-by-key: this block now resolves to exactly the same 64 depth
        # keys as main's student (the one exception is hole_fill_backend, see below).

        # Exact physical-camera ray cast. 480x270 is the D435 resolution, and it is
        # what makes the PW-hole / edge-corruption lengths below physically mean
        # anything: pw_hole_scale is in RAW pixels, so filling a 106x60 frame with a
        # 32-pixel blob would smear ~5 cm instead of the intended ~6 mm.
        original = (480, 270)

        # Deploy pads exposure->latent to a hard 80 ms. In 20 ms capture frames:
        # floor 4 = that constant, cap 5 = tick phase + measured late tail.
        latency_range = [4, 5]
        # Real deployment always has this 80-100 ms transport latency. Do not
        # let the depth-noise curriculum create artificially fresh observations.
        latency_scale_with_noise = False

        # The 50 Hz capture grid (capture_interval = 1, inherited) exists ONLY to
        # give the lag 20 ms resolution: the encoder runs at 10 Hz, so 4 of every
        # 5 frames an env renders are never shown to a network. Cast per-env on
        # demand instead -- identical frames and lag, ~5x fewer rays at 480x270.
        render_on_demand = True
        empty_below_m = 0.15   # sub-15 cm readings become far/invalid

        # Only the low-rate, large PW hole process remains enabled below.
        noise_scale_per_env = True
        noise_scale_min = 0.0
        noise_scale_init_random = True

        # Real D435 sees its own legs. Changes the depth distribution, so a policy
        # trained without it cannot be resumed from, only re-distilled.
        self_geometry = True
        self_geometry_links = None

        # Geometric camera artifacts are disabled; only PW holes remain.
        stereo_shadow = True
        stereo_baseline = 0.010
        pw_blindspot_cols = 0     # superseded by the geometric band

        # All other stochastic depth terms are disabled. Edge corruption runs on the
        # full 480x270 Warp frame: 10 cm gradient threshold, half the flagged pixels,
        # no dilation so artifacts stay on the actual silhouette edge.
        pw_edge_thresh = 0.10
        pw_edge_prob   = 1.00
        pw_edge_dilate = 0
        dis_noise      = 0.0
        noise_gaussian = 0.0
        noise_dropout  = 0.0
        pw_hole_prob   = 0.007
        # PW runs before resize; ~64 raw pixels = the old ~8 policy pixels, preserving large organic blobs.
        pw_hole_scale  = 32
        pw_hole_smooth_sigma = 1.5  # rounded organic blob footprints
        pw_hole_drift = 0.50       # move across the full frame, still coherent
        pw_empty_min  = 0.0
        pw_blur_sigma = 0.0

        # Same deployment-compatible RealSense fill after the PW holes.
        hole_fill = True
        hole_fill_iters = 8
        # Same algorithm as main's student (which only had the numpy/cv2 path), but
        # as max/min pools: bit-identical output (tests/test_hole_fill_backend.py),
        # 0.3 ms vs 481 ms per control step at 480x270 x 280 envs. The cv2 path is
        # unusable at this resolution, and this branch's own student config already
        # uses "gpu" for the same reason.
        hole_fill_backend = "gpu"
        hole_fill_upscale = False
        hole_fill_width = 480
        hole_fill_height = 270

        # Keep the training terrain mix (100% flywall_box) during camera replay.
        # The default True rewrites to a multi-terrain mix for student distillation,
        # but here we want the same terrain the teacher trained on.
        override_terrain_mix_for_camera = False
        # Also keep the same terrain grid (horizontal_scale, num_cols) so the
        # wall position matches the height-scan layout exactly.
        override_terrain_grid_for_camera = False

    class noise(Go2RoughCfg.noise):
        # The DR ramp above scales these proprio/height noise samples by dr_scale;
        # privileged state stays clean. Depth DR is deliberately not overridden here.
        add_noise = True
        noise_level = 1.0

        class noise_scales(Go2RoughCfg.noise.noise_scales):
            rotation = 0.05

    class asset( Go2RoughCfg.asset ):
        penalize_contacts_on        = ["thigh", "calf", "Head"]
        terminate_after_contacts_on = ["base", "Head"]

    class terrain( Go2RoughCfg.terrain ):
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3,
                             0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        # Max-pool only over recorded hurdle X-ranges, so a thin bar between
        # 15 cm samples stays visible (radius_x=8 -> 17 cm at 1 cm res).
        scandot_maxpool = True
        scandot_maxpool_hurdles_only = True
        scandot_maxpool_radius_x = 8
        scandot_maxpool_radius_y = 0
        scandot_occlusion_mask = True

        curriculum  = True
        no_demotion = False   # the demote band below needs demotion on
        num_cols    = 80

        horizontal_scale        = 0.01   # 1 cm cells represent the thickness curriculum
        terrain_length          = 10.0
        border_size             = 1.5
        hf2mesh_method           = "fast"
        max_error                = 1.0

        height = [0.02, 0.06]  # curriculum roughness for the rough half of tiles
        flyball_roughness_zero_prob = 0.5  # 50% smooth tiles, 50% rough curriculum

        flyball_add_hurdle                  = True
        flyball_add_second_hurdle           = True
        # Rows 0-3 are hurdle-free. Rows 4-9 interpolate 7 cm -> 12 cm.
        flyball_single_hurdle_height_range  = (0.07, 0.12)
        flyball_single_hurdle_zero_below_level = 4
        flyball_single_hurdle_height_max      = 0.12
        # Bar presence and height DECORRELATED from row difficulty: lerping bar height
        # on the difficulty variable taught the arms to read bar height as a difficulty
        # cue. On rows >= 4 each slot is independently EMPTY with probability 0.25 or a bar with height uniform over the
        # (0.07, 0.12) range — never below the 7 cm deployment minimum, and
        # never a function of the row. The jitter is zeroed here so a revert to the
        # lerp cannot reintroduce sub-minimum bars.
        flyball_hurdle_height_decorrelate   = True
        flyball_hurdle_zero_prob            = 0.25
        # Hurdle count: main teacher has no single-hurdle slot, i.e. every non-empty
        # tile carries TWO bars. flyball_single_hurdle_prob is left at its code
        # default (0.0) so the tile distribution matches main exactly.
        flyball_hurdle_height_jitter_m      = 0.0
        flyball_single_hurdle_thickness      = 0.01
        flyball_single_hurdle_thickness_range = None
        flyball_single_hurdle_thickness_range_uniform = False
        flyball_hurdle_thickness_jitter_m     = 0.0
        flyball_single_hurdle_thickness_max   = 0.01
        flyball_single_hurdle_width           = 1.20
        flyball_single_hurdle_width_range     = (1.00, 1.30)
        # Clearance parabola: 20 cm apex at hurdle centre, roots 10 cm each side.
        flyball_hurdle_clear_half_span         = 0.10
        flyball_hurdle_clear_box_height        = 0.17
        flyball_hurdle_roughness_height       = (0.0, 0.01)
        # Distance jitter scales with difficulty (main): True = low rows get a narrower
        # range; False would give every row the full [1.30, 1.90] m range.
        flyball_hurdle_jitter_scale_with_difficulty = True

        flyball_spawn_x                       = 1.0
        flyball_hurdle_front_from_spawn       = 1.60
        flyball_hurdle_front_from_spawn_jitter = 0.30
        flyball_second_hurdle_offset          = 2.00
        flyball_second_hurdle_offset_jitter   = 0.30
        flyball_wall_to_hurdle_rear           = 2.00
        flyball_wall_to_hurdle_rear_jitter    = 0.30
        # Hard floor on the JITTERED hurdle-2-rear -> wall-toe gap: the -0.30 tail put
        # the last hurdle 1.70 m from the wall, too little run-up for plant speed.
        # Floors rather than shifts: ~half the tiles sit at 2.00 m, the rest 2.00-2.30 m.
        flyball_wall_to_hurdle_rear_min       = 2.00
        flyball_post_hurdle_goal_distance     = 0.0
        flyball_randomize_return_goal_x        = True
        flyball_plate_width                 = 1.3
        flyball_plate_length                = 1.1   # along-slope (hypotenuse)
        flyball_runup_dist_range            = (2.25, 2.25)  # hurdle-free fallback only
        flyball_runup_jitter                = 0.0
        flyball_pit_height_offset_range     =  (0,0)
        flyball_wall_angle_deg_range        = (30.0, 75.0)  # +-5 jitter -> true max 80
        flyball_wall_angle_jitter_deg       = 5.0

        # foothold GOAL LOCATIONS [swimmer]: hand-authored per-5deg table (30-80 deg);
        # flyball_wall.py skips the z-offset clamp / >50deg handling for it.
        flyball_foothold_table_path         = _FOOTHOLD_TABLE_MANUAL_30_80
        # [swimmer] 0.05 (was 0.15 for the default foothold goals). Raises every WALL
        # foot target by this much in z (up-slope by dz/sin(theta), so the target stays
        # on the face and x moves toward the wall by dz/tan(theta)). Inert for a
        # final_on_face table such as the one above.
        flyball_foothold_z_offset_m         = 0.05

        # Front-foot floor (landing) targets used by the post-turn landing stage.
        flyball_landing_foot_standoff       = 0.80   # [swimmer] goals 5 cm farther from the wall
        flyball_landing_foot_y_mid          = 0.00   # [swimmer] center FL/FR goals on the corridor y
        flyball_landing_foot_stance_half    = 0.10
        flyball_landing_foot_x_stagger      = 0.10   # [swimmer] was 0.00
        # Rear wall footholds: shift +5 cm in +y on walls steeper than 50 deg.
        flyball_rear_foothold_positive_y_shift_m  = 0.05   # [swimmer] was 0.00
        flyball_rear_foothold_shift_min_angle_deg = 50.0   # [swimmer]

        # Compact 4-waypoint goal circuit [swimmer]: g0 between-hurdles midpoint,
        # g1 wall contact, g2 that midpoint again on the return leg, g3 final return
        # goal (g4..g7 padded). Takeoff and landing waypoints are dropped.
        flyball_swimmer_goal_layout = True
        # Every env19 goal gate in legged_robot.py reads these two indices, so
        # they MUST track the layout above (wall 2 -> 1, return phase 4 -> 2).
        flyball_wall_goal_idx   = 1
        flyball_return_goal_idx = 2

        # Cold curriculum start: every env begins on row 0 and has to earn its way up.
        # main's teacher starts spread over [0, 5]; starting at 0 is what makes the DR
        # ramp (which reads the fleet mean level) genuinely begin from zero.
        max_init_terrain_level = 0
        resample_at_max_level  = True
        resample_min_level     = 4
        # [swimmer] Goal curriculum re-anchored to the 4-waypoint circuit. max_goal_idx
        # means: 1 = passed g0, 2 = plant complete, 3 = back between the hurdles,
        # >=4 = finished. Promote at "plant done + one waypoint", demote at "no plant".
        promote_goal_frac      = 0.375   # int(0.375*8) = 3
        demote_goal_frac       = 0.25    # int(0.25*8)  = 2

    class rewards( Go2RoughCfg.rewards ):
        base_height_target = 0.25
        max_contact_force = 110.0   # [N] per-foot resultant-force threshold
        exempt_hurdle_contact_from_clip   = True
        exempt_hurdle_clearance_from_clip = False
        exempt_wococo_contact_pts_from_clip = True   # n_wrong must reach negative
        # OFF. With the clip on (legged_gym's default), a step whose term sum is
        # negative is rewritten to 0 before `termination` is added, so the ONLY
        # thing that reaches the gradient is the -3.0 fall price: measured on
        # my-teacher ckpt1000, the episode ledger was +0.35 of task reward against
        # -34.4 of penalties, clipped to 0, leaving Mean reward (total) = -0.03 =
        # 1% falls * -3.0. That objective pays for standing still and nothing else
        # (the policy crouched, froze at 98.5% timeout, terrain_level 0 for 1143
        # iterations), so the shaping terms must keep their sign. IsaacLab's Go2
        # flat config has no such clip either.
        # The two lines below are inert while this is False; they only matter if
        # the clip is ever switched back on.
        only_positive_rewards = False
        only_positive_rewards_exempt_high_levels = True
        only_positive_rewards_clip_below_level    = 2

        # Arms at a fixed iteration, not a level: a level trigger feeds back (the
        # penalty demotes the fleet, and demotion then disarms it).
        hurdle_clearance_arm_step             = 2000 * 24
        hurdle_clearance_apex_full_mean_level = None
        hurdle_clearance_arm_mean_level       = 0.0
        hurdle_clearance_ramp_delay_steps     = 0
        hurdle_clearance_ramp_steps           = 4000 * 24
        # Left at main's value on purpose: this term's alpha rides a fixed iteration
        # budget (arm_step / ramp_steps), not the terrain curriculum, so it keeps the
        # per-env Bernoulli(p=alpha) draw. Only the curriculum-driven DR below went
        # fleet-wide.
        hurdle_clearance_per_env_draw        = True
        hurdle_pair_sym_sigma_rad             = 0.40   # summed mirror error per pair

        exempt_penalties_at_wall_plant = False
        wococo_roll_exempt_deg              = 100.0
        wococo_roll_exempt_post_sentinel_s  = 2.0

        progressive_tracking_gating       = False
        gate_tracking_during_wococo_plant = True
        tracking_goal_vel_unmask_stage    = 2   # vel re-enables at the LANDING
        tracking_yaw_unmask_stage         = 1   # yaw re-enables at the REAR_PLANT
        tracking_fadeout_episodes         = 0.0
        gate_tracking_yaw_at_standstill   = True
        standstill_dof_error_scale        = 20.0
        standstill_dof_error_ramp_s       = 1.0
        standstill_orientation_sigma      = 0.05

        # Reaching the final goal zeroes the command and holds the episode open instead
        # of resetting, so braking to a standstill is trained.
        enable_final_goal_hard_stop = True
        final_goal_stop_s           = 5.0

        # P4L stages: 0 FRONT_PLANT -> 1 REAR_PLANT -> 2 LANDING -> 3 sentinel.
        wococo_design               = 'P4L'
        wococo_plant_complete_stage = 2   # wall goal advances entering the landing
        wococo_landing_foot_region_radius = 0.60
        wococo_landing_contact_force_threshold = 5.0
        wococo_landing_roll_tol_deg = 25.0
        # Under wococo_landing_posture_only this is the tolerance from the RETURN
        # BEARING (of flyball_return_goal_idx), not from a hard-coded 180 deg as in the
        # foothold landing, which reads the same knob as "within X deg of 180" — so
        # reverting posture_only tightens that gate from the historic 45/70 to 40.
        wococo_landing_yaw_tol_deg  = 40.0
        wococo_landing_F_task_sigma = 1.5
        wococo_landing_stagger_window_steps = 30   # lets the pair land sequentially
        wococo_goal_advance_mode    = 'plant_stage'
        wococo_stage_reward_mode    = 'oneshot_linear'
        wococo_n_wrong_weight       = 1.0

        # Anneal the one-shot stage bonus to zero on FLEET-MEAN terrain level: it buys
        # its faster curriculum onset below level 5, and kept live afterwards it keeps
        # compressing the plant trigger against distribution-specific cues. Ramps
        # 25 -> 0 over fleet-mean level 5 -> 7 and LATCHES, so demotions and resumes
        # cannot resurrect it. The FSM is unaffected; start = None disables.
        wococo_stage_anneal_start_mean_level = 5.0
        wococo_stage_anneal_end_mean_level   = 7.0
        wococo_stage_anneal_final_frac       = 0.0

        # Posture-only LANDING (stage 2), ON: stage 2 stops regulating foot contacts
        # entirely (no footholds, no n_wrong, n_con = 0) and becomes a pure BASE-STATE
        # gate on the return posture: |roll| < roll_tol, |pitch| < pitch_tol,
        # |yaw - bearing(return goal)| < yaw_tol, height in [h_lo, h_hi], and
        # return-projected speed >= frac * cmd, all held for wococo_landing_n_persist
        # steps (that hold is what makes the gate unreachable in mid-air).
        # Turning this OFF restores the foothold landing but
        # ALSO requires reverting wococo_landing_n_persist (-> 1) and the
        # wococo_landing_foot_target scale (-> 25.0) below.
        wococo_landing_posture_only  = True
        wococo_landing_pitch_tol_deg = 35.0   # a 1 m/s trot pitches +/-5-10 deg
        wococo_landing_pitch_weight  = 1.0    # pitch weight in the dense err
        wococo_landing_height_weight = 10.0   # metres of height shortfall -> rad
        wococo_landing_h_lo          = 0.15   # below = trunk down (belly-slide)
        # h_hi bounds ballistic dwell: a body entering the band from above holds
        # it for at most sqrt(2*(h_hi-h_lo)/g) = 10.1 steps at [0.15, 0.35].
        # RAISING h_hi WITHOUT RAISING n_persist reopens mid-air completion.
        wococo_landing_h_hi          = 0.35
        wococo_landing_speed_frac    = 0.8    # fraction of commands[:, 0]
        # Stage-2-only multiplier on wococo_base_orient (stages 0/1 unaffected).
        # 1.5 = +50%: the ONLY dense signal for roll/pitch/base-height at the landing,
        # carrying a 15-step pause-on-error hold. Raising it also raises the landing's
        # draw on wococo_income_episode_cap -- the two must move together.
        wococo_landing_base_orient_gain = 1.5

        # Per-wall-angle radii OFF: every consumer falls back to its flat scalar below.
        # A dict re-enables it (interpolated per env angle, clamped outside 30-80 deg).
        wococo_per_angle_radii = None
        # Cold-start gates, restricted to bootstrap levels 0-3; above level 3 the scalar mature-row tolerances hold.
        wococo_per_angle_radii_low_below_level = 4
        wococo_per_angle_radii_low = {
            30.0: (0.500, 0.12, 0.35, 0.15, 0.850, 0.20),
            35.0: (0.480, 0.12, 0.35, 0.15, 0.820, 0.20),
            40.0: (0.420, 0.12, 0.30, 0.15, 0.780, 0.20),
            45.0: (0.350, 0.11, 0.23, 0.15, 0.660, 0.18),
            50.0: (0.275, 0.09, 0.18, 0.15, 0.540, 0.18),
            55.0: (0.253, 0.09, 0.15, 0.15, 0.480, 0.17),
            60.0: (0.209, 0.09, 0.10, 0.15, 0.456, 0.17),
            65.0: (0.198, 0.08, 0.10, 0.15, 0.432, 0.17),
            70.0: (0.209, 0.08, 0.10, 0.15, 0.420, 0.16),
            75.0: (0.209, 0.08, 0.10, 0.15, 0.420, 0.16),
            80.0: (0.209, 0.08, 0.10, 0.15, 0.420, 0.16),
        }
        # Dict tuple order: (wall_accept_r, wall_kappa_oneshot, foot_track_kappa,
        # base_orient_kappa, landing_accept_r, landing_kappa_oneshot)

        wococo_foot_track_kappa          = 0.20
        wococo_foot_target_kappa_oneshot = 0.12
        wococo_landing_foot_target_kappa_oneshot = 0.20
        wococo_foot_target_kappa         = 0.20
        wococo_foot_target_force_thresh  = 20.0

        enable_premature_wall_termination = False

        # Plant-FSM eases reverted to 125a41f: full-strength stage-0 rear-air
        # gate and 40 deg stage-1 yaw tolerance from level 0 (no bootstrap ease).
        enable_wococo_plant_fail_termination = True
        wococo_plant_fail_term_arm_mean_level = 4.0
        # Kill the episode when the base overshoots the landing footholds toward spawn
        # by more than this. RAISED 0.15 -> 0.45 for the posture-only landing, putting
        # it in DIRECT CONFLICT with the stage-2 gate, which demands
        # wococo_landing_speed_frac * cmd for
        # wococo_landing_n_persist steps, i.e. the base MUST travel
        #     0.8 * 1.0 m/s * 15 * 0.02 s = 0.24 m
        # toward spawn while holding it. At margin 0.15 the robot died after 0.15 m,
        # before the required hold could finish: measured at level 9, 57% of episodes
        # (12/21) died on wococo_stage2_overshoot.
        # 0.45 = 0.24 mandatory travel + 0.21 slack. KEEP THIS >= that product
        # if n_persist / speed_frac / the command band change.
        wococo_stage2_overshoot_margin_m     = 0.45
        wococo_stage1_ground_max_z           = 0.05

        # Pinned ON to match main's teacher, which never exempted the attitude /
        # contact terminations. Kept as a knob only: t18 set it False and measured
        # what the exemption costs once armed -- with ramp_steps = 0 it deleted
        # roll/pitch/base_contact termination in ONE step (mean reward +0.03 ->
        # -48.57, term_timeout 0.12 -> 0.42, s1_hold_ok 0.303 -> 0.000), and no
        # reward term pays for self-righting, so the robot fell and lay still.
        enable_wococo_armed_term_exempt      = True
        wococo_armed_term_exempt_min_mean_level = 6.0
        wococo_armed_term_exempt_ramp_steps  = 0
        wococo_armed_roll_term_max_deg       = None
        wococo_armed_pitch_term_max_deg      = None
        wococo_armed_disable_attitude_terms_at_full = True
        wococo_armed_base_contact_grace_steps = None

        # Anti-exploit: cap on positive sub-sentinel shaping income (contact_pts /
        # foot_track / base_orient) per episode. Raised 3.0 -> 3.1 for the posture-only
        # landing, whose dense shaping rides on the capped base_orient while the wall
        # stages exhaust 3.0 alone. The landing branch earns at most
        # scale * gain * dt * 1.0 * n_persist = 0.25 * 1.5 * 0.02 * 1.0 * 15 = 0.1125,
        # so +0.15 fits it. Re-derive if the gain or n_persist changes.
        wococo_income_episode_cap = 3.15
        # E10 (teacher-from-flat-v4-2000 post-mortem): the warm-started policy
        # collapsed into 'press the wall, stay alive, wait for timeout' -- timeout
        # costs 0 and only falls pay termination=-150, so waiting dominates
        # attempting the plant. 11815 iters, s0_hold_ok_frac ~0.001, zero
        # curriculum promotions. Dwell termination makes stalling expensive:
        # once armed (committed to the wall) the stage must advance within
        # 200 steps (4 s) or the episode is hard-terminated (-150).
        # OFF again for the scratch line: a cold policy needs an unpenalized
        # exploration window around the wall; re-enable if the stall equilibrium
        # reappears (it was verified to fire correctly, term_plant_dwell ~0.5).
        enable_plant_dwell_termination = False
        wococo_plant_dwell_max_steps   = 200
        wococo_n_persist = 5   # anti-tap: hold the foothold N steps to bank
        wococo_stage0_n_persist = 5
        wococo_stage1_n_persist = 3
        # Landing hold, set to the reported value of 1.
        # WARNING: under wococo_landing_posture_only there is no foot-contact or
        # foothold test, so this hold is the ONLY thing preventing the stage from
        # completing in mid-air. A ballistic body entering [h_lo, h_hi] from above
        # satisfies the height band for sqrt(2*(h_hi-h_lo)/g) = 0.202 s = 10.1
        # steps, and horizontal speed / attitude are all satisfiable in flight.
        # At 1 the landing was measured completing in a SINGLE mid-air step; 15
        # steps (0.30 s) cleared the ballistic window by ~48%. Persistence 1 is only
        # sound together with the foothold-based landing (wococo_landing_posture_only
        # = False plus the >5 N in-target contact test), as the reported config used it.
        wococo_landing_n_persist = 1
        wococo_foot_region_radius         = 0.30
        # [swimmer] Per-level wall-press force gate. legged_robot.py prefers this table
        # when it is a non-empty 1-D sequence and falls back to the scalar below only
        # when it is None, so the scalar is inert as written.
        wococo_wall_press_force_threshold_by_level = [10.0, 10.0, 35.0, 45.0, 50.0]
        wococo_wall_press_force_threshold = 10.0   # 125a41f flat gate (fallback only)
        wococo_arm_x_offset_m             = 0.0
        # Jackpot discount min(run_len,K)/K; run_len caps at the stage n_persist, so K=1 at the landing restores the full one-shot bonus.
        wococo_jackpot_persist_K = 6.0
        wococo_landing_jackpot_persist_K = 1.0
        hip_pos_episode_multiplier = 1.0

        class scales( Go2RoughCfg.rewards.scales ):
            collision = -20.
            dof_pos_limits = -10.0

            tracking_goal_vel = 1.5   # combined tracking budget 2.0
            tracking_yaw      = 0.5
            standstill_orientation = 0.5
            feet_contact_forces = -0.01

            # WoCoCo wall-plant set. contact_pts drives the FSM and must never
            # be 0: _prepare_reward_function POPS zero scales, which would
            # freeze the stage machine.
            wococo_contact_pts = 0.5
            wococo_stage       = 25.0
            wococo_foot_track  = 0.75
            wococo_foot_target = 25.0
            # OFF under wococo_landing_posture_only: the one-shot payout for putting each
            # FRONT foot on its landing foothold, i.e. the foot-placement regulation the
            # posture-only landing removes. Restore to 25.0 with posture_only = False.
            wococo_landing_foot_target = 0.0
            wococo_base_orient = 0.25

            # Hurdle-WoCoCo FSM and the legacy hand-tuned wall rewards are off.
            wall_push_force        = 0.0
            wococo_hurdle_contact_pts = 0.0
            wococo_hurdle_stage       = 0.0
            wall_turn_left         = 0.0
            wall_feet_alignment    = 0.0
            wall_approach_decel    = 0.0
            forbidden_zone_contact = 0.0
            wall_multi_touch       = 0.0
            hurdle_contact         = -10.0

            termination      = -150.0
            timeout          = 0.0
            hurdle_clearance = -1.0   # per body per step inside the keep-out shell
            hurdle_pair_sym  = 5.0    # one-shot per leg pair per hurdle slot


class Go2TeacherCfgPPO( Go2RoughCfgPPO ):
    class policy( Go2RoughCfgPPO.policy ):
        init_noise_std = 1.0
        max_noise_std  = 1.0
        # The actor must receive the same dual-encoder switches as PPO. Keeping
        # these only under algorithm leaves the actor at 114 inputs and silently
        # discards depth_latent instead of building the 146-input fusion layer.
        use_dual_encoder = True
        num_depth_latent = 32
    
    class runner( Go2RoughCfgPPO.runner ):
        save_interval = 1000
        # main's ladder (1x -> 2x at 2500 -> 5x at 5000) leaves 5000 iterations
        # (~6.5 h here) between checkpoints, and this shared container has already
        # OOM-killed the process once, so that window is the worst-case loss. Keep
        # 1000 up to 5000 and then stay at 2000 iterations (~2.6 h) instead of 5x.
        # 200 checkpoints would also not fit: 112 MB each, disk has ~37 GB free, so
        # 2x means ~100 files (~11 GB) over the full 200000-iteration budget.
        save_throttle = ((0, 1), (5000, 2))
        # The terrain curriculum only starts to climb once the fleet has consumed
        # roughly 6144-env-equivalent x 12000 iterations of experience; at 512 envs
        # main's 10000 would end the run before the first promotion. 200000 leaves
        # enough headroom to walk the DR ramp (full at mean level 6) too.
        max_iterations = 200000
    
    class algorithm( Go2RoughCfgPPO.algorithm ):
        # Dual-encoder consistency loss configuration
        use_dual_encoder = True  # Enable dual-encoder mode (height + depth)
        consistency_loss_weight = 0.1  # Weight for latent consistency loss
        num_depth_latent = 32  # Dimension of depth latent
        
        # Adaptive height weight decay triggers
        height_decay_trigger_success_rate = 0.5  # Min success rate to start decay
        height_decay_trigger_mean_level = 2  # Min terrain level to start decay
        height_decay_trigger_consistency = 0.1  # Max latent consistency to start decay
        height_decay_rate = 0.995  # Decay rate per trigger (0.995 = 0.5% per decay)
