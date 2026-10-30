# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn
class LeggedRobotCfg(BaseConfig):
    class play:
        load_student_config = False
        mask_priv_obs = False
    class env:
        num_envs = 6144

        n_scan = 132
        n_priv = 3+3 +3
        n_priv_latent = 4 + 1 + 12 +12
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5
        history_len = 10

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_privileged_obs = None # if not None, step() returns a privileged_obs_buf (asymmetric critic obs)
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True
        episode_length_s = 20
        obs_type = "og"
        roll_term_deg = 85.943669  # |roll| > this triggers term_roll. Default ≈ 1.5 rad (legacy hardcoded value).

        # Terminate when |wrap(target_yaw - yaw)| > thresh, keeping the DAgger
        # rollout in the teacher's state support. Active ONLY when
        # depth.use_camera; exempted at the wall via _tracking_stage_mask().
        terminate_on_yaw_tracking = True
        yaw_tracking_term_thresh = 0.6   # rad (≈34°)

        # Terminate if |world-y - env-origin-y| exceeds lateral_deviation_term_m.
        terminate_on_lateral_deviation = False
        lateral_deviation_term_m = 1.5   # m


        
        
        
        history_encoding = True
        reorder_dofs = True
        
        

        include_foot_contacts = True
        
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

        # Feed the "on flat terrain" obs bit (slot 12); False holds it at 0.
        give_flat_terrain_flag = True

    class depth:
        use_camera = False
        camera_num_envs = 280
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.33, 0, 0.08]  # front camera (matches Isaaclab Go2 mount)
        angle = [15, 25]  # positive pitch down (~20° nominal, ±5° noise matching Isaaclab)

        # Capture and encoder cadence in 50 Hz control steps; capturing faster than the encoder gives finer latency history.
        capture_interval = 5
        update_interval = 5  # 5 works without retraining, 8 worse

        # Cast each env only on the capture step whose frame it consumes: same
        # lag, ~update_interval x fewer rays; needs use_warp, capture_interval 1.
        render_on_demand = False

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        latency_range = [0, 0]   # [min, max] capture frames of camera lag
        # True: the DR ramp lerps both bounds from 0 toward latency_range. False:
        # sample the full range from step 0 — use when latency_range mirrors an
        # ENFORCED deploy constant (80 ms pad),
        # so the student never trains on depth fresher than hardware delivers.
        latency_ride_dr_ramp = True

        near_clip = 0
        far_clip = 2
        # RealSense hole fill, applied last on the policy input. Deploy defaults: mode 1, 8 iters.
        hole_fill = False
        hole_fill_mode = 1
        hole_fill_iters = 8
        # "cpu" = the reference numpy/cv2 filter; "gpu" = the same algorithm as max/min
        # pools, bit-identical but ~50x cheaper. At 480x270 x 280 envs the CPU path costs
        # 0.58 s per control step -- 69% of the entire depth step -- almost all of it in
        # the eight dilate passes plus a 145 MB device->host->device round trip.
        hole_fill_backend = "cpu"
        # RealSense min-range (minZ) emulation: distances closer than this read
        # INVALID on the D435 and deploy remaps invalid -> far. Applied to the raw
        # render before noise/resize, matching sim2sim's 0.15 m. 0.0 = off (upstream:
        # close pixels clamp to near_clip, which the real camera never produces).
        empty_below_m = 0.0
        dis_noise = 0.0

        noise_gaussian = 0.0
        noise_dropout = 0.0
        dropout_to_far = False
        # If set (0..1), overrides dropout_to_far: this fraction of dropped pixels read FAR, the rest NEAR.
        dropout_far_frac = None
        # Aggregate dropout into far+near blobs around each seed instead of iid
        # pixels (noise_dropout = seed probability); _ext = chance of a 2nd pixel
        # each side (cluster size 1..2 per colour).
        dropout_cluster = False
        dropout_cluster_ext = 0.6
        dropout_jitter = 0.0     # m; randomize cluster-pixel grayscale around far/near (0 = pure)

        # Parkour-in-the-Wild depth degradation (arXiv 2505.11164 §2.4) on the
        # RESIZED depth in the Warp path: empty (<pw_empty_min m -> far), edge
        # noise, temporal holes, left blind spot, blur. False = no-op.
        pw_noise = False
        pw_empty_min = 0.15      # m; |depth| below this (and invalid 0) -> far/empty
        pw_edge_thresh = 0.10    # m; |depth gradient| above this flagged as an edge
        pw_edge_prob = 0.5       # fraction of edge pixels corrupted (near/far/shuffle)
        pw_edge_near_frac = 1/3  # of corrupted edge pixels: this fraction -> near (closer)
        pw_edge_far_frac = 1/3   # -> far; remainder -> neighbor-shuffle
        pw_hole_prob = 0.10      # fraction of pixels dropped to far via the smooth field
        pw_hole_scale = 8        # hole-blob size = resized_dim // this (low-res field)
        pw_hole_drift = 0.1      # temporal drift of the hole field (0=frozen, 1=iid/flicker)
        pw_blindspot_cols = 5    # max band width; per-env side, drifts (stereo blind spot)
        pw_blindspot_side = 'left'     # 'left' | 'right' | 'random' — which image edge is blind
        pw_blindspot_drift = 0.05      # per-frame random-walk of the band width (0=frozen, 1=iid)
        pw_blindspot_slope_max = 0.15  # max off-vertical lean of the band (px per row)
        pw_blur_sigma = 1.0      # Gaussian blur sigma (px) on the final image; 0 = off

        use_warp = False

        # Warp-only right-imager visibility test for stereo shadows and blind bands.
        stereo_shadow = False
        stereo_baseline = 0.050  # m; D435 imager spacing (D455: 0.095)
        stereo_eps = 0.015       # m; occluder must be this much nearer than
                                 # the hit point (guards endpoint self-hits)

        # Warp-only: also ray-cast the robot's OWN collision primitives so the
        # student sees its legs like the real D435 does. No measurable cost.
        self_geometry = False
        # Optional list of link-name substrings to restrict the pass, e.g.
        # ["FL_", "FR_"]. None = every link with collision geometry; links the
        # frustum cannot reach are provably never hit and cost nothing.
        self_geometry_links = None

        scale = 1
        invert = True

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2
    class noise:
        # On: inject IMU/encoder/height jitter into proprio obs to close the MuJoCo sim2sim gap.
        add_noise = True
        noise_level = 1.0
        quantize_height = True
        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

    class terrain:
        mesh_type = 'trimesh' # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        # --save_vts export resolution [m]: block-max pooling cell size, so
        # ParaView can open dense (1 cm) terrains. <= horizontal_scale = full res.
        vts_export_cell_size = 0.05
        flyball_pit_width_range = (0.20, 0.65)         # [m] Delta_x: pit width in robot forward dir
        flyball_pit_height_offset_range = (0.05, 0.20) # [m] Delta_z: wall base height above approach floor
        flyball_pit_depth = 2.0                         # [m] fixed pit depth (deep enough to be terminal)
        flyball_wall_angle_deg_range = (20.0, 50.0)    # [deg] Alpha: wall incline (50=easy, 90=hard)
        flyball_plate_width = 1.5                       # [m] inclined contact plate lateral width
        flyball_backing_wall_height = 2.0               # [m] full-column wall height above approach floor
        flyball_backing_wall_thickness = 0.35           # [m] wall depth behind the plate's upper edge
        flyball_wall_contact_fxfz_threshold = -0.5     # fx/fz ratio threshold: foot on wall when ratio < this
        # WoCoCo contact region parameters (offsets relative to wall_front_x or corridor centre)
        flyball_liftoff_x_range   = (-1.0500, -0.8500)
        flyball_liftoff_y_mid     = -0.5600
        flyball_liftoff_y_half    =  0.5300
        flyball_liftoff_z_above   =  0.0500
        flyball_wall_y_half       =  0.8200
        flyball_wall_z_min_above  =  0.1800
        flyball_wall_x_depth_scale =  1.0
        flyball_landing_x_range   = (-1.6500, -0.8500)
        flyball_landing_y_mid     =  0.6500
        flyball_landing_y_half    =  1.0000
        flyball_landing_z_above   =  0.0500
        flyball_air_x_range       = (-1.6500, -0.1000)
        flyball_air_y_mid         =  0.6500
        flyball_air_y_half        =  1.0000
        flyball_air_z_min_above   =  0.0500
        flyball_air_z_max_above   =  0.8000
        # WoCoCo hurdle AABBs (per-hurdle, env_class 15): liftoff extends `back` m
        # behind the front face, landing `forward` m past the back face, air sits
        # above each hurdle for the AIRBORNE-stage clearance gate.
        flyball_hurdle_liftoff_back     = 0.60
        flyball_hurdle_liftoff_gap      = 0.20  # [m] gap between the liftoff AABB's far edge and the hurdle's front face, so feet stancing right next to the hurdle don't satisfy the gate.
        flyball_hurdle_landing_forward  = 0.60
        flyball_hurdle_aabb_y_half      = 0.50
        flyball_hurdle_aabb_z_above     = 0.10
        flyball_hurdle_air_z_above      = 0.30
        # Per-hurdle parabolic CLEARANCE volume (_reward_hurdle_clearance): peaks
        # 30 cm above the floor at the hurdle centre, meets the floor 15 cm
        # before/after, extruded across the hurdle width plus pad_y each side.
        flyball_hurdle_clear_pad_x      = 0.05
        flyball_hurdle_clear_pad_y      = 0.05
        flyball_hurdle_clear_pad_z      = 0.05
        flyball_hurdle_clear_half_span  = 0.15
        flyball_hurdle_clear_height     = 0.30
        curriculum = True

        all_vertical = False
        no_flat = True

        # When True, the parkour_flat tile gets a 5 cm × 5 cm raised strip along the
        # corridor centerline as a depth landmark on an otherwise featureless tile.
        centerline_strip = False
        centerline_strip_width = 0.05   # [m] lateral width of the strip
        centerline_strip_height = 0.05  # [m] height above the local height field
        # Width of the smooth (roughness-zeroed) band centered on the strip. None =
        # centerline_strip_width; wider (e.g. 0.10–0.15 m) also cleans its flanks.
        centerline_strip_clear_width = None
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0
        # Occlusion-aware scandots: cast a line of sight from the camera mount to each
        # measured dot; a blocked dot reports the BLOCKER's height, as a depth
        # sensor would. Keeps the teacher from conditioning on ground the
        # depth student can never see (e.g. behind a hurdle). Runs at the
        # depth.update_interval cadence in _post_physics_step_callback.
        scandot_occlusion_mask    = False
        scandot_occlusion_samples = 12    # LOS samples per camera→dot ray
        scandot_occlusion_eps     = 0.05  # [m] clearance before a sample counts as blocking (rides out add_roughness)

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None
        max_init_terrain_level = 5 # starting curriculum state
        resample_at_max_level = False  # if True, envs that promote past the last curriculum row are uniformly redistributed across [0, max-1] (the ad05628 game-inspired curriculum). If False (default), envs are clamped at the last row and stay there. False suits the wococo and foundation training where you want the policy locked into the hardest terrain once it gets there.
        resample_min_level = 0  # floor for the resample_at_max_level redistribution: envs promoting past the last row land uniformly in [resample_min_level, num_rows-1] instead of [0, num_rows-1]. 0 (default) = original full-range behaviour. No effect when resample_at_max_level is False.
        min_terrain_level = 0  # hard FLOOR on the curriculum row an env can occupy. 0 (default) = original behaviour. A positive floor (a) starts the fleet in [min_terrain_level, max_init_terrain_level] instead of [0, max_init_terrain_level] and (b) stops demotion from walking an env below it. Complements resample_min_level, which only floors the promote-past-the-top resample. Use when a resumed/distilled policy must never train on rows its predecessor stopped visiting. Deliberately does NOT touch fixed_level init or a direct terrain_levels assignment, so play.py's --level pin and --eval_min_level sweeps can still probe rows below the floor.
        no_demotion = False  # if True, suppress curriculum demotion (move_down). Envs that promote stay promoted forever. Use to break cascade-collapse where the policy temporarily learns a skill, fails on a few episodes, and the entire population demotes back to level 0 destroying all progress.
        max_promoted_level = None  # if set, integer cap on how far envs can promote (clamps move_up). Use to freeze the curriculum at a moderate-difficulty shelf so the policy consolidates before facing harder rows. None (default) = curriculum runs to num_rows-1.
        terrain_length = 18.
        terrain_width = 4
        num_rows= 10 # terrain rows = curriculum levels
        num_cols = 40 # terrain cols = types
        
        terrain_dict = {"smooth slope": 0.,         
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "parkour_stones": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,
                        "two_hurdles_wall_turn": 0.2,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2,
                        "demo": 0.0,
                        }                         # proportions of each terrain type, should sum to 1.0, order should match terrain types in terrain.py
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 11.43  # tan(85°); snaps only slopes steeper than 85°
        origin_zero_z = True

        num_goals = 8

    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        # If true, ignore resampling_time and draw one command per episode reset.
        resample_only_on_reset = False
        heading_command = True # if true: compute ang vel command from heading error
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4
        zero_command_probability = 0.0  # fraction of resamples that get cmd=(0,0,0); 0 = disabled
        # Easy ranges
        class ranges:
            lin_vel_x = [0.0, 1.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Hard ranges
        class max_ranges:
            lin_vel_x = [1.5, 2.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [0, 0]

        class crclm_incremnt:
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1    # min max [rad/s]
            heading = 0.4

        waypoint_delta = 0.7

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.6, 2.]
        randomize_base_mass = True
        added_mass_range = [0., 3.]
        # Multiplicative alternative to added_mass_range: [lo, hi] x the ACTUAL base
        # mass Isaac reports after collapse_fixed_joints; when set it REPLACES the
        # additive draw. The priv_latent channel still reports signed kg deviation.
        base_mass_range = None
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        delay_update_global_steps = 24 * 8000
        action_delay = False
        action_curr_step = [1, 1]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8
        # Fleet-level latch for the fixed 1-step (20 ms) action delay: with
        # action_delay=True the delay is 0 steps until the FLEET-MEAN terrain
        # level first reaches this value, then 1 step permanently (sticky, and
        # runtime-only, so a resume re-latches). While set, the global-step
        # schedule (action_curr_step / delay_update_global_steps) is ignored.
        # The 20 ms models the ENTIRE deploy proprio pipeline (16 ms enforced
        # receipt->publish pad + ~3 ms firmware/DDS ends, 2026-07-25
        # constant-latency deploy) — do NOT combine with
        # randomize_proprio_latency or randomize_action_latency; that would
        # double-count the same physical delay.
        action_delay_latch_mean_level = None
        # Alternative trigger: latch at a fixed env-step count instead of a curriculum
        # level; when not None this REPLACES action_delay_latch_mean_level. Units are
        # env steps = iteration * num_steps_per_env, so at 24 steps/iteration
        # iteration 5000 = 120 000. Runtime-only, so a --resume restarts the count.
        action_delay_latch_step = 5000 * 24
        action_delay_ramp_steps = 0  # 0 = hard latch; otherwise episode-level 0/1-step delay mix
        # Start with the latch already engaged. Set by --delay (helpers.py);
        # REQUIRED for distillation so the student never trains on undelayed
        # actions the deployed stack cannot produce.
        action_delay_start_latched = False
        # play.py --full-dr sets this True so per-env randomized action latency
        # runs with a GUI viewer open (instead of the constant action_delay_view).
        force_action_latency = False

        # Per-env proprio latency (encoders + IMU), sampled per episode around the
        # paper's 0.016 s, realized sub-control-step (0.016 s < 20 ms) by interpolation.
        randomize_proprio_latency = False
        proprio_latency_range_s = [0.012, 0.020]   # seconds, ~0.016 s ±4 ms

        # Per-env, per-foot contact-detection threshold [N] for the OBSERVED contact
        # bit only — rewards/FSM keep the clean 2 N signal. Models the real Go2
        # foot-force sensors, whose offsets/scales put the deploy threshold anywhere
        # from grazing contact to light stance (FR above threshold in as little as
        # 0.9% of standing samples). Deviation from 2 N scales with dr_scale.
        randomize_contact_threshold = False
        contact_threshold_range = [0.5, 15.0]   # [N], nominal 2.0

    class rewards:
        class scales:
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            ang_vel_z  = 0.0     # off by default. Foundation cfg switches it on. Zeroed during wococo plant stages 1-3 inside _reward_ang_vel_z.
            orientation = -1.
            # POSITIVE base-level carrot paid only while commanded to stand still
            # (|cmd_x| < lin_vel_clip), on every terrain class.
            standstill_orientation = 0.0
            dof_acc = -2.5e-7
            collision = -20.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.8
            dof_error = -0.04
            feet_stumble = -1
            feet_edge = -1
            wall_turn_left = 0.0
            wall_approach_decel = 0.0
            forbidden_zone_contact = -1.0
            # Per-body-per-step penalty for collision-geometry overlap with a hurdle's
            # extruded parabolic clearance volume. Independent of hurdle_contact.
            hurdle_clearance = 0.0
            # P4L stage-2 front-foot landing-force shaping. Task configs opt in.
            wococo_landing_z_force = 0.0
            wococo_landing_positive_x_force = 0.0
            # Thresholded per-foot 3-D contact-force excess cost; suppressed while the WoCoCo arm latch is set.
            feet_contact_forces = 0.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_exempt_high_levels = False  # preserve penalties at and above the configured level
        only_positive_rewards_clip_below_level = 2  # the clip applies to envs at terrain_level < this; at/above it the clip is skipped (only consulted when only_positive_rewards_exempt_high_levels is True).
        exempt_wococo_contact_pts_from_clip = False  # if True, wococo_contact_pts is summed in *after* the only_positive_rewards clip (same pattern as 'termination'), so its n_wrong penalty can drive total reward negative.
        exempt_collision_from_clip = False  # same pattern for the collision penalty: when True, collision is summed in *after* the only_positive_rewards clip so it can drive total reward negative.
        wococo_n_wrong_weight = 0.3  # initial per-env weight for the −n_con·n_wrong_active term in _reward_wococo_contact_pts. Each env's weight ratchets up by `wococo_n_wrong_weight_step` after every clean episode (episode-sum wococo_contact_pts ≥ 0) once the env has reached stage 4 in that episode, capped at `wococo_n_wrong_weight_max`. Per-env, monotonic non-decreasing. Must be ≤ wococo_n_wrong_weight_max or it will be immediately clamped to the cap on first reset.
        wococo_n_wrong_weight_max  = 0.8  # per-env curriculum cap
        wococo_n_wrong_weight_step = 0.05  # +per-episode increment when the just-finished episode reached stage 4 AND had zero negative wococo_contact_pts steps.
        wococo_liftoff_stance_force_threshold = 50.0  # [N] minimum per-foot contact force counted as a "stance" in the wococo stage-0 LIFTOFF block. Lower → easier to register a rear-foot plant (partial-weight or single-leg attempts qualify).
        wococo_wall_press_force_threshold = 30.0  # [N] minimum per-foot contact force counted as a "wall press" in the wococo stage-1/2 FRONT_WALL/REAR_WALL blocks. Lower → easier to register a wall plant (partial-weight presses qualify).
        # Optional table indexed by terrain level (levels past the last entry clamp to it); None keeps the scalar above.
        wococo_wall_press_force_threshold_by_level = None
        wococo_arm_x_offset_m = 0.0  # [m] added to the liftoff AABB far-edge x to set the plant-arm trigger (where the uniform wococo_armed latch — reward + n_wrong penalty — fires). 0.0 = legacy (arm at the liftoff far edge, ~0.90 m in front of the wall). POSITIVE shifts the trigger toward the wall, so the robot must get CLOSER before any wococo plant reward fires (anti "stop-short-and-farm"). Keep well below the ~0.90 m liftoff offset or the plant may never arm (a banked base can't reach the wall front).
        enable_premature_wall_termination = False  # if True, terminate the episode when any foot enters the wall AABB during wococo stage 0 with force > 30 N. Otherwise the soft _reward_collision penalty handles avoidance.
        wococo_stage2_decay_T = 50.0  # [steps] dwell-time T for the wococo stage-2 F_task decay (1 - (t/T)^2). Larger → more grace before stage-2 reward collapses. Lower → harsher pressure to advance to stage 3 quickly.
        wococo_stage3_decay_T = 100.0  # [steps] dwell-time T for the wococo stage-3 LANDING F_task decay (1 - (t/T)^2). Larger → more grace before landing reward collapses. Lower → harsher pressure to commit to a clean landing fast.
        wococo_stage4_entry_bonus = 0.0  # [per_step mode only] one-shot reward when wococo_contact_stage first reaches 4 in an episode. Magnitude is raw; the wococo_stage scale and dt are applied in compute_reward like every other term. Default 0 preserves legacy behaviour. Ignored in oneshot_const / oneshot_linear modes (they already pay on stage entry via their own mechanism).
        wococo_stage_reward_mode = "per_step"  # "per_step" (Eq. 4: n_stage·F_task each step), "oneshot_const" (+1 the first time each new stage is reached in an episode), or "oneshot_linear" (+stage_idx the first time each new stage is reached). One-shot modes ignore F_task — the stage-advance gates in _reward_wococo_contact_pts are the gate.
        wococo_jackpot_persist_K = 1.0  # [steps] persistence horizon for the 2·n_con² jackpot in the reduced-stage FSM (_wococo_contact_pts_few_stage). The jackpot is scaled by min(run_len, K)/K, where run_len is the current consecutive hold_ok streak. K=1 (default) = legacy always-full jackpot (instantaneous contact pays full). K>1 makes a single wall tap pay only 1/K of the jackpot while a sustained hold ramps to full after K steps — kills the tap-farm exploit. Set K above wococo_n_persist so a committed plant earns the stage one-shot anyway; the jackpot need not fully ramp before advancing.
        # Optional per-stage wall persistence overrides; None = wococo_n_persist.
        wococo_stage0_n_persist = None
        wococo_stage1_n_persist = None
        wococo_landing_n_persist = None  # [steps|None] P4L stage-2 landing hold. None reuses wococo_n_persist; set independently to relax only the landing without changing the wall-plant stages.
        hip_pos_episode_multiplier = None  # [float|None] constant multiplier for _reward_hip_pos throughout an episode. None preserves the legacy 5x env19 wall-plant-only multiplier.
        wococo_landing_roll_tol_deg = 10.0  # [deg] absolute stage-3 landing tolerance
        wococo_landing_yaw_tol_deg = 45.0  # [deg] tolerance from 180° (back-facing) for the wococo stage-3 LANDING yaw gate. Pass condition: |yaw| > π − tol_rad. Default 45° → threshold |yaw| > 2.356 rad. Larger → easier (allows more partial turns).
        wococo_landing_region_shrink_from_level = None  # [level|None] terrain level at/above which the P4L landing foothold acceptance radius is multiplied by wococo_landing_region_shrink_factor. None = one width on every row (legacy). Tightens landing precision with the curriculum; note the same radius defines n_wrong, so shrinking it also starts billing loaded front feet that fall outside.
        wococo_landing_region_shrink_factor = 0.5  # multiplier applied to the landing acceptance radius on rows at/above wococo_landing_region_shrink_from_level. 0.5 = half the radius (a quarter of the disc area). Inert while the from_level gate is None.
        wococo_landing_z_force_threshold = 100.0  # [N] front-foot Fz allowed before the quadratic landing-force cost starts.
        wococo_landing_force_normalization = 100.0  # [N] nondimensionalizes both landing-force costs before reward scaling.
        wococo_landing_x_leaky_slope = 0.2  # leaky credit slope for negative world-Fx in the quadratic positive-Fx landing cost.
        max_contact_force = 150.0  # [N] per-foot 3-D force magnitude above which feet_contact_forces applies.
        # WoCoCo hurdle-foothold knobs (env_class 15 only): the wococo Eq.3/Eq.4
        # structure on a separate counter (wococo_hurdle_stage / _idx) shared by both
        # hurdles — on advance to terminal (3) with idx==0, stage resets and idx ticks.
        wococo_hurdle_stance_force_threshold = 20.0  # [N] minimum per-foot force to count as a "stance" plant in the hurdle LIFTOFF / LANDED stages.
        wococo_hurdle_air_decay_T            = 10.0  # [steps] dwell-time T for the hurdle AIRBORNE-stage F_task decay (1 - (t/T)^2). Lower = harsher pressure to land quickly.
        wococo_hurdle_stage_reward_mode      = "per_step"  # "per_step", "oneshot_const", or "oneshot_linear" — same semantics as wococo_stage_reward_mode.
        wococo_hurdle_n_wrong_weight         = 0.3  # multiplier on the −n_con·n_wrong term inside _reward_wococo_hurdle_contact_pts.
        exempt_wococo_hurdle_contact_pts_from_clip = False  # if True, wococo_hurdle_contact_pts is summed in *after* the only_positive_rewards clip so its n_wrong penalty can drive total reward negative.
        # Armed wall-plant termination tolerance. When enabled the fleet-mean trigger
        # is sticky and alpha ramps over synchronous control steps (independent of
        # num_envs). None limits = full exemption at alpha=1; prefer finite overrides.
        enable_wococo_armed_term_exempt = False
        wococo_armed_term_exempt_start_armed = False
        wococo_armed_term_exempt_min_mean_level = 6.0
        wococo_armed_term_exempt_ramp_steps = 0
        wococo_armed_roll_term_max_deg = None
        wococo_armed_pitch_term_max_deg = None
        wococo_armed_disable_attitude_terms_at_full = False
        wococo_armed_base_contact_grace_steps = None
        # hurdle CLEARANCE keep-out penalty (_reward_hurdle_clearance)
        hurdle_clearance_start_armed = False  # trigger its configured delay/ramp at process step 0
        hurdle_clearance_arm_mean_level = 6.0  # sticky fleet-MEAN trigger
        hurdle_clearance_ramp_delay_steps = 0  # synchronous control steps after trigger
        hurdle_clearance_ramp_steps = 0  # 0 preserves the legacy immediate full scale
        hurdle_clearance_per_env_draw = False  # sample penalty activation per episode during its ramp
        exempt_hurdle_clearance_from_clip = False  # if True, hurdle_clearance is summed in *after* the only_positive_rewards clip, so it still bites on envs sitting below only_positive_rewards_clip_below_level.
        gate_tracking_during_wococo_plant = True  # if True, tracking_goal_vel and tracking_yaw return 0 while wococo_contact_stage ∈ {1, 2, 3} (during the FRONT_WALL → REAR_WALL → LANDING sequence). If False, tracking rewards fire at every stage. Stage 0 (approach) and stage 4 (terminal/return) always fire. Ignored when progressive_tracking_gating=True.
        progressive_tracking_gating = False  # if True, tracking is silenced PER ENV based on the highest stage that env has ever reached: silenced iff 1 ≤ current_stage ≤ max_stage_unlocked (sticky across episode resets). Lets a slow env keep tracking signal until it has demonstrated capability at each stage. Supersedes gate_tracking_during_wococo_plant.
        tracking_goal_vel_unmask_stage = None  # [stage|None] per-term unmask under gate_tracking_during_wococo_plant: stage at/after which tracking_goal_vel re-enables during the plant. None = the sentinel (same as tracking_yaw). P4L cliffside sets 2 (the LANDING) so vel drives toward the return goal during the landing while yaw stays masked to the sentinel.
        tracking_yaw_unmask_stage = None  # [stage|None] per-term unmask under gate_tracking_during_wococo_plant: stage at/after which tracking_yaw re-enables during the plant. None = the sentinel (the legacy behaviour). P4L cliffside sets 1 (the REAR_PLANT) so the goal-facing yaw reward is live from the rear-leg plant onward instead of staying dead until the sentinel.
        tracking_fadeout_episodes = 0.0  # if > 0, fade tracking_goal_vel + tracking_yaw from 1× to 0× over N per-env episodes after fadeout_start_episodes. 0 = no fade. Per-env episode count = common_step_counter / max_episode_length.
        tracking_fadeout_start_episodes = 0.0  # per-env episode count at which the fade starts. Tracking holds at 1× until this point, then begins decaying. Default 0 = fade from the very first episode.
        tracking_fadeout_curve = "log"  # "linear" or "log". log = concave (fast early drop, slow tail): factor = 1 − log(1+p) / log(2), where p = (ep − start) / N. linear: factor = 1 − p.
        disable_base_contact_levels = []  # list of curriculum-level indices where base-contact termination is suppressed (only applies when wococo_stage > 0). Empty → fire on every level.
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        standstill_orientation_sigma = 0.05  # kernel width of the standstill_orientation carrot: exp(-‖projected_gravity_xy‖²/sigma), and ‖g_xy‖² ≈ sin²(tilt). 0.05 → ~0.86 at 5° tilt, ~0.55 at 10°, ~0.10 at 20°. Smaller = stricter about a level base.
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        base_height_target = 1.





    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
    
    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        # Clamps of the adaptive schedule. The KL check runs per MINIBATCH (20 lr
        # updates per iteration), so once kl_mean exceeds 2*desired_kl the lr walks
        # down to min_learning_rate within one iteration. Override with --lr_floor.
        min_learning_rate = 1.e-5
        max_learning_rate = 1.e-2
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 5000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
    
    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24
        # Add teacher/student latent L2 loss alongside behavioral cloning.
        use_depth_encoder_loss = True
        # Zero-mean Gaussian noise on the action ACTUALLY SENT TO THE SIM during DAgger
        # collection (P4W Alg. 1: a <- pi_student(o) + N(0, sigma^2)). The BC loss still
        # regresses the CLEAN student output onto the teacher, so this only widens the
        # supervised state distribution. 0.0 = off (teacher itself learned at std ~0.68).
        action_noise_std = 0.0

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio
        num_scan = LeggedRobotCfg.env.n_scan

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000 # number of policy updates

        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
