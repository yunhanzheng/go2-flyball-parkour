"""STUDENT — depth distillation of `go2_teacher`. Last link of the chain.

    go2_teacher_config.py  Go2TeacherCfg(Go2RoughCfg)
      -> go2_student_config.py    Go2StudentCfg(Go2TeacherCfg)   <- you are here

    python train.py --exptid xxx-student --task go2_student \\
        --resume --resumeid <swimmer teacher> --use_camera --headless --device cuda:0

Inherits the teacher (not a separate base) so both stages share one dynamics
definition. Adds the depth/camera pipeline, student-specific safety knobs, and
the re-arms a `--resume` needs: `common_step_counter` restarts at 0, so any
teacher schedule expressed as a step count would re-arm from the resume point
instead of from the teacher's own history.
"""

from legged_gym.envs.go2.go2_teacher_config import (
    Go2TeacherCfg, Go2TeacherCfgPPO)


class Go2StudentCfg(Go2TeacherCfg):

    class env(Go2TeacherCfg.env):
        # Highest-impact student knob: this termination kills most approach episodes
        # before the exempted wall goal, because the student cannot see heading in
        # depth. Lateral-deviation termination stays ON.
        terminate_on_yaw_tracking = False

    class commands(Go2TeacherCfg.commands):
        # Restated from the teacher: the bimodal {0,1} sampler is gated on a
        # `wococo_stage > 0` proxy, so a config that zeroes the wococo shaping silently
        # falls back to a continuous U[ranges.lin_vel_x] draw. Verify by histogramming
        # `env.commands`, not `max_command_x`.
        wococo_bimodal_cmd_force = True

    class terrain(Go2TeacherCfg.terrain):
        # Floor raised so every student hurdle is camera-visible; the ceiling
        # reads off the teacher and follows it automatically.
        flyball_single_hurdle_height_range = (
            0.07, Go2TeacherCfg.terrain.flyball_single_hurdle_height_range[1])

        # Rows 0-3 are off-distribution for DAgger (the teacher resamples into
        # [4, 9]). Demotion stays on within the band.
        min_terrain_level = 4

    class rewards(Go2TeacherCfg.rewards):
        hurdle_clearance_start_armed = True   # teacher arms at iter 4000 of ITS run

    class domain_rand(Go2TeacherCfg.domain_rand):
        action_delay_latch_step = 0   # deploy enforces the delay unconditionally

    class depth(Go2TeacherCfg.depth):
        # Exact physical-camera ray cast for the deployment-mirror experiment.
        original = (480, 270)
        # Without these, helpers.py rewrites the terrain mix/grid for the standard parkour set and the resumed teacher faceplants.
        override_terrain_mix_for_camera = False
        override_terrain_grid_for_camera = False

        # Deploy pads exposure->latent to a hard 80 ms. In 20 ms capture frames:
        # floor 4 = that constant, cap 5 = tick phase + measured late tail.
        latency_range = [4, 5]
        # Real deployment always has this 80-100 ms transport latency. Do not
        # let the depth-noise curriculum create artificially fresh observations.
        latency_scale_with_noise = False

        # The 50 Hz capture grid (capture_interval = 1, inherited) exists ONLY to
        # give the lag 20 ms resolution: the encoder runs at 10 Hz, so 4 of every
        # 5 frames an env renders are never shown to a network. `latency_range`
        # cannot be collapsed to buy that back; cast per-env-on-demand instead —
        # identical frames and lag, ~2.5x fewer rays, same deployment latency regime.
        render_on_demand = True
        empty_below_m = 0.15   # sub-15 cm readings become far/invalid

        # Only the low-rate, large PW hole process remains enabled below.
        noise_scale_per_env = True
        noise_scale_min = 0.0
        noise_scale_init_random = True

        # Real D435 sees its own legs. Changes the depth distribution, so students
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
        pw_hole_prob   = 0.007  # 30% below the previous 0.01 rate
        # PW runs before resize; ~64 raw pixels = the old ~8 policy pixels, preserving large organic blobs.
        pw_hole_scale  = 32
        pw_hole_smooth_sigma = 1.5  # rounded organic blob footprints
        pw_hole_drift = 0.50       # move across the full frame, still coherent
        pw_empty_min  = 0.0
        pw_blur_sigma = 0.0
        # Apply the same deployment-compatible RealSense fill after PW holes.
        hole_fill = True
        hole_fill_iters = 8
        hole_fill_upscale = False
        hole_fill_width = 480
        hole_fill_height = 270


class Go2StudentCfgPPO(Go2TeacherCfgPPO):
    class runner(Go2TeacherCfgPPO.runner):
        save_interval = 1000

    class depth_encoder(Go2TeacherCfgPPO.depth_encoder):
        # DAgger action-noise injection, OFF. When > 0 it is applied to the EXECUTED
        # action only, after the clean student output is captured as the DAgger target,
        # so supervision stays clean and only the visited state distribution widens.
        # It was 0.15; hardening measured a 2.9 pp cost on clean dynamics
        # (0.626 vs 0.655), which is why it is disabled here. play.py's
        # --action_noise still injects it at replay time for robustness A/Bs.
        action_noise_std = 0.0
