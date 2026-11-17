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
        # Student-only: count envs that already graduated (promoted past row 9 and then
        # re-sampled into [4, 9]) as the TOP row in every fleet-mean-level statistic.
        # The re-sample is an anti-forgetting device, not lost progress, so the mean must
        # not be dragged down by it (dr_scale ramp / mean-level latches).
        # The teacher keeps main's raw mean (flag absent => off).
        count_graduated_in_mean_level = True

    class rewards(Go2TeacherCfg.rewards):
        hurdle_clearance_start_armed = True   # teacher arms at iter 4000 of ITS run

    class domain_rand(Go2TeacherCfg.domain_rand):
        action_delay_latch_step = 0   # deploy enforces the delay unconditionally
        # Student-only DR semantics; the teacher keeps main's per-env draw and no latch.
        #  * ONE fleet-wide dr_scale (no per-env spread below the ceiling),
        #  * monotonic ramp: DR only rises with the curriculum and never rolls back on a
        #    mean-level dip. See legged_robot.reset_idx.
        dr_scale_sample_per_env = False
        dr_ramp_monotonic       = True

    class depth(Go2TeacherCfg.depth):
        # Exact physical-camera ray cast for the deployment-mirror experiment.
        original = (480, 270)
        # Without these, helpers.py rewrites the terrain mix/grid for the standard parkour set and the resumed teacher faceplants.
        override_terrain_mix_for_camera = False
        override_terrain_grid_for_camera = False

        # Default env count on the camera path. helpers.update_cfg_from_args seeds
        # env.num_envs from this value whenever use_camera + headless, but an explicit
        # --num_envs on the command line is applied AFTER that and therefore wins; this
        # only fixes the default so a run without the flag cannot silently fall back to
        # the base's 280.
        # Why 1024: 280 was sized for pure distillation (one forward per iteration, no
        # PPO) and under depth-RL leaves 6720 samples per iteration = 1680 per minibatch,
        # ~22x below the teacher's 36864, where the advantage estimate is too noisy to
        # read any hyperparameter effect. 1024 gives 24576 samples / 6144 per minibatch
        # at ~1.24 GB of update memory (measured) and ~135 s rollout.
        # Terrain has 10 x 40 = 400 tiles here (student keeps the base grid), so each
        # tile carries ~2.6 envs; curriculum levels stay per-env.
        camera_num_envs = 1024

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

        # Depth-RL (train.py --depth_rl --use_camera): PPO on the depth student
        # against the env reward instead of the DAgger L2 fit. depth_actor + its
        # action std, the depth encoder (joint: replayed inside the PPO graph), the
        # privileged critic and the estimator all move; only the teacher actor is
        # frozen.
        #
        # PROBE RUN: the optimisation recipe below is the TEACHER's verbatim, not a
        # tuned fine-tune set. The point is one controlled experiment -- "does the
        # student improve faster under the exact recipe the teacher was tuned
        # under?". If it stalls, the first knob to put back is rl_init_std.
        rl_depth = False
        # ARM-A CONTROL: the lr is pinned at the value the schedule actually used.
        # long4 sat on rl_min_learning_rate on 125 of 197 late iterations (63%) while
        # kl kept a 2.43 median (max 24.3), so the controller could not go lower and
        # contributed nothing but step-size jitter. With rl_adaptive_lr off, sigma and
        # the encoder are the only moving parts left, and a flat curve becomes an
        # unambiguous verdict on the actor instead of a three-way confound.
        rl_learning_rate = 1.e-5            # teacher: 2e-4
        rl_adaptive_lr = False              # keep the KL diagnostic, drop the schedule
        rl_min_learning_rate = 1.e-5        # teacher: 1e-5
        rl_max_learning_rate = 1.e-2        # teacher: 1e-2
        # Trust-region target, expressed as RMS action-mean displacement / sigma.
        #
        # Why not a fixed desired_kl: for this Gaussian kl = 6*(dmu_rms/sigma)^2, so a
        # fixed kl silently rescales the allowed step whenever sigma changes. That is
        # how the 96-iteration sigma-0.05 run pinned its lr to the 1e-5 floor on every
        # iteration.
        #
        # Why 0.4, and a correction to the 0.05 this block carried for one run: 0.05
        # was derived by matching the teacher's DESIRED_KL at sigma 0.598, which is a
        # unit conversion and not a step size. What actually matters is the ABSOLUTE
        # action-mean displacement the teacher was making. From its own logs
        # (kl 0.0086-0.0185 at sigma 0.598): dmu_rms = sigma*sqrt(kl/6) ~ 0.028 rad.
        # Reproducing that step at sigma 0.05 needs kl = 6*(0.028/0.05)^2 ~ 1.9, i.e.
        # ratio 0.56. The 0.05 value allowed dmu_rms = 0.0025 rad -- 11x smaller -- so
        # the schedule still dived on iteration 0 and never recovered.
        #
        # The second, independent trap was reachability. The recovery branch needs
        # kl < target/2. At the 1e-5 lr floor this run measured kl = 0.026-0.035, so
        # with target 0.015 the gate (kl < 0.0075) was physically unreachable and the
        # lr could never climb back. 0.4 gives target 0.96 and a gate at 0.48, safely
        # above that floor, and dmu_rms = 0.4*0.05 = 0.020 rad -- deliberately just
        # under the teacher's 0.028 rather than above it.
        rl_step_ratio = 0.4
        rl_desired_kl = 0.01
        # Hard ceiling on the action std, applied per forward pass. The teacher's
        # ActorCritic clamps actor_critic.std at 1.0 inside update_distribution;
        # the depth student owns an independent depth_actor_std parameter that
        # nothing would otherwise bound, while rl_entropy_coef below actively
        # pushes it up.
        rl_max_noise_std = 1.0

        # Encoder lr for the JOINT replay: FROZEN at 0.0 for the ARM-A control run.
        # It is the teacher's DAgger encoder and it is supposed to stay that way -- the
        # anchor term (below) exists precisely because the PPO gradient has no reason to
        # preserve it. Two checkpoints say the anchor alone is not enough: coef 10 held
        # the anchor loss down (1.144 -> 0.883 over long4) yet the weights still moved
        # GLOBAL_REL 8.6e-2 after 100 iterations and 1.2e-1 after 200, with
        # output_mlp.0.weight at 19% -- and the anchor metric could not show it because
        # it is a function of the visited states. A frozen encoder is also what makes
        # this run interpretable: every remaining change is attributable to the actor.
        rl_encoder_learning_rate = 0.0
        # Teacher anchor on the encoder replay: an L2 term between the frozen
        # scan_encoder latent and the replayed depth latent. Both sides are 32-D and
        # directly comparable (scan_encoder_dims ends at 32), and this is the SAME
        # objective DAgger trains the encoder with -- the sibling of the commented-out
        # update_depth_encoder call.
        #
        # Why 10 and not 1: the two gradients are not on the same scale. The surrogate
        # reaches the latent amplified by ~adv/sigma ~ 1/0.05 = 20, while a unit-norm
        # L2 anchor carries gradient magnitude 1. Measured at coef 1.0 over 8
        # iterations: anchor loss climbed 0.632 -> 1.135 LINEARLY (+0.07/iter, so
        # ~21 after 300 iterations against a teacher latent whose norm is O(3-5)),
        # with the encoder grad norm unchanged at 9-12. That is an unconstrained
        # encoder, i.e. the coef was doing nothing. 10 puts the two on the same
        # order of magnitude; raise further if the loss still climbs.
        # Set 0.0 to reproduce the anchor-free behaviour (encoders drifts freely).
        rl_encoder_anchor_coef = 10.0

        # Annealed ACTION-level BC anchor (the stage-B hybrid: teacher reward with
        # an imitation term decaying to zero). An L2 between the frozen teacher's
        # action mean and the student's PPO mean inside update_depth_rl, weight
        # rl_bc_coef -> rl_bc_coef_end over rl_bc_anneal_iters healthy iterations.
        # The teacher action is recomputed from the stored obs (true priv slice
        # spliced back, hist_encoding=False), so no rollout/storage change is
        # needed and the target is stationary. Off by default (0.0): every pure-RL
        # recipe above is unchanged until rl_bc_coef is raised. Suggested stage-B
        # start: 0.25, annealing to 0 over ~1500 iters, from a CONVERGED DAgger
        # ckpt with a FRESH critic (rl_critic lr 3e-5 magnitude, or freeze it for
        # the first 100-200 iters -- see the warm-start critic divergence lesson).
        # rl_bc_value_guard freezes the anneal clock on iterations whose value
        # loss exceeds guard x its recent EMA (warm-start critic divergence
        # brake); None/0 disables the guard.
        rl_bc_coef = 0.0
        rl_bc_coef_end = 0.0
        rl_bc_anneal_iters = 1500
        rl_bc_value_guard = 2.0

        # History-encoder lr, the update_dagger() term ported to this path. Matches
        # the teacher's hist_encoder_optimizer lr (2e-4): it is a clean L2
        # regression onto priv_encoder(true latent). It gets a SEPARATE optimiser
        # and is excluded from depth_rl_optimizer, so this is the only step the
        # head ever receives (the teacher's arrangement lets two optimisers step
        # it, doubling every update).
        rl_hist_encoder_learning_rate = 2.e-4

        # Minibatches for the depth-RL update. Kept at the teacher's 4 so every
        # iteration takes the teacher's 20 Adam steps on 6144-sample batches; the
        # encoder replay's conv activations are bounded by rl_replay_chunk instead
        # (gradient accumulation), so memory does NOT force a smaller batch.
        # Inflating this to 40 would give 200 steps/iteration -- 10x the teacher's
        # trust-region movement -- and pin the adaptive KL on min_learning_rate.
        rl_num_mini_batches = 4

        # Gradient-accumulation width of the encoder replay. update_depth_rl walks
        # each minibatch in slices of this many samples, back-propagates each slice
        # and steps the optimiser ONCE, so the update is identical to a whole-batch
        # step. Measured peaks with 1024 envs x 24 steps: 1.0 GB at 512, 1.5 GB at
        # 768, 2.3 GB at 1228, 2.9 GB at 1536 -- and the wall time is flat (~2.1 s
        # per epoch) because the replay is throughput-bound, not latency-bound.
        rl_replay_chunk = 512

        # Action width, as a SCALAR fill (12-D). Kept at 0.05 because that is where
        # the only depth-RL run that ever improved started (it settled at 0.106 and
        # reached max_goal_idx 0.11-0.16), and because a distilled student is a
        # deterministic mean-follower that was never trained to tolerate the
        # teacher's 0.598.
        # MEASURED COMPARISON, and a correction to the note this block used to carry:
        # 33 it at sigma 0.598 vs 96 it at sigma 0.05 gave the SAME flat
        # max_goal_idx (0.05-0.07) and the same episode length (~140), so sigma was
        # NOT what flattened the curve. What flattened it was that the actor never
        # moved: at a narrow sigma the fixed desired_kl is ~144x larger in KL terms,
        # so the lr hit its floor on iteration 0 and stayed there for 96/96
        # iterations. That is fixed by rl_step_ratio above; sigma itself only sets
        # the exploration width, so it stays at the value the good run used.
        rl_init_std = 0.05
        # ARM-A CONTROL: keep sigma an actual constant. The comment above used to claim
        # that sample noise "still drifts sigma slowly" -- it is neither slow nor
        # negligible: long4 moved the mean 0.05006 -> 0.08760 (+75%) in 217 iterations,
        # and the consequence is structural rather than cosmetic. kl_target is calibrated
        # as 6*step_ratio^2 at sigma 0.05 while KL = 6*(dmu/sigma)^2, so once sigma has
        # drifted to 0.088 the SAME dmu registers only 0.32x the KL and every later step
        # is allowed to be ~3x larger -- the trust region quietly widens by itself. That
        # is also why a frozen sigma and a fixed lr have to ship together: with sigma
        # pinned the KL target means what it says again.
        rl_freeze_std = True
        # The entropy term is the only EXPLICIT thing in the PPO objective that moves
        # sigma, and its sign in update_depth_rl (`loss -= coef * entropy`) maximises
        # entropy; at 0.01 it drove sigma monotonically up for 33 straight iterations.
        # It stays 0. rl_max_noise_std remains as a backstop for anything that clamps.
        rl_entropy_coef = 0.0
