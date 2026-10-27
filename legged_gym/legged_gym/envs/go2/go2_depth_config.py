"""go2_depth — one-stage camera policy: student observations, teacher rewards.

The release trains two networks (`go2_teacher` -> privileged scan policy, then
`go2_student` -> depth distillation of it). This config collapses that chain into
a SINGLE PPO run:

  * actor   : the deployable observation set — proprio + estimator-backed explicit
              state, with the 32-D depth-encoder latent fed in place of the 32-D
              height-scan latent (through `scandots_latent`, so it is literally the
              teacher's actor network). obs[:, 6:8] (the two heading command slots)
              are blanked: on hardware only the camera can supply heading.
  * critic  : unchanged privileged observation (asymmetric actor-critic).
  * reward  : the teacher's reward function, verbatim. Nothing is distilled — PPO on
              the teacher's shaping is what trains the camera policy.
  * encoder : trained jointly. PPO gradients flow through the latent into the CNN,
              and an auxiliary MSE anchors the latent to the actor's OWN stop-grad
              height-scan latent (`aux_scan_coef`), which is what makes a teacher
              warm start meaningful.

Cold start (stage 1): terrain row 0 for every env — rows 0-3 are hurdle-free but
still carry the wall + pit — and all depth degradations off, so the first thing
learned is "run at the wall off a clean image". Promote normally, then re-enable
`class depth` (stage 2) for the deployment-mirroring camera.

    # from scratch
    python train.py --exptid xxx-depth --task go2_depth --use_camera --delay \
        --headless --device cuda:0

    # warm start from the privileged teacher (same 114-D actor input; the encoder
    # is re-initialised, the optimizer state is dropped with a warning)
    python train.py --exptid xxx-depth --task go2_depth --resume --resumeid teacher \
        --use_camera --delay --headless --device cuda:0

`--use_camera` is what flips `depth_encoder.if_depth` on (helpers.py), so it is
mandatory. Do NOT set EP_USE_YAW=1: the encoder head then emits 34 dims and can no
longer stand in for the 32-D scan latent (the runner raises).
"""

from legged_gym.envs.go2.go2_student_config import (
    Go2StudentCfg, Go2StudentCfgPPO)


class Go2DepthCfg(Go2StudentCfg):

    class env(Go2StudentCfg.env):
        # Depth cannot resolve which of the two symmetric wall approaches is meant, and
        # this termination (inherited from the teacher) killed most approaches in the
        # distillation run. Lateral-deviation termination stays ON, so the corridor is
        # still enforced from depth alone.
        terminate_on_yaw_tracking = False

    class terrain(Go2StudentCfg.terrain):
        # Cold start: every env spawns on row 0. Promotion then climbs the same
        # curriculum the teacher used (rows 4+ add the hurdles), and demotion is left on
        # because it is the only thing that pulls a collapsed fleet back down.
        min_terrain_level = 0
        max_init_terrain_level = 0
        no_demotion = False

    class depth(Go2StudentCfg.depth):
        # Stage 1 = clean geometry, clean camera, no lag.
        #
        # `noise_scale_per_env` is switched OFF on purpose. While it is on, every env
        # draws s ~ U[smin, 1] and every degradation is scaled by 2s, so the zeros below
        # would silently come back for the noisier half of the fleet. Off => the
        # configured rates below are exactly what is applied.
        noise_scale_per_env = False
        # Latency in capture frames (20 ms each). Deploy pads ~80 ms; this is the knob
        # to restore in stage 2 ([4, 5] as in the student).
        latency_range = [0, 0]
        latency_scale_with_noise = False

        # All degradations off for stage 1 (see the class docstring for what to restore).
        empty_below_m = 0.0
        dis_noise = 0.0
        noise_gaussian = 0.0
        noise_dropout = 0.0
        pw_noise = False            # gates edge/blind-spot/blur/min-range as a block
        pw_hole_prob = 0.0
        pw_edge_prob = 0.0
        pw_blur_sigma = 0.0
        stereo_shadow = False
        self_geometry = False       # its own legs in view: enable with the rest of stage 2
        hole_fill = False           # per-frame CPU hole fill, pointless with no holes


class Go2DepthCfgPPO(Go2StudentCfgPPO):

    class runner(Go2StudentCfgPPO.runner):
        # 24 control steps = 4.8 encoder ticks per iteration (~1.2 s of experience).
        num_steps_per_env = 24
        max_iterations = 15000
        save_interval = 500

    class algorithm(Go2StudentCfgPPO.algorithm):
        # The encoder is replayed for every sample inside update(), so each extra epoch
        # costs another num_mini_batches worth of CNN forwards; 4 x 4 = 16 replays per
        # iteration is already ample for the latent to track the moving policy.
        learning_rate = 3.e-4
        num_learning_epochs = 4
        num_mini_batches = 4

    class depth_encoder(Go2StudentCfgPPO.depth_encoder):
        # <- the switch: train the encoder inside PPO instead of DAgger/BC.
        train_with_ppo = True
        # MSE(depth latent, stop-grad scan latent). Without it the latent's coordinate
        # frame is free, and a teacher warm start hands the actor a latent it has never
        # seen; with it the camera bottleneck lands where the privileged actor expects.
        aux_scan_coef = 1.0
        # Encoder base lr; it is a separate parameter group INSIDE the PPO optimizer and
        # is rescaled proportionally when the KL schedule moves the policy lr.
        learning_rate = 1.e-3
        # Run the encoder and the actor on the deployed observation (blank heading slots).
        zero_heading_obs = True
        # DAgger-only knob, inert on this path.
        action_noise_std = 0.0
